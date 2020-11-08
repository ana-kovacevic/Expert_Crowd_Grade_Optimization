# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 22:45:35 2020

@author: akovacevic
"""

### Set working directory
import os
os.chdir('F:\PROJEKTI\ONR_FON\Experiments\Credibility-Factors2020')
#### Import libreries
import warnings
warnings.simplefilter('ignore')

import sys
sys.path.append('F:\PROJEKTI\ONR_FON\Experiments\Expert-Crowd')

from Matrix_Factorization import ExplicitMF
#import Matrix_Factorization

import pandas as pd
import numpy as np

from Data_Prepare import prepare_crowd_data
from Data_Prepare import remap_answers
from Data_Prepare import prepare_expert_data
from Data_Prepare import get_aggregated_data
from Data_Prepare import create_ratings_and_mapping
from Data_Prepare import train_test_split
from Data_Prepare import calculate_sparsity
from Data_Prepare import get_user_ids_from_mapping

from Optimize_Matrix_Factorization import find_best_parms_for_ALS

from OutlierDetection import detect_outliers
from OutlierDetection import remove_outliers


from Clustering import create_cluster_experts
from Clustering import optimize_k_number
from Clustering import determnin_cluster_quality


from Weighting_And_Votes_Aggregation import get_voter_weights_by_distance
from Weighting_And_Votes_Aggregation import get_weights_by_cluster_quality
from Weighting_And_Votes_Aggregation import calculate_alternatives_scores_and_get_selection
from Weighting_And_Votes_Aggregation import get_top_N

from Result_Analysis import results_satisfaction
from Result_Analysis import results_satisfaction_points
#from Result_Analysis import result_distance_from_expert_alternatives
from Result_Analysis import results_basic_weighted_satisfaction
from Result_Analysis import results_basic_weighted_satisfaction_points


'''
     Create neccesery variables and result datasets  
'''
expert_weight = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
mask_test_size = 3
latent_factors = [ 10, 20, 30, 50, 100] #[20, 30] #
regularizations =  [0., 0.1, 1., 10., 100.] #[0., 0.1] #
regularizations.sort()
iter_array = [10, 50, 100, 150] #[1, 2, 5, 10, 25, 50, 100]
points_rank = [5, 4, 3, 2, 1]

res_satisfaction = pd.DataFrame(columns = ['Case','NoMethod', 'Overlap - Dist', 'Factorization',  'num_experts', 'num_factors'])
res_satisfaction_basic = pd.DataFrame(columns = ['Case', 'weight', 'Satisfaction',  'year', 'event'])

res_points = pd.DataFrame(columns = ['NoMethod', 'Overlap - Dist', 'Factorization',  'num_experts', 'num_factors'])
res_points_basic = pd.DataFrame(columns = ['Case', 'weight','abs_diff_mean', 'year', 'event'])

tsne_result = pd.DataFrame(columns = ['voter_id' ,	'voter',	'tsne-one', 	'tsne-two',	'Country',	'group', 'Country_list', 'year', 'event', 'num_experts', 'num_factors', 'clust_label'])
tsne_result_all = pd.DataFrame(columns = ['voter_id' ,	'voter',	'tsne-one', 	'tsne-two',	'Country',	'group', 'Country_list', 'year', 'event', 'num_clust', 'num_factors', 'clust_label'])


res_w_satisfaction = pd.DataFrame(columns = ['Method', 'Case', 'weight', 'Satisfaction',  'year', 'event', 'num_experts', 'num_factors'])
res_w_points = pd.DataFrame(columns = ['Method', 'Case', 'weight','abs_diff_mean', 'year', 'event', 'num_experts', 'num_factors' ])


data_folder = 'credibilitycoalition-credibility-factors2020\\'
data_folder_crowd = 'credibilitycoalition-2019-study\\'

crowd_file = 'CredCo_2019_Crowd Annotators-FULL.csv'
#crowd_file_subset = 'CredCo 2019 Study - CplusJ 2020 subset.csv'
crowd_map = pd.read_csv(data_folder + 'CredCo Study 2019 Crowd Annotators -simple.csv')
#crowd_all = pd.read_csv(data_folder_crowd + crowd_file)


### Read crowd data

crowd = pd.read_csv(data_folder_crowd + crowd_file)

df_crowd, alt_names, alternative_map = prepare_crowd_data(crowd)
df_crowd = remap_answers(df_crowd)

#df_crowd.dtypes
#type(alt_names[0])

df_expert = prepare_expert_data(data_folder, alternative_map)

df_expert_crowd = pd.concat([df_expert, df_crowd], ignore_index=True)

n_crowd = len(df_crowd['voter'].unique())
#df_expert_crowd['voter'].value_counts()

#a = df_crowd.sort_values(['voter', 'vote'], ascending = (True,True))

#df_crowd[df_crowd[['voter', 'vote']].duplicated()]


############# Aggregate data
crowd_agg = get_aggregated_data(df_crowd, alt_names)
expert_agg = get_aggregated_data(df_expert, alt_names)
#expert_agg = aggregate_experts(expert_agg[alt_names], points_rank, team_size, alt_names)
expert_crowd_agg = get_aggregated_data(df_expert_crowd, alt_names)


######## get sparse data and mapping data set of user id and alternative ids
#df_sparse = convert_data_to_sparse_and_create_mapping(df_expert_crowd)[0]

ratings, alts_lookup, voters_lookup = create_ratings_and_mapping(expert_crowd_agg, alt_names)
train, test = train_test_split(ratings, mask_test_size)
#print(df_sparse)
# Check sparsity of data

print("Sparsity of data is: {:.2f} %. ".format( calculate_sparsity(ratings)))

"""
################### Difine Factorisation Model 
-------------- Optimize number of factors based on MAE
"""

best_params_als = find_best_parms_for_ALS(latent_factors, regularizations, iter_array, train, test)
model_als = best_params_als['model']
num_factors = best_params_als['n_factors']

#### initialize a model
#model = implicit.als.AlternatingLeastSquares(factors = num_factors)
#model.fit(df_sparse.T)

##### User and alternative factor matrix 
#user_factors = model.user_factors
#alt_factors = model.item_factors
user_factors = model_als.user_vecs
alt_factors = model_als.item_vecs
r = user_factors.dot(alt_factors.T)
dense_all_agg = pd.DataFrame(r, columns = alt_names)  

#### extract expert and crowd ids for similarity
expert_ids = get_user_ids_from_mapping(voters_lookup, 'expert')
crowd_ids = get_user_ids_from_mapping(voters_lookup, 'crowd')

#sorter = voters_lookup['voter'] 
#sorterIndex = dict(zip(sorter,range(len(sorter))))

"""
######################################## CLUSTER Quality on All Data
"""

best_params_clust_all, silhouette_dict_all = optimize_k_number(user_factors, kmin = 2, kmax = 11)

all_centroids = create_cluster_experts (user_factors, best_params_clust_all['model'])[0]

#best_params_clust_all['model'].inertia_
num_clusters = best_params_clust_all['n_k']

clust_labels_all =  best_params_clust_all['model'].labels_

cluster_quality = determnin_cluster_quality(user_factors, voters_lookup, clust_labels_all, all_centroids)
mesures_included = ['entropy', 'N', 'compactenss']


"""
###################################### VOTER WEIGHTS CALCULATION
"""
#crowd_weights = get_voter_weights(crowd_agg[alt_names], expert_agg[alt_names])
crowd_weights = get_voter_weights_by_distance(crowd_agg[alt_names], expert_agg[alt_names], metrics = 'euclidian')
#crowd_weights_overlap = get_voter_overlapping(crowd_agg[alt_names], expert_agg[alt_names], team_size)

user_weights_sim = get_voter_weights_by_distance(user_factors[crowd_ids], user_factors[expert_ids], metrics = 'euclidian')

#user_weights_clust = get_voter_weights_by_distance(user_factors[crowd_ids], expert_centroids, metrics = 'euclidian')

#user_weights_outliers = get_voter_weights_by_distance(user_factors[crowd_ids], compact_experts, metrics = 'euclidian')

user_weights_clust_quality = get_weights_by_cluster_quality(voters_lookup, clust_labels_all, cluster_quality, mesures_included, weight_type = 'fuzzy', norm_name = 'l2' )

"""
########################################## VOTING CALCULATION (GET SELECTION OF ALTENATIVES BY DIFFERENT METHODS)
"""
####### Scocrs and Wining Alternatives in current EUROSONG setup
#
team_size = 5

all_scores, all_selection = calculate_alternatives_scores_and_get_selection(expert_crowd_agg[alt_names], team_size)

### Calcluate result with using all predictions
dense_all_scores, dense_all_selection = calculate_alternatives_scores_and_get_selection(dense_all_agg, team_size)

#### Socrs and Wining Alternatives in if only crowd voted
#
crowd_scores,  crowd_selection = calculate_alternatives_scores_and_get_selection(crowd_agg[alt_names], team_size)

 ### Calcluate result with using all predictions
dense_crowd_scores, dense_crowd_selection = calculate_alternatives_scores_and_get_selection(dense_all_agg.iloc[crowd_ids,:], team_size)


#### Scores and Wining alternatives if only experts voted
#
expert_scores,  expert_selection = calculate_alternatives_scores_and_get_selection(expert_agg[alt_names], team_size)

dense_expert_scores, dense_expert_selection = calculate_alternatives_scores_and_get_selection(dense_all_agg.iloc[expert_ids, :], team_size)
###### WEIGHTED results by distance
#
w_crowd_scores,  W_crowd_selection = calculate_alternatives_scores_and_get_selection(crowd_agg[alt_names], team_size, crowd_weights)
#wo_crowd_scores, WO_crowd_selection = calculate_alternatives_scores_and_get_selection(crowd_agg[alt_names], team_size, crowd_weights_overlap)

###### WEIGHTED results by similaritis from factors
#
sf_crowd_scores, SF_crowd_selection = calculate_alternatives_scores_and_get_selection(crowd_agg[alt_names],  team_size, user_weights_sim) #user_weights_sim)[1]

###### WEIGHTED results by distance from clusterd experts
# 
#cf_crowd_scores, CF_crowd_selection = calculate_alternatives_scores_and_get_selection(crowd_agg[alt_names],  team_size, user_weights_clust)

###### WEIGHTED WITH EXPERTS THAT DOESN'T CONTAIN OUTLIERS

#of_crowd_scores, OF_crowd_selection= calculate_alternatives_scores_and_get_selection(crowd_agg[alt_names], team_size, user_weights_outliers)

####### 
#runfile('F:/PROJEKTI/ONR_FON/Experiments/Expert-Crowd/Weighting_And_Votes_Aggregation.py', wdir='F:/PROJEKTI/ONR_FON/Experiments/Expert-Crowd')
cq_all_scores, cq_all_selection = calculate_alternatives_scores_and_get_selection(expert_crowd_agg[alt_names],  team_size, user_weights_clust_quality)    

W_selection = get_top_N(w_crowd_scores + expert_scores,  team_size)
SF_selection = get_top_N(sf_crowd_scores + expert_scores, team_size)
#CF_selection = get_top_N(cf_crowd_scores + expert_scores, team_size)
#OF_selection = get_top_N(of_crowd_scores + expert_scores, team_size)
DENS_selection = get_top_N(dense_crowd_scores + expert_scores, team_size)
all_dens_selection = get_top_N(dense_all_scores, team_size)

methods_dict = {'NoMethod': all_selection, 'Overlap - Dist': W_selection, 'Factorization' : SF_selection
            , 'Crowd-predicted' : DENS_selection, 'All-predicted' : all_dens_selection, 'clust_quality': cq_all_selection }

methods_list = ['NoMethod', 'Overlap - Dist', 'Factorization', 'Crowd-predicted', 'All-predicted', 'clust_quality']
scores = ['w_crowd_scores', 'sf_crowd_scores',  'dense_crowd_scores', 'dense_all_scores', 'cq_all_scores']


"""
################################################## RESULT ANALISYS
"""
#runfile('F:/PROJEKTI/ONR_FON/Experiments/Expert-Crowd/Result_Analysis.py', wdir='F:/PROJEKTI/ONR_FON/Experiments/Expert-Crowd')
#res_satisfaction = res_satisfaction.append( results_satisfaction(all_selection, expert_selection, crowd_selection, W_selection, SF_selection, CF_selection, OF_selection, expert_crowd_agg, expert_agg, crowd_agg, team_size, alt_names))
res_satisfaction = res_satisfaction.append( results_satisfaction(methods_dict, expert_crowd_agg, expert_agg, crowd_agg, team_size, alt_names))

#res_points_diff = results_satisfaction_points(all_selection, expert_selection, crowd_selection, W_selection, SF_selection, CF_selection, OF_selection, expert_crowd_agg, expert_agg ,crowd_agg, points_rank, alt_names, team_size)
res_points_diff = results_satisfaction_points(methods_dict, expert_crowd_agg, expert_agg ,crowd_agg, points_rank, alt_names, team_size)
#res_alts_distance = result_distance_from_expert_alternatives(all_selection, expert_selection, crowd_selection, W_selection,SF_selection, CF_selection, alts_lookup, alt_factors)

res_points = res_points.append( res_points_diff.pivot(index =  'Case', columns = 'Method', values = 'abs_diff_mean').reindex_axis(methods_list, axis=1))

#res_satisfaction['year'].fillna(year, inplace = True)
#res_satisfaction['event'].fillna(event_type, inplace = True)
#res_satisfaction['num_experts'].fillna(num_experts, inplace = True)
res_satisfaction['num_factors'].fillna(num_factors, inplace = True)



res_points['num_factors'].fillna(num_factors, inplace = True)

"""
############################## Weighted methodes
"""        
print(" ------------------------- Weighting method started ----------------------------------")
for cs in scores:
    for w in expert_weight:
        ponder_scores = ((1 - w) * eval(cs)) + (w * expert_scores)
        selected = get_top_N( ponder_scores,  team_size)
#  
    
        res_w_satisfaction = res_w_satisfaction.append( results_basic_weighted_satisfaction(selected, expert_crowd_agg, expert_agg, crowd_agg, team_size, alt_names))
        res_w_points_diff = results_basic_weighted_satisfaction_points(selected, expert_crowd_agg, expert_agg ,crowd_agg, points_rank, alt_names, team_size)[['Case',  'abs_diff_mean']]
    
    
        res_w_points = res_w_points.append (res_w_points_diff)
    
       
        res_w_satisfaction['weight'].fillna(w, inplace = True)
        res_w_satisfaction['Method'].fillna(cs, inplace = True)
        #res_w_satisfaction['num_experts'].fillna(num_experts, inplace = True)
        res_w_satisfaction['num_factors'].fillna(num_factors, inplace = True)
    
                
        
        res_w_points['weight'].fillna(w, inplace = True)
        res_w_points['Method'].fillna(cs, inplace = True)
        #res_w_points['num_experts'].fillna(num_experts, inplace = True)
        res_w_points['num_factors'].fillna(num_factors, inplace = True)


from datetime import datetime
now = datetime.now() # current date and time
date_time = now.strftime("%Y_%m_%d")

res_sat_string = 'results/result_satisfaction_'  + date_time  + '.csv' 
res_points_string = 'results/result_points_'  +  date_time  + '.csv'

#res_tsne_string = 'data_results/tsne_results_'  + date_time +  '.csv'
#res_tsne_string_all = 'data_results/tsne_results_all_clust_' + date_time + '.csv'

res_satisfaction.to_csv(res_sat_string)  
res_points.to_csv(res_points_string)
#tsne_result.to_csv(res_tsne_string)
#tsne_result_all.to_csv(res_tsne_string_all)

######## Write results with weighted method    
    
methods_dict = {'w_crowd_scores': 'Overlap - Dist', 'sf_crowd_scores' : 'Factorization', 'cf_crowd_scores' : 'Clust-Factors'
                , 'of_crowd_scores': 'Outliers-Factors' , 'dense_crowd_scores' : 'Crowd-predicted'
                , 'dense_all_scores' : 'All-predicted', 'cq_all_scores' : 'clust_quality'}      
res_w_satisfaction = res_w_satisfaction.replace({"Method": methods_dict})
res_w_points = res_w_points.replace({"Method": methods_dict})


res_w_sat_string = 'results/result_sat_weighted_methods_'  + date_time + '.csv' 
res_w_points_string = 'results/result_points_weighted_methods_'  + date_time + '.csv'

res_w_satisfaction.to_csv(res_w_sat_string)
res_w_points.to_csv(res_w_points_string)

#from Result_Analysis import pivot_result_for_analysis

for w in expert_weight:
    ponder_scores = ((1 - w) * crowd_scores) + (w * expert_scores)
    selection = get_top_N( ponder_scores,  team_size)
    
    #from Voting_Evaluation_Methods import satisfaction_overlap
    #satisfaction_overlap(selection, alt_names, expert_agg, team_size)
    
    res_satisfaction_basic = res_satisfaction_basic.append( results_basic_weighted_satisfaction(selection, expert_crowd_agg, expert_agg, crowd_agg, team_size, alt_names))
    res_points_diff = results_basic_weighted_satisfaction_points(selection, expert_crowd_agg, expert_agg ,crowd_agg, points_rank, alt_names, team_size)[['Case',  'abs_diff_mean']]
    
    
    res_points_basic = res_points_basic.append (res_points_diff)
    
    

    res_satisfaction_basic['weight'].fillna(w, inplace = True)
    

    res_points_basic['weight'].fillna(w, inplace = True)
        
        

#res_satisfaction_basic.to_csv('results/result_weighted.csv')
#res_points_basic.to_csv('results/result_points_weighted.csv')


pivoted_sat = pivot_result_for_analysis(res_w_sat_string, 'results/result_weighted.csv', 'Satisfaction' )
pivoted_points = pivot_result_for_analysis(res_w_points_string, 'results/result_points_weighted.csv', 'abs_diff_mean' )

pivoted_sat_string = 'results/pivoted_weighted_sat_method_' + date_time  + '.csv'
pivoted_points_string = 'results/pivoted_weighted_points_method_' + date_time  + '.csv'

pivoted_sat.to_csv(pivoted_sat_string)
pivoted_points.to_csv(pivoted_points_string)
