No docum# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 13:13:40 2020

@author: akovacevic
"""
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

from Data_Prepare import crete_alternatives_map
from Data_Prepare import crete_voter_map
from Data_Prepare import prepare_crowd_data
from Data_Prepare import remap_answers
from Data_Prepare import prepare_expert_data
from Data_Prepare import get_aggregated_data
from Data_Prepare import create_ratings_and_mapping
from Data_Prepare import train_test_split
from Data_Prepare import calculate_sparsity
from Data_Prepare import get_user_ids_from_mapping

from Optimize_Matrix_Factorization import find_best_parms_for_ALS

from PrepareData_SupervisedApproach import aggregate_experts_per_alternatives
from PrepareData_SupervisedApproach import calculate_differences


'''
    Read Data
'''
### Files locations

data_folder_2020 = 'credibilitycoalition-credibility-factors2020\\'
data_folder_2019 = 'credibilitycoalition-2019-study\\'

crowd_file_2019 = 'CredCo_2019_Crowd Annotators-FULL.csv'
crowd_file_2020 = 'CredCo 2019 Study - CplusJ 2020 subset.csv'
#crowd_file_pool = 'CredCo Study 2019 Crowd Annotators -simple.csv'

#crowd_map = pd.read_csv(data_folder + 'CredCo Study 2019 Crowd Annotators -simple.csv')

### Read crowd data
crowd_2020 = pd.read_csv(data_folder_2020 + crowd_file_2020)
crowd_2020 = crowd_2020.drop('Unnamed: 0', axis = 1)
crowd_all = pd.read_csv(data_folder_2019 + crowd_file_2019)
crowd_all = remap_answers(crowd_all)

##### save map of all alternatives
alternative_map = crete_alternatives_map(crowd_all)
alts_dict = dict(zip(alternative_map['alternative_id'] , alternative_map['media_url']))

#### Read and prepare expert data
df_expert = prepare_expert_data(data_folder_2020, alternative_map)
exp_alt = list(df_expert['vote'].unique()) # alts that experts gave opinion on
exp_urls = [alts_dict.get(e,'') for e in exp_alt]  # alts that experts gave opinion on

#### filter alternatives same as experts
crowd_2020 = crowd_2020[crowd_2020['media_url'].isin(exp_urls)]
merged_crowd = pd.merge(crowd_all, crowd_2020, how='left',left_on = ['annotator', 'media_url'], right_on = ['annotator', 'media_url'], indicator = True)
#### take only users that are not part of earlier 
crowd_rest = crowd_all[merged_crowd['_merge'] == 'left_only']

df_crowd, alt_names = prepare_crowd_data(crowd_all, alternative_map)
df_crowd = remap_answers(df_crowd)

df_crowd_2020, _ = prepare_crowd_data(crowd_2020, alternative_map)
df_crowd_2020 = remap_answers(df_crowd_2020)
#crowd_rest = prepare_crowd_data(crowd_rest, alternative_map)[0] 
#crowd_rest = remap_answers(crowd_rest)

#### create mapping of all avaible users
voter_map = crete_voter_map([df_expert, df_crowd])
voter_dict = dict(zip(voter_map['voter_id'], voter_map['voter']))

#### transacional data of expert and crowd that labeled same alternatives as experts
df_expert_crowd = pd.concat([df_expert, df_crowd], ignore_index=True)
#n_crowd = len(df_crowd['voter'].unique())

############# Aggregate data
crowd_agg = get_aggregated_data(df_crowd, alt_names)
expert_agg = get_aggregated_data(df_expert, alt_names)
#expert_agg = aggregate_experts(expert_agg[alt_names], points_rank, team_size, alt_names)
expert_crowd_agg = get_aggregated_data(df_expert_crowd, alt_names)


'''
    Aggregate expert ratings on different levels    

'''

#### exert - alternative ratings (on expert level aggregated to one expert or aggregated to group expert)
df_expert_stats = aggregate_experts_per_alternatives(df_expert, voter_map)
crowd_stats = aggregate_experts_per_alternatives(df_crowd_2020, voter_map)


df_votes = pd.merge(df_crowd, df_expert_stats.drop('voter', axis=1), how = "inner", left_on = 'vote', right_on ='vote' )
df_votes = pd.merge(df_votes, voter_map, how = 'inner', left_on= 'voter', right_on= 'voter')

#cols = ['voter_id', 'voter', 'vote', 'expert_id', 'expert_type',  'rate', 'expert_rate',  'average_rate', 'group_rate']
df_votes = df_votes.reindex(columns=['voter_id', 'voter', 'vote', 'expert_id', 'expert_type',  'rate', 'expert_rate',  'average_rate', 'group_rate'])
# df_crowd_alt =  pd.merge(df_expert_stats, df_crowd_2020, how='inner', on='vote', #left_on=None, right_on=None,
#          left_index=False, right_index=False, #sort=True,  #suffixes=('_x', '_y'), copy=True, indicator=False,
#          validate=None)


'''
     Create neccesery variables and result datasets  
'''

mask_test_size = 3
latent_factors = [20, 30] #[ 10, 20, 30, 50, 100] 
regularizations =  [0., 0.1] # [0., 0.1, 1., 10., 100.] 
regularizations.sort()
iter_array = [10, 50, 100, 150] #[1, 2, 5, 10, 25, 50, 100]


'''
!!!!!!!! TO do: proveriti da li ovaj voters_lookup radi dobro
'''
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

user_factors = model_als.user_vecs
alt_factors = model_als.item_vecs
r = user_factors.dot(alt_factors.T)
dense_all_agg = pd.DataFrame(r, columns = alt_names)  

#### extract expert and crowd ids for similarity
expert_ids = get_user_ids_from_mapping(voters_lookup, 'expert')
crowd_ids = get_user_ids_from_mapping(voters_lookup, 'crowd')


"""
Create data for supervised learning
"""
df_crowd_alt =  pd.merge(df_votes, voters_lookup, how='inner', on='voter', #left_on=None, right_on=None,
         left_index=False, right_index=False,suffixes=('_map', '_lookup'), #sort=True,  # copy=True, indicator=False,
         validate=None)


user_factors_df = pd.DataFrame(user_factors[crowd_ids])
col_users = ['UF' + str(x) for x in range(1,num_factors+1)]

user_factors_df.columns = col_users
user_factors_df['voter_id'] = crowd_ids

alt_factors_df = pd.DataFrame(alt_factors)
col_alts = ['AF' + str(x) for x in range(1,num_factors+1)]
alt_factors_df.columns = col_alts

alt_factors_df['alternative_id'] = alts_lookup['alternative_id']

#np.sum(alts_lookup['alternative_id'] != alts_lookup['alternative'])
len(df_crowd_alt['vote'].unique())

df_crowd_alt =  pd.merge(df_crowd_alt, user_factors_df, how='inner', left_on='voter_id_lookup', right_on='voter_id',  #on='voter_id',
         left_index=False, right_index=False, #sort=True,  #suffixes=('_x', '_y'), copy=True, indicator=False,
         validate=None)

df_crowd_alt =  pd.merge(df_crowd_alt, alt_factors_df, how='inner', # on='voter_id', 
                         left_on='vote', right_on='alternative_id',
         left_index=False, right_index=False, #sort=True,  #suffixes=('_x', '_y'), copy=True, indicator=False,
         validate=None)

final_data =calculate_differences(df_crowd_alt)

'''
    Select data for model
'''

final_data = final_data.drop(['voter', 'expert_id', 'vote', 'expert_type', 'rate',
                 'expert_rate', 'average_rate', 'group_rate', 'voter_id_lookup', 
                'abs_diff_each', 'abs_diff_group'], axis=1).drop_duplicates()
# 'abs_diff_each', 'abs_diff_one', 'abs_diff_group'
"""
    Linear Regression
"""

from sklearn.model_selection import train_test_split
X = final_data.loc[:, final_data.columns != 'Diff']
y = final_data['abs_diff_one']
   
    ###### train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)



from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

print(regr.coef_)  
 # The mean square error
np.mean((regr.predict(X_test) - y_test)**2)

# Explained variance score: 1 is perfect prediction
# and 0 means that there is no linear relationship
# between X and y.
regr.score(X_test, y_test)


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

