# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 19:57:48 2020

@author: akovacevic
"""

import os
os.chdir('F:\PROJEKTI\ONR_FON\Experiments\Credibility-Factors2020')
#### Import libreries
import warnings
warnings.simplefilter('ignore')
 
import sys
sys.path.append('F:\PROJEKTI\ONR_FON\Experiments\Expert-Crowd')

#from Matrix_Factorization import ExplicitMF
#import Matrix_Factorization

import pandas as pd
import numpy as np

from Load_credibility_data import read_data_credibility


from Data_Prepare import crete_voter_map
from Data_Prepare import get_aggregated_data
from Data_Prepare import create_ratings_and_mapping
from Data_Prepare import train_test_split
from Data_Prepare import calculate_sparsity
from Data_Prepare import get_user_ids_from_mapping

from Optimize_Matrix_Factorization import find_best_parms_for_ALS

from PrepareData_SupervisedApproach import create_train_data_for_predicting_rates
from PrepareData_SupervisedApproach import create_test_data_for_predicting_rates
from PrepareData_SupervisedApproach import prepare_data_for_grade_optimization

from Find_Best_Regression_Model import optimize_predictive_model_and_predict


from Optimize_Grades import objective_function_grades_absolute
from Optimize_Grades import optimize_grade_absolute_dist
from Optimize_Grades import nash_bargaining_solution
from Optimize_Grades import expectation_maximization_nash
from Optimize_Grades import maximization_kalai_smorodinsky



'''
    Read Data
'''
alternative_map, alt_names, df_crowd, _, _ , df_science, df_journal = read_data_credibility()

alts_dict = dict(zip(alternative_map['alternative_id'] , alternative_map['media_url']))
#### create mapping of all avaible users
voter_map = crete_voter_map([df_science, df_crowd])
voter_dict = dict(zip(voter_map['voter_id'], voter_map['voter']))

#### transacional data of expert and crowd that labeled same alternatives as experts
df_expert_crowd = pd.concat([df_science, df_crowd], ignore_index=True)
#n_crowd = len(df_crowd['voter'].unique())

############# Aggregate data
crowd_agg = get_aggregated_data(df_crowd, alt_names)
expert_agg = get_aggregated_data(df_science, alt_names)
#expert_agg = aggregate_experts(expert_agg[alt_names], points_rank, team_size, alt_names)
expert_crowd_agg = get_aggregated_data(df_expert_crowd, alt_names)


'''
     Create neccesery variables and result datasets  
'''

mask_test_size = 3
latent_factors = [20, 30] #[ 10, 20, 30, 50, 100] 
regularizations =  [0., 0.1] # [0., 0.1, 1., 10., 100.] 
regularizations.sort()
iter_array = [10, 50, 100, 150] #[1, 2, 5, 10, 25, 50, 100]

alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#b = (1,5)    
#bnds = [b]   
#x0 = 1.0

'''
Create data for factorization
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
#r = user_factors.dot(alt_factors.T)
#dense_all_agg = pd.DataFrame(r, columns = alt_names)  

#### extract expert and crowd ids for similarity
expert_ids = get_user_ids_from_mapping(voters_lookup, 'expert')
crowd_ids = get_user_ids_from_mapping(voters_lookup, 'crowd')


"""
Create data for supervised learning
"""
final_train_data = create_train_data_for_predicting_rates(df_expert_crowd, user_factors, alt_factors, num_factors, voters_lookup, alts_lookup)
non_rated_combinations = create_test_data_for_predicting_rates(final_train_data, user_factors, alt_factors, num_factors, voters_lookup, alts_lookup)

test_combinations = non_rated_combinations[[ 'voter_id', 'alternative_id']]


'''
    Select data for model
'''

final_train_data = final_train_data.drop(['voter', 'voter_id', 'vote', 'alternative_id'], axis=1)#.drop_duplicates()
non_rated_combinations = non_rated_combinations.drop([ 'voter_id',  'alternative_id'], axis=1)



"""
    Find Best Predictive model
"""
all_pred, grids, best_mod = optimize_predictive_model_and_predict(final_train_data, non_rated_combinations, folds = 3)



all_exp_grades, all_crowd_grades = prepare_data_for_grade_optimization(all_pred, test_combinations, df_science, df_crowd, voters_lookup, expert_ids, crowd_ids)


'''
Optimize grade absolute

'''

result_optimization_abs = optimize_grade_absolute_dist(alt_names, all_exp_grades, all_crowd_grades, alphas )

expert_satisfaction_fixed = np.mean(5 - np.abs(expert_votes - vote_fixed))
crowd_satisfaction_fixed = np.mean(5 - np.abs(crowd_votes - vote_fixed))

print(expert_satisfaction_fixed, crowd_satisfaction_fixed, expert_satisfaction_fixed * crowd_satisfaction_fixed)

print(nash_bargaining_solution(expert_satisfaction_fixed, crowd_satisfaction_fixed))

results = expectation_maximization_nash(expert_votes, crowd_votes, lambda_expert, verbose=False)
results

results = maximization_kalai_smorodinsky(expert_votes, crowd_votes)
results



#result_data = optimize_grade_square_dist(alt_names, all_exp_grades, all_crowd_grades, alphas, x0, bnds )

### Add satisfaction to results
#result_data = calculate_avg_distance_from_optimal(all_exp_grades, all_crowd_grades, result_data, alphas)