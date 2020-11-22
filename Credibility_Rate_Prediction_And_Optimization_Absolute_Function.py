# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 19:57:48 2020

@author: akovacevic
"""

import os
os.chdir('F:\PROJEKTI\ONR_FON\Experiments\Expert_Crowd_Grade_Optimization')
#### Import libreries
import warnings
warnings.simplefilter('ignore')
 
import sys
sys.path.append('F:\PROJEKTI\ONR_FON\Experiments\Expert-Crowd')

#from Matrix_Factorization import ExplicitMF
#import Matrix_Factorization

import pandas as pd
import numpy as np
import time


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

from scipy.optimize import minimize #,optimize
#from Optimize_Grades import objective_function_grades_absolute
#from Optimize_Grades import optimize_grade_absolute_dist
from Optimize_Grades import calculate_satisfaction_absolute
#from Optimize_Grades import nash_bargaining_solution
#from Optimize_Grades import expectation_maximization_nash
#from Optimize_Grades import maximization_kalai_smorodinsky
from Optimize_Grades import nash_solution
from Optimize_Grades import lambda_const

from Quick_Data_Exploration import plot_medians

from Evaluate_and_Results import nash_results
from Evaluate_and_Results import kalai_results
from Evaluate_and_Results import calculate_baseline_stats_satisfaction
from Evaluate_and_Results import avg_satisfaction_by_group
from Evaluate_and_Results import relative_detail_satisfaction_nash
from Evaluate_and_Results import relative_detail_satisfaction_kalai
from Evaluate_and_Results import relative_detail_satisfaction_baseline
from Evaluate_and_Results import relative_overall_satisfaction
from Evaluate_and_Results import add_median_variation

'''
    Read Data
'''
alternative_map, alt_names, df_crowd, _, _ , df_science, df_journal = read_data_credibility()
df_science['rate']= df_science['rate'].astype('float')
df_journal['rate']= df_journal['rate'].astype('float')

df_selected_expert = df_science # df_journal
expert_type = 'science'

alts_dict = dict(zip(alternative_map['alternative_id'] , alternative_map['alternative_name']))
#### create mapping of all avaible users
#voter_map = crete_voter_map([df_selected_expert, df_crowd])
#voter_map = crete_voter_map([df_crowd])

#### transacional data of expert and crowd that labeled same alternatives as experts
df_expert_crowd = pd.concat([df_selected_expert, df_crowd], ignore_index=True)
#n_crowd = len(df_crowd['voter'].unique())

############# Aggregate data
crowd_agg = get_aggregated_data(df_crowd, alt_names)
expert_agg = get_aggregated_data(df_selected_expert, alt_names)
#expert_agg = aggregate_experts(expert_agg[alt_names], points_rank, team_size, alt_names)
expert_crowd_agg = get_aggregated_data(df_expert_crowd, alt_names)


'''
     Create neccesery variables and result datasets  
'''

mask_test_size = 0.1
latent_factors =  [20, 30, 40, 50, 70, 100] #[20, 30]
regularizations = [0., 0.1, 0.3, 0.5, 0.7, 1., 10., 100.]  #[0.3, 0.5, 0.7]    # [0., 0.1] 
regularizations.sort()
iter_array = [10, 50, 100, 150] #[1, 2, 5, 10, 25, 50, 100]

alphas = [0.0, 0.5, 1.0] #[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#b = (1,5)    
#bnds = [b]   
#x0 = 1.0

'''
Create data for factorization
'''

ratings, alts_lookup, voters_lookup = create_ratings_and_mapping(expert_crowd_agg, alt_names)
#check that the mask of alternatives is same
assert(np.sum(np.array(alts_lookup['alternative']) != np.array(alts_lookup['alternative_id'])) == 0)
#ratings, alts_lookup, voters_lookup = create_ratings_and_mapping(crowd_agg, alt_names)
train, test = train_test_split(ratings, mask_test_size, 3)
#print(df_sparse)
voter_dict = dict(zip(voters_lookup['voter_id'], voters_lookup['voter']))
# Check that non of alternatives or user is not empty in train set
print(np.sort(np.count_nonzero(train, axis= 0)))
print(np.sort(np.count_nonzero(train, axis= 1)))

# Check sparsity of data
print("Sparsity of data is: {:.2f} %. ".format( calculate_sparsity(ratings)))
print("Sparsity of training data is: {:.2f} %. ".format( calculate_sparsity(train)))

##### replace voters name with ids in all dataframes
df_crowd = pd.merge( voters_lookup,df_crowd, how = 'inner', on = 'voter').drop('voter', axis = 1)
df_expert_crowd = pd.merge( voters_lookup, df_expert_crowd, how = 'inner', on = 'voter').drop('voter', axis = 1)
df_selected_expert = pd.merge(voters_lookup,  df_selected_expert, how = 'inner', on = 'voter').drop('voter', axis = 1)
crowd_agg = pd.merge(voters_lookup,  crowd_agg, how = 'inner', on = 'voter').drop('voter', axis = 1)
expert_agg = pd.merge(voters_lookup,  expert_agg, how = 'inner', on = 'voter').drop('voter', axis = 1)

"""
################### Difine Factorisation Model 
-------------- Optimize number of factors based on MAE
"""

start = time.time()
best_params_als = find_best_parms_for_ALS(latent_factors, regularizations, iter_array, train, test)
end = time.time()
print('Total time to find embeddings (in min): ', str((end - start)/60))
print(best_params_als)

model_als = best_params_als['model']
num_factors = best_params_als['n_factors']

#pd.DataFrame(best_params_als).to_csv('results/ALS_best_params_' + expert_type + '.csv')

user_factors = model_als.user_vecs
alt_factors = model_als.item_vecs

#pd.DataFrame(user_factors).to_csv('results/user_factors_without_experts_sci.csv')
#pd.DataFrame(alt_factors).to_csv('results/alts_factors_without_experts_sci.csv')

pd.DataFrame(user_factors).to_csv('results/user_factors_' + expert_type + '.csv')
pd.DataFrame(alt_factors).to_csv('results/alts_factors_' + expert_type + '.csv')
#r = user_factors.dot(alt_factors.T)
#dense_all_agg = pd.DataFrame(r, columns = alt_names)  
#user_factors =np.array( pd.read_csv('results/user_factors_' + expert_type + '.csv').drop('Unnamed: 0', axis = 1))
#alt_factors = np.array( pd.read_csv('results/alts_factors_' + expert_type + '.csv').drop('Unnamed: 0', axis = 1))


#### extract expert and crowd ids for similarity
expert_ids = get_user_ids_from_mapping(voters_lookup, 'expert')
crowd_ids = get_user_ids_from_mapping(voters_lookup, 'crowd')


"""
Create data for supervised learning
"""
final_train_data = create_train_data_for_predicting_rates(df_expert_crowd, user_factors, alt_factors, voters_lookup, alts_lookup)
#final_train_data = create_train_data_for_predicting_rates(df_crowd, user_factors, alt_factors, num_factors, voters_lookup, alts_lookup)
non_rated_combinations = create_test_data_for_predicting_rates(final_train_data, user_factors, alt_factors, num_factors, voters_lookup, alts_lookup)

test_combinations = non_rated_combinations[[ 'voter_id', 'alternative_id']]


'''
    Select data for model
'''

final_train_data = final_train_data.drop(['voter', 'voter_id', 'alternative_id'], axis=1)#.drop_duplicates()
non_rated_combinations = non_rated_combinations.drop([ 'voter_id',  'alternative_id'], axis=1)


"""
    Find Best Predictive model
"""
start = time.time()
all_pred, grids, best_mod = optimize_predictive_model_and_predict(final_train_data, non_rated_combinations, folds = 3)
all_exp_grades, all_crowd_grades = prepare_data_for_grade_optimization(all_pred, test_combinations, df_selected_expert, df_crowd, voters_lookup, expert_ids, crowd_ids)
end = time.time()
print('Total time to find embeddings (in min): ', str((end - start)/60))

#_, all_crowd_grades = prepare_data_for_grade_optimization(all_pred, test_combinations, df_selected_expert, df_crowd, voters_lookup, [228,229,230], crowd_ids)

#
all_exp_grades.to_csv('results/data_all_ ' + expert_type +'_expert_grades.csv')
all_crowd_grades.to_csv('results/data_all_crowd_grades_' + expert_type +'.csv')

#all_crowd_grades.to_csv('results/data_all_crowd_grades_without_experts.csv')
all_crowd_grades = pd.read_csv('results/data_all_crowd_grades_without_experts.csv').drop('Unnamed: 0', axis = 1)
## read predicted grades
#all_crowds_pred_alone = pd.read_csv('results/data_all_crowd_grades_without_experts.csv').drop('Unnamed: 0', axis = 1)
#all_crowds_pred_journal = pd.read_csv('results/data_all_crowd_grades_journal.csv').drop('Unnamed: 0', axis = 1)
#all_crowds_pred_science = pd.read_csv('results/data_all_crowd_grades_science.csv').drop('Unnamed: 0', axis = 1)

'''
Optimize grade absolute

'''
######## Read optimal grades or optimize grades
#result_optimization_abs = optimize_grade_absolute_dist(alt_names, all_exp_grades, all_crowd_grades, alphas )

#result_optm_abs = pd.read_csv('results/absolute_optimization_grades1.csv') 

df_alt_votes = get_aggregated_data(pd.concat([all_crowd_grades, all_exp_grades]), voters_lookup['voter_id'], 
                                   index_column = 'alternative_id', column= 'voter_id', value = 'rate')
#df_alt_votes = get_aggregated_data(all_crowd_grades, voters_lookup['voter_id'], index_column = 'alternative_id', column= 'voter_id', value = 'rate')



#result_optm_abs = optimize_grade_absolute_dist(df_alt_votes, expert_ids, crowd_ids, alphas ) 

result_optm_abs0 = pd.DataFrame(df_alt_votes['alternative_id'], columns=(['alternative_id']))
result_optm_abs1 = pd.DataFrame(df_alt_votes['alternative_id'], columns=(['alternative_id']))

result_optm_abs0['optimal_grade'] = df_alt_votes[crowd_ids].apply(lambda x: np.median(x), axis =1)
result_optm_abs0['alpha'] = 0.0

result_optm_abs1['optimal_grade']  = df_alt_votes[expert_ids].apply(lambda x: np.median(x), axis =1)
result_optm_abs1['alpha'] = 1.0

result_optm_abs = pd.concat([result_optm_abs0, result_optm_abs1])

result_optm_abs = calculate_satisfaction_absolute(df_alt_votes, result_optm_abs, expert_ids, crowd_ids)

del(result_optm_abs0)
del(result_optm_abs1)
#result_optm_abs.to_csv('results/absolute_optimization_grades_and_sat.csv')

#expert_satisfaction_fixed = result_optimization_abs['expert_sat'] #np.mean(5 - np.abs(expert_votes - vote_fixed))
#crowd_satisfaction_fixed = result_optimization_abs['crowd_sat'] #np.mean(5 - np.abs(crowd_votes - vote_fixed))



'''
################################ Results
'''
###### nash


cons = [{'type':'eq', 'fun': lambda_const}]
bnds = ((0.01, 0.99), (0.01, 0.99), (1, 5))


res_nash = nash_results(df_alt_votes , crowd_ids, expert_ids, cons, lambda_expert = 0.5)

#res_nash = nash_results(df_alt_votes, result_optm_abs , crowd_ids, expert_ids, lambda_expert = 0.5)

res_nash.to_csv('results/results_nash' + expert_type + ' .csv')   

###### kalai
res_kalai = kalai_results(df_alt_votes, result_optm_abs, crowd_ids, expert_ids)

res_kalai.to_csv('results/results_kalai_' + expert_type + ' .csv')

# res_kalai = pd.read_csv('results/results_kalai.csv').drop('Unnamed: 0', axis = 1)

####### Baseline methods

res_baseline = calculate_baseline_stats_satisfaction( df_alt_votes, crowd_ids, 
                                          expert_ids ,stats = ['np.mean', 'np.median', 'mode'])

# res_all = pd.merge(res_baseline, 
#                    res_kalai.rename(columns= {'vote' : 'kalai_optimal', 'crowd_sat': 'crowd_sat-kalai', 
#                                               'expert_sat' : 'expert_sat-kalai', 'satisfaction_area':'satisfaction_area-kalai',
#                                               'satisfaction_sum':'satisfaction_sum-kalai'}),
#                    on = 'alternative_id')

# res_all[['expert_sat-expert_median', 'expert_sat-kalai']]

# res_all['expert_sat-expert_median'] - res_all['expert_sat-kalai']

res_overal_sat = avg_satisfaction_by_group(res_kalai, res_nash, res_baseline).reset_index()
res_overal_sat.to_csv('results/results_overall_avg_satisfaction'+ expert_type +'.csv')
###### relative satisfaction calculation
# max_satisfaction = pd.DataFrame()
# max_satisfaction['alternative_id'] = result_optm_abs['alternative_id'].unique()

max_satisfaction = result_optm_abs[['alternative_id', 'crowd_sat', 'expert_sat']].groupby(by='alternative_id' ).agg('max').reset_index()
max_satisfaction = max_satisfaction.rename(columns = {'crowd_sat':'max_crowd_sat', 'expert_sat' : 'max_expert_sat'})
max_satisfaction['max_satisfaction_sum'] = max_satisfaction['max_crowd_sat'] + max_satisfaction['max_expert_sat']
max_satisfaction['max_satisfaction_area'] = max_satisfaction['max_crowd_sat'] * max_satisfaction['max_expert_sat']

# max_satisfaction['max_crowd_sat'] = result_optm_abs[result_optm_abs['alpha'] == 0.0]['crowd_sat'].reset_index().drop('index', axis = 1)
# max_satisfaction['max_expert_sat'] = result_optm_abs[result_optm_abs['alpha'] == 1.0]['expert_sat'].reset_index().drop('index', axis = 1)
# max_satisfaction['max_satisfaction'] = max_satisfaction['max_crowd_sat'] + max_satisfaction['max_expert_sat']


#sat_col = [col for col in res_kalai.columns if 'sat' in col and 'diff' not in col]
#maxsat_col = [col for col in max_satisfaction.columns if 'sat' in col]

###### add relative satisfection by each alternative
res_nash = relative_detail_satisfaction_nash(res_nash, max_satisfaction)
res_kalai = relative_detail_satisfaction_kalai(res_kalai, max_satisfaction)
res_baseline = relative_detail_satisfaction_baseline(res_baseline, max_satisfaction)

res_nash.to_csv('results/results_nash_' + expert_type +'.csv')
res_kalai.to_csv('results/results_kalai_' + expert_type +'.csv')
res_baseline.to_csv('results/results_baseline_'+ expert_type +'.csv')

 ## ----------------------------------------------------------------------------
######### SUMMARIZE RESULTS

res_relative_sat = relative_overall_satisfaction(res_nash, res_kalai, res_baseline, max_satisfaction)

res_relative_sat.to_csv('results/results_overall_relative_satisfaction'+ expert_type +'.csv')
# a = pd.merge(res_baseline[['alternative_id','crowd_sat-crowd_median']],
#          res_kalai[['alternative_id', 'crowd_sat']], on = 'alternative_id')

#################### Result analysis - lower uncertanty
    
#df_crowd_sample = df_crowd.groupby('vote', group_keys = False).apply(lambda x: x.sample(min(len(x),3)))


import matplotlib.pyplot as plt
import seaborn as sns


df_votes_crowd_alone = get_aggregated_data(all_crowds_pred_alone , voters_lookup['voter_id'], index_column = 'alternative_id', column= 'voter_id', value = 'rate')
df_votes_crowd_journal = get_aggregated_data(all_crowds_pred_journal , voters_lookup['voter_id'], index_column = 'alternative_id', column= 'voter_id', value = 'rate')
df_votes_crowd_science = get_aggregated_data(all_crowds_pred_science , voters_lookup['voter_id'], index_column = 'alternative_id', column= 'voter_id', value = 'rate')

journal_T = df_journal.pivot(index = 'alternative_id', columns = 'voter', values = 'rate').groupby('alternative_id').sum().reset_index()
journal_T['median'] = journal_T[['Journalism-1_expert', 'Journalism-2_expert',  'Journalism-3_expert']].apply(lambda x: np.median(x), axis = 1)
exp_votes = np.array(journal_T[['Journalism-1_expert', 'Journalism-2_expert',  'Journalism-3_expert']] )
exp_median = np.array(journal_T[['median']])
journal_T['avg_diff'] = np.mean(np.abs(exp_votes - exp_median), axis = 1)
journal_alts = list(journal_T[journal_T['avg_diff'] < 0.5]['vote'])
final_journal = df_journal[df_journal['vote'].isin(journal_alts)]

science_T = df_science.pivot(index = 'alternative_id', columns = 'voter', values = 'rate').groupby('alternative_id').sum().reset_index()
science_T['median'] = science_T[['Science-1_expert', 'Science-2_expert', 'Science-3_expert']].apply(lambda x: np.median(x), axis = 1)

exp_votes = np.array(science_T[['Science-1_expert', 'Science-2_expert', 'Science-3_expert']] )
exp_median = np.array(science_T[['median']])
science_T['avg_diff'] = np.mean(np.abs(exp_votes - exp_median), axis = 1)
science_alts = list(science_T[science_T['avg_diff'] < 0.5]['vote'])
final_science = df_science[df_science['vote'].isin(science_alts)]


dfs_for_plotting = [('Journal experts and ', df_journal, 'Crowd factors predictions', df_alt_votes)
                    ,('Journal experts and ', df_journal, 'Crowd and Journal factors predictions', df_alt_votes)
                    #,('Science experts and ', final_science, 'Crowd factors predictions', df_votes_crowd_alone)
                    #,('Science experts and ', final_science, 'Crowd and Science factors predictions', df_votes_crowd_science)
                    ]

for i in range(len(dfs_for_plotting)):
    parameters = dfs_for_plotting[i]
    plot_medians(parameters[0], df_crowd, parameters[1], parameters[3], crowd_ids, parameters[2])
    print('Finished: ', str(i))


# examp = np.array(df_alt_votes[crowd_ids])[0:4,0:5]
# examp = df_alt_votes.iloc[0:4,1:6].round()
# examp.shape
#df_votes = alt_by_crowd
#n_samples = i


# for i in range(sel[0].shape[1]):
#     print(np.median(sel[0][i]))


n_repetitions = 50
a = crowd_agg.T
a.dtypes

sel_crowd = df_crowd[df_crowd['alternative_id'].isin( list(df_selected_expert['alternative_id'].unique()))]

alt_by_crowd = get_aggregated_data(sel_crowd, list(sel_crowd['voter_id'].unique()),
                        index_column = 'alternative_id', column= 'voter_id', value = 'rate')

ind = alt_by_crowd.iloc[:,1:][alt_by_crowd.iloc[:,1:] >0 ].count(axis= 0 )
ind = ind[ind ==35].reset_index()
ind = list(ind['voter_id'])
alt_by_crowd = alt_by_crowd[ind]

res_variation= pd.DataFrame()
for i in range(1,78):
    res_variation['var_'+str(i)] = add_median_variation(alt_by_crowd, n_repetitions, i)


plot_data = pd.DataFrame(np.mean(res_variation)).reset_index().rename(columns=({'index': 'var', 0:"avg_var"}))
plt.figure(figsize=(30,10))
sns.barplot(x = plot_data['var'], y = plot_data['avg_var'])
plt.title('Average varioation')
#plt.ylim(0.2, 1.1)
plt.show()


crowd_by_alt = df_crowd.groupby('alternative_id').agg('count').reset_index().rename(columns = { 'rate':'crowd_number'})
expert_by_alt = df_selected_expert.groupby('alternative_id').agg('count').reset_index().rename(columns = { 'rate':'expert_number'})

num_by_alt = list(pd.merge(expert_by_alt, crowd_by_alt, how = 'inner', on = 'alternative_id', indicator = True)['alternative_id'])
df_crowd[df_crowd['alternative_id'].isin(num_by_alt)]

only_crowd_alts = num_by_alt[num_by_alt['_merge']== 'left_only'][['alternative_id', 'crowd_number']]
both_alts = num_by_alt[num_by_alt['_merge']== 'both'][['alternative_id', 'crowd_number', 'expert_number']]
