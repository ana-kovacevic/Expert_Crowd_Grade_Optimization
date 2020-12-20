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
#import time

from Load_credibility_data import read_data_credibility

#from Data_Prepare import crete_voter_map
from Data_Prepare import get_aggregated_data
from Data_Prepare import create_ratings_and_mapping
from Data_Prepare import get_user_ids_from_mapping


from Weighting_And_Votes_Aggregation import get_voter_weights_by_distance
from Weighting_And_Votes_Aggregation import calculate_alternatives_scores_and_get_selection

from Evaluate_and_Results import calculate_baseline_stats_satisfaction
from Evaluate_and_Results import satisfaction_calculation_weighted_methods
from Evaluate_and_Results import avg_satisfaction_by_group
from Evaluate_and_Results import get_min_and_max_satisfactions
from Evaluate_and_Results import relative_detail_satisfaction_nash
from Evaluate_and_Results import relative_detail_satisfaction_kalai
from Evaluate_and_Results import relative_detail_satisfaction_baseline
from Evaluate_and_Results import relative_overall_satisfaction


'''
    Read Data
'''
alternative_map, alt_names, df_crowd, _, _ , df_science, df_journal = read_data_credibility()
df_science['rate']= df_science['rate'].astype('float')
df_journal['rate']= df_journal['rate'].astype('float')

df_selected_expert =  df_journal #df_science #
expert_type = 'journal' #'science'  #

alts_dict = dict(zip(alternative_map['alternative_id'] , alternative_map['alternative_name']))


#### transacional data of expert and crowd that labeled same alternatives as experts
df_expert_crowd = pd.concat([df_selected_expert, df_crowd], ignore_index=True)
#n_crowd = len(df_crowd['voter'].unique())

############# Aggregate data
crowd_agg = get_aggregated_data(df_crowd, alt_names)
expert_agg = get_aggregated_data(df_selected_expert, alt_names)
expert_crowd_agg = get_aggregated_data(df_expert_crowd, alt_names)


'''
     Create neccesery variables and result datasets  
'''

#mask_test_size = 0.1
#latent_factors =  [20, 30, 40, 50, 70, 100] #[20, 30]
#regularizations = [0., 0.1, 0.3, 0.5, 0.7, 1., 10., 100.]  #[0.3, 0.5, 0.7]    # [0., 0.1] 
#regularizations.sort()
#iter_array = [10, 50, 100, 150] #[1, 2, 5, 10, 25, 50, 100]

#alphas = [0.0, 0.5, 1.0] #[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
max_grade = 5
#b = (1,5)    
#bnds = [b]   
#x0 = 1.0

'''
Create data for factorization
'''

_, alts_lookup, voters_lookup = create_ratings_and_mapping(expert_crowd_agg, alt_names)

voter_dict = dict(zip(voters_lookup['voter_id'], voters_lookup['voter']))

##### replace voters name with ids in all dataframes
df_crowd = pd.merge( voters_lookup,df_crowd, how = 'inner', on = 'voter').drop('voter', axis = 1)
df_expert_crowd = pd.merge( voters_lookup, df_expert_crowd, how = 'inner', on = 'voter').drop('voter', axis = 1)
df_selected_expert = pd.merge(voters_lookup,  df_selected_expert, how = 'inner', on = 'voter').drop('voter', axis = 1)
crowd_agg = pd.merge(voters_lookup,  crowd_agg, how = 'inner', on = 'voter').drop('voter', axis = 1)
expert_agg = pd.merge(voters_lookup,  expert_agg, how = 'inner', on = 'voter').drop('voter', axis = 1)

"""
################### Read factors  
-------------- read factors obtained by previous experiments
"""

user_factors =np.array( pd.read_csv('results/user_factors_' + expert_type + '.csv').drop('Unnamed: 0', axis = 1))
alt_factors = np.array( pd.read_csv('results/alts_factors_' + expert_type + '.csv').drop('Unnamed: 0', axis = 1))


#### extract expert and crowd ids for similarity
expert_ids = get_user_ids_from_mapping(voters_lookup, 'expert')
crowd_ids = get_user_ids_from_mapping(voters_lookup, 'crowd')



"""
    Read all predictions
"""

all_exp_grades = pd.read_csv('results/data_all_ ' + expert_type +'_expert_grades.csv').drop('Unnamed: 0', axis = 1)
all_crowd_grades = pd.read_csv('results/data_all_' + expert_type +'_crowd_grades.csv').drop('Unnamed: 0', axis = 1)

'''
Aggregate grades based on INDIVIDUAL WEIGHTS

'''
######## Read optimal grades or optimize grades
result_optm_abs = pd.read_csv('results/absolute_optimization_grades_and_sat_' + expert_type + '.csv').drop('Unnamed: 0', axis = 1)

## pivot vote data for each alternative
df_alt_votes = get_aggregated_data(pd.concat([all_crowd_grades, all_exp_grades]), voters_lookup['voter_id'], 
                                   index_column = 'alternative_id', column= 'voter_id', value = 'rate')
## pivot vote data for each voter
df_voter_votes = get_aggregated_data(pd.concat([all_crowd_grades, all_exp_grades]), alts_lookup['alternative_id'], 
                                   index_column = 'voter_id', column= 'alternative_id', value = 'rate')

#df_alt_votes = get_aggregated_data(all_crowd_grades, voters_lookup['voter_id'], index_column = 'alternative_id', column= 'voter_id', value = 'rate')

crd_user_weights = get_voter_weights_by_distance(user_factors[crowd_ids], user_factors[expert_ids])[1]
W_crowd_selection = calculate_alternatives_scores_and_get_selection(
                    df_voter_votes[df_voter_votes['voter_id'].isin(crowd_ids)][alt_names], 0, 
                    crowd_weights = crd_user_weights)[0]


all_user_weights = get_voter_weights_by_distance(user_factors, user_factors[expert_ids])[1]
W_all_selection = calculate_alternatives_scores_and_get_selection(
                  df_voter_votes[alt_names], 0, 
                  crowd_weights = all_user_weights)[0]

weighted_grades = {'w_crowd_grades' : W_crowd_selection, 'w_all_grades': W_all_selection}

'''
################################ Results
'''
###### nash
#cons = [{'type':'eq', 'fun': lambda_const}]
#bnds = ((0.01, 0.99), (0.01, 0.99), (1, 5))


#res_nash = nash_results(df_alt_votes, max_grade, crowd_ids, expert_ids, cons, bnds, lambda_expert = 0.5)
#res_nash = nash_results(df_alt_votes, result_optm_abs , crowd_ids, expert_ids, lambda_expert = 0.5)
#res_nash.to_csv('results/results_nash' + expert_type + '.csv')   
res_nash = pd.read_csv('results/results_nash_' + expert_type + '.csv').drop('Unnamed: 0', axis = 1)

###### kalai
#res_kalai = kalai_results(df_alt_votes, result_optm_abs,max_grade, crowd_ids, expert_ids)
#res_kalai.to_csv('results/results_kalai_' + expert_type + '.csv')

res_kalai = pd.read_csv('results/results_kalai_' + expert_type + '.csv').drop('Unnamed: 0', axis = 1)

####### Baseline methods

res_baseline = pd.read_csv('results/results_baseline_'+ expert_type +'.csv').drop('Unnamed: 0', axis = 1)

#### weighted method
res_weighted = satisfaction_calculation_weighted_methods(df_alt_votes, max_grade, crowd_ids, expert_ids, weighted_grades)

res_overal_sat = avg_satisfaction_by_group(res_kalai, res_nash, res_baseline, res_weighted).reset_index()
res_overal_sat.to_csv('results/results_overall_avg_satisfaction_'+ expert_type +'.csv')


###### relative satisfaction calculation

min_satisfaction, max_satisfaction, ref_satisfaction =  get_min_and_max_satisfactions(result_optm_abs)


###### add relative satisfection by each alternative and gains
#res_nash = relative_detail_satisfaction_nash(res_nash, max_satisfaction)
#res_kalai = relative_detail_satisfaction_kalai(res_kalai, max_satisfaction)
#res_baseline = relative_detail_satisfaction_baseline(res_baseline, max_satisfaction)
res_weighted = relative_detail_satisfaction_baseline(res_weighted, max_satisfaction)

# res_nash['gain_ratio'] = pd.merge(ref_satisfaction, res_nash, on = 'alternative_id').apply( 
#         lambda x: np.abs( ( ( x['lambda_exp']*x['max_expert_sat'] + (1 - x['lambda_exp']) * x['min_expert_sat'])/x['max_expert_sat']) 
#                     - ((x['lambda_exp']*x['min_crowd_sat'] + (1 - x['lambda_exp']) * x['max_crowd_sat'])/x['max_crowd_sat']))
#         , axis = 1)
    
# res_kalai['gain_ratio'] = pd.merge(ref_satisfaction, res_kalai, on = 'alternative_id').apply( 
#     lambda x: np.abs(( ( x['lambda_exp']*x['max_expert_sat'] + (1 - x['lambda_exp']) * x['min_expert_sat'])/x['max_expert_sat']) 
#                 - ((x['lambda_exp']*x['min_crowd_sat'] + (1 - x['lambda_exp']) * x['max_crowd_sat'])/x['max_crowd_sat']))
#     , axis = 1)

# res_nash.to_csv('results/results_nash_' + expert_type +'.csv')
# res_kalai.to_csv('results/results_kalai_' + expert_type +'.csv')
#res_baseline.to_csv('results/results_baseline_'+ expert_type +'.csv')
res_weighted.to_csv('results/results_weighted_'+ expert_type +'.csv')

 ## ----------------------------------------------------------------------------
######### SUMMARIZE RESULTS

res_relative_sat = relative_overall_satisfaction(res_nash, res_kalai, res_baseline, res_weighted, ref_satisfaction)
res_relative_sat = res_relative_sat.rename(columns = ({'crowd_sat': 'rel-crowd_sat', 'expert_sat':'rel-expert_sat', 
                                                       'satisfaction_area' : 'rel-satisfaction_area',
                                                       'satisfaction_sum': 'rel-satisfaction_sum'}))


all_sum_res = pd.merge(res_relative_sat, res_overal_sat, on = 'method')
all_sum_res = all_sum_res.drop('index', axis = 1)

all_sum_res.to_csv('results/results_overall_relative_'+ expert_type +'.csv')

#################### Result analysis - lower uncertanty
    
#df_crowd_sample = df_crowd.groupby('vote', group_keys = False).apply(lambda x: x.sample(min(len(x),3)))

##################################### Plots
'''
import matplotlib.pyplot as plt
import seaborn as sns


#df_votes_crowd_alone = get_aggregated_data(all_crowds_pred_alone , voters_lookup['voter_id'], index_column = 'alternative_id', column= 'voter_id', value = 'rate')
#df_votes_crowd_journal = get_aggregated_data(all_crowds_pred_journal , voters_lookup['voter_id'], index_column = 'alternative_id', column= 'voter_id', value = 'rate')
#df_votes_crowd_science = get_aggregated_data(all_crowds_pred_science , voters_lookup['voter_id'], index_column = 'alternative_id', column= 'voter_id', value = 'rate')

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
'''