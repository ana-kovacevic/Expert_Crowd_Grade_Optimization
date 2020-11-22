# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 20:37:17 2020

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



from Data_Prepare import get_aggregated_data
from Data_Prepare import get_user_ids_from_mapping
from Data_Prepare import crete_alternatives_map


from Optimize_Grades import lambda_const
from Optimize_Grades import calculate_satisfaction_absolute

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




#alternative_map, alt_names, df_crowd, _, _ , df_science, df_journal = read_data_credibility()



def experiment_artifical_data(df_expert_crowd):
    df_expert_crowd = df_expert_crowd.rename(columns = {'votes' : 'rate', 'group': 'voter', 'case': 'alternative_name'})
    df_expert_crowd['voter'] = df_expert_crowd['voter'] +  '_' +  df_expert_crowd['id'].astype(str)
    df_expert_crowd['rate']= df_expert_crowd['rate'].astype('float')

    alternative_map = crete_alternatives_map(df_expert_crowd, alternative_name = 'alternative_name')
    alt_names = list(alternative_map['alternative_id'].unique())

    voter_lookup = df_expert_crowd.copy()
    voter_lookup['voter_id'] = voter_lookup.groupby('voter').ngroup()
    voter_lookup = voter_lookup[['voter', 'voter_id']].drop_duplicates().reset_index().drop('index', axis = 1)
    voter_lookup = voter_lookup.sort_values('voter_id')

    df_expert_crowd = pd.merge(df_expert_crowd, alternative_map, on = 'alternative_name')[['voter', 'alternative_id', 'rate']]
    df_expert_crowd = pd.merge(voter_lookup, df_expert_crowd,  on = 'voter').drop('voter', axis = 1)

    expert_ids = get_user_ids_from_mapping(voter_lookup, 'expert')
    crowd_ids = get_user_ids_from_mapping(voter_lookup, 'crowd')

    df_expert = df_expert_crowd[df_expert_crowd['voter_id'].isin(expert_ids)]
    df_crowd = df_expert_crowd[df_expert_crowd['voter_id'].isin(crowd_ids)]

    '''
    Optimize grade absolute
    
    '''

    df_alt_votes = get_aggregated_data(df_expert_crowd, voter_lookup['voter_id'], 
                                       index_column = 'alternative_id', column= 'voter_id', value = 'rate')
    
    
    result_optm_abs0 = pd.DataFrame(df_alt_votes['alternative_id'], columns=(['alternative_id']))
    result_optm_abs1 = pd.DataFrame(df_alt_votes['alternative_id'], columns=(['alternative_id']))
    
    result_optm_abs0['optimal_grade'] = df_alt_votes[crowd_ids].apply(lambda x: np.median(x), axis =1)
    result_optm_abs0['alpha'] = 0.0
    
    result_optm_abs1['optimal_grade']  = df_alt_votes[expert_ids].apply(lambda x: np.median(x), axis =1)
    result_optm_abs1['alpha'] = 1.0
    
    result_optm_abs = pd.concat([result_optm_abs0, result_optm_abs1])
    
    result_optm_abs = calculate_satisfaction_absolute(df_alt_votes, result_optm_abs, expert_ids, crowd_ids)
    
    # del(result_optm_abs0)
    # del(result_optm_abs1)


    '''
    ################################ Results
    '''
    ###### nash
    
    
    cons = [{'type':'eq', 'fun': lambda_const}]
    bnds = ((0.01, 0.99), (0.01, 0.99), (1, 10))
        
    #res_nash_extreme = nash_results(df_alt_votes , crowd_ids, expert_ids, cons, lambda_expert = 0.5)
    res_nash = nash_results(df_alt_votes , crowd_ids, expert_ids, cons, bnds, lambda_expert = 0.5)
    
    #res_nash.to_csv('results/results_nash'  + ' .csv')   
    
    ###### kalai
    #res_kalai_extreme = kalai_results(df_alt_votes, result_optm_abs, crowd_ids, expert_ids)
    res_kalai = kalai_results(df_alt_votes, result_optm_abs, crowd_ids, expert_ids)
    
    #res_kalai.to_csv('results/results_kalai_'  + ' .csv')
    
    # res_kalai = pd.read_csv('results/results_kalai.csv').drop('Unnamed: 0', axis = 1)
    
    ####### Baseline methods
    
    # res_baseline_extreme = calculate_baseline_stats_satisfaction( df_alt_votes, crowd_ids, 
    #                                           expert_ids ,stats = ['np.mean', 'np.median', 'mode'])
    
    res_baseline = calculate_baseline_stats_satisfaction( df_alt_votes, crowd_ids, 
                                              expert_ids ,stats = ['np.mean', 'np.median', 'mode'])
    
    
    
    #res_overal_sat_extreme = avg_satisfaction_by_group(res_kalai_extreme, res_nash_extreme, res_baseline_extreme).reset_index()
    res_overal_sat = avg_satisfaction_by_group(res_kalai, res_nash, res_baseline).reset_index()
    #res_overal_sat.to_csv('results/results_overall_avg_satisfaction'+ expert_type +'.csv')
    
    max_satisfaction = result_optm_abs[['alternative_id', 'crowd_sat', 'expert_sat']].groupby(by='alternative_id' ).agg('max').reset_index()
    max_satisfaction = max_satisfaction.rename(columns = {'crowd_sat':'max_crowd_sat', 'expert_sat' : 'max_expert_sat'})
    max_satisfaction['max_satisfaction_sum'] = max_satisfaction['max_crowd_sat'] + max_satisfaction['max_expert_sat']
    max_satisfaction['max_satisfaction_area'] = max_satisfaction['max_crowd_sat'] * max_satisfaction['max_expert_sat']
    
    
    ###### add relative satisfection by each alternative
    # res_nash_extreme = relative_detail_satisfaction_nash(res_nash_extreme, max_satisfaction)
    # res_kalai_extreme = relative_detail_satisfaction_kalai(res_kalai_extreme, max_satisfaction)
    # res_baseline_extreme = relative_detail_satisfaction_baseline(res_baseline_extreme, max_satisfaction)
    
    
    res_nash = relative_detail_satisfaction_nash(res_nash, max_satisfaction)
    res_kalai = relative_detail_satisfaction_kalai(res_kalai, max_satisfaction)
    res_baseline = relative_detail_satisfaction_baseline(res_baseline, max_satisfaction)
    
    # res_nash.to_csv('results/results_nash_' + expert_type +'.csv')
    # res_kalai.to_csv('results/results_kalai_' + expert_type +'.csv')
    # res_baseline.to_csv('results/results_baseline_'+ expert_type +'.csv')
    
     ## ----------------------------------------------------------------------------
    ######### SUMMARIZE RESULTS
    
    # res_relative_sat_ext = relative_overall_satisfaction(res_nash_extreme, res_kalai_extreme, res_baseline_extreme, max_satisfaction)
    res_relative_sat = relative_overall_satisfaction(res_nash, res_kalai, res_baseline, max_satisfaction)
    
    #res_relative_sat.to_csv('results/results_overall_relative_satisfaction'+ expert_type +'.csv')
    # a = pd.merge(res_baseline[['alternative_id','crowd_sat-crowd_median']],
    #          res_kalai[['alternative_id', 'crowd_sat']], on = 'alternative_id')
    
    #################### Result analysis - lower uncertanty
        
    #df_crowd_sample = df_crowd.groupby('vote', group_keys = False).apply(lambda x: x.sample(min(len(x),3)))
    res_kalai = pd.merge(alternative_map, res_kalai, on = 'alternative_id')
    res_nash = pd.merge(alternative_map, res_nash, on = 'alternative_id')
    res_baseline = pd.merge(alternative_map, res_baseline, on = 'alternative_id')
    
    # res_kalai_extreme = pd.merge(alternative_map, res_kalai_extreme, on = 'alternative_id')
    # res_nash_extreme = pd.merge(alternative_map, res_nash_extreme, on = 'alternative_id')

    return res_kalai, res_nash, res_baseline, res_overal_sat, res_relative_sat

'''
    Read Data
'''

df_expert_crowd_extreme= pd.read_csv('results/artifical_data_extreme.csv').drop('Unnamed: 0', axis = 1)
df_expert_crowd = pd.read_csv('results/artifical_data_example.csv').drop('Unnamed: 0', axis = 1)

res_kalai, res_nash, res_baseline, res_overal_sat, res_relative_sat = experiment_artifical_data(df_expert_crowd)

res_kalai_extreme, res_nash_extreme, res_baseline_extreme, res_overal_sat_extreme, res_relative_sat_extreme = experiment_artifical_data(df_expert_crowd_extreme)

def results_on_artifical_data(res_kalai, res_nash, res_baseline, extreme_flag):
    res_kalai['method'] = 'Kalai'
    res_nash['method'] = 'Nash'
    
    res_nash = res_nash.rename(columns = {'optimal_grade': 'vote'})
    
    std_df = res_kalai[['alternative_id', 'crowd_std', 'expert_std']]
    alternative_map = res_kalai[['alternative_name', 'alternative_id']]
    
    var_votes = [col for col in res_baseline.columns if 'sat' not in col and 'alternative' not in col and 'rel' not in col]
    var_exp_sat = [col for col in res_baseline.columns if 'expert_sat' in col] # and 'sat' in col ] #and 'alternative' not in col and 'rel' not in col]
    var_crd_sat = [col for col in res_baseline.columns if 'crowd_sat' in col] # and 'sat' in col ] #and 'alternative' not in col and 'rel' not in col]
    #rel_expert = [col for col in res_baseline.columns if 'rel_expert' in col]
    #rel_crowd = [col for col in res_baseline.columns if 'rel_crowd' in col]


    res_baseline_vote = pd.melt(res_baseline, id_vars='alternative_id', value_vars=var_votes,
                                var_name='method', value_name='vote')
    
    res_baseline_exp_sat = pd.melt(res_baseline, id_vars='alternative_id', 
                                   value_vars=var_exp_sat, var_name='method', value_name='expert_sat')
    
    res_baseline_exp_sat['method'] = res_baseline_exp_sat.apply(lambda x:  x['method'].split('-')[1], axis = 1)
    
    res_baseline_crd_sat = pd.melt(res_baseline, id_vars='alternative_id', 
                                   value_vars=var_crd_sat, var_name='method', value_name='crowd_sat')
    
    res_baseline_crd_sat['method'] = res_baseline_crd_sat.apply(lambda x:  x['method'].split('-')[1], axis = 1)
   
    
    res_base = pd.merge(res_baseline_vote, res_baseline_exp_sat, on = ['alternative_id', 'method'])
    res_base = pd.merge(res_base, res_baseline_crd_sat, on = ['alternative_id', 'method'])
    res_base['lambda_exp'] = np.nan
    res_base['satisfaction_area'] = res_base['expert_sat'] * res_base['crowd_sat']
    res_base['satisfaction_sum'] = res_base['expert_sat'] + res_base['crowd_sat']
    
    res_base = pd.merge(alternative_map, res_base, on = 'alternative_id')
    
    
    cols = ['alternative_name', 'alternative_id', 'lambda_exp', 'vote', 'expert_sat', 'crowd_sat', 'satisfaction_area', 'satisfaction_sum', 'method']
    res_kalai = res_kalai[cols]
    res_nash = res_nash[cols]
    res_base = res_base[cols]
    
    res_all = pd.concat([res_kalai, res_nash, res_base])
    res_all['diff_sat'] = res_all['expert_sat'] - res_all['crowd_sat']
    res_all['extreme'] = extreme_flag
    
    res_all = pd.merge(res_all, std_df, on = 'alternative_id')
    
    return res_all

a = results_on_artifical_data(res_kalai, res_nash, res_baseline, 0)
b = results_on_artifical_data(res_kalai_extreme, res_nash_extreme, res_baseline_extreme, 1)

res_all = pd.concat([a,b])
res_all.to_csv('results/all_methods_results_by_alt.csv')

'''
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