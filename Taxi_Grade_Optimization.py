# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 20:34:54 2021

@author: akovacevic
"""

import os
os.chdir('F:\PROJEKTI\ONR_FON\Experiments\Expert_Crowd_Grade_Optimization')

#### Import libreries
import warnings
warnings.simplefilter('ignore')
 

import pandas as pd
import numpy as np
#import time

from Load_taxi_data import Load_TX_Data
#from Prepare_Data import get_user_ids_from_mapping

from Optimize_Grades import calculate_satisfaction_absolute
from Optimize_Grades import lambda_const
#from Optimize_Grades import optimize_grade_absolute_dist

from Evaluate_and_Results import nash_results
from Evaluate_and_Results import kalai_results
from Evaluate_and_Results import calculate_baseline_stats_satisfaction
from Evaluate_and_Results import get_min_and_max_satisfactions

from Evaluate_and_Results import relative_detail_satisfaction_nash
from Evaluate_and_Results import relative_detail_satisfaction_kalai
from Evaluate_and_Results import relative_detail_satisfaction_baseline

# import sys
# sys.path.append('F:\PROJEKTI\ONR_FON\Experiments\Expert-Crowd')

# import Clustering as clust
from Cluster_Votes import optimize_k_number
from Cluster_Votes import create_cluster_experts
from Cluster_Votes import determnin_cluster_quality

# expert_type = 'traffic'
# expert_type = 'all'
expert_type = 'driver'
data_dict = Load_TX_Data(expert_type = expert_type)


df_selected_expert = data_dict['df_selected_expert']
#df_crowd = data_dict['df_crowd'] 

expert_ids = data_dict['expert_ids']
crowd_ids = data_dict['crowd_ids']


#####
df_alt_votes = data_dict['df_alt_votes']

df_user_votes = df_alt_votes.loc[:, df_alt_votes.columns != 'question_id'].T.reset_index()
df_expert = df_user_votes[df_user_votes['voter_id'].isin(expert_ids)]
df_expert = df_expert.dropna()


non_na_ids = df_expert['voter_id']
df_expert = df_expert.loc[:, df_expert.columns != 'voter_id']


voters_lookup = data_dict['voter_map'][data_dict['voter_map']['voter_id'].isin(non_na_ids)]


#df_user_votes[df_user_votes['voter_id'].isin(expert_ids)]
#data_dict['question_map']

max_grade = 3
# alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#alphas = [0.0, 0.5, 1.0]


"""
        #########################################  CLUSTERING 
       
"""
        ####### Optimize num of clusters 
        
    
best_params_clust, silhouette_dict = optimize_k_number(df_expert, kmin = 2, kmax = 11)
#best_params_clust, silhouette_dict = optimize_k_number(user_factors, kmin = 2, kmax = 11)


expert_centroids = create_cluster_experts(df_expert, best_params_clust['model'])[0]
num_experts = best_params_clust['n_k']

    
clust_labels =  best_params_clust['model'].labels_


cluster_quality = determnin_cluster_quality(df_expert, voters_lookup, clust_labels, expert_centroids)
mesures_included = ['entropy', 'N', 'compactenss']



'''
    Optimize grade 
'''
#opt_res = optimize_grade_absolute_dist(df_alt_votes, expert_ids, crowd_ids, alphas, 'question_id' ) 


result_optm_abs0 = pd.DataFrame(df_alt_votes['question_id'], columns=(['question_id']))
result_optm_abs1 = pd.DataFrame(df_alt_votes['question_id'], columns=(['question_id']))

result_optm_abs0['optimal_grade'] = df_alt_votes[crowd_ids].apply(lambda x: np.nanmedian(x), axis =1)
result_optm_abs0['alpha'] = 0.0

result_optm_abs1['optimal_grade']  = df_alt_votes[expert_ids].apply(lambda x: np.nanmedian(x), axis =1)
result_optm_abs1['alpha'] = 1.0

result_optm_abs = pd.concat([result_optm_abs0, result_optm_abs1])

result_optm_abs = calculate_satisfaction_absolute(df_alt_votes, result_optm_abs, max_grade, expert_ids, crowd_ids, 'question_id')

#result_optm_abs.to_csv('results_tx/absolute_optimization_grades_and_sat_' + expert_type + '.csv')
del(result_optm_abs0)
del(result_optm_abs1)


'''
    Nash solution
'''
cons = [{'type':'eq', 'fun': lambda_const}]
bnds = ((0.01, 0.99), (0.01, 0.99), (1, max_grade))

res_nash = nash_results(df_alt_votes, max_grade, crowd_ids, expert_ids, cons, bnds, lambda_expert = 0.5, alt_attribute='question_id')
res_nash = res_nash[['question_id','lambda_exp', 'optimal_grade', 'expert_sat', 'crowd_sat', 'satisfaction_area', 'satisfaction_sum']]

'''
    Kali Solution
'''
res_kalai = kalai_results(df_alt_votes, result_optm_abs, max_grade, crowd_ids, expert_ids, 'question_id')
res_kalai = res_kalai[['question_id','lambda_exp', 'vote', 'expert_sat', 'crowd_sat', 'satisfaction_area', 'satisfaction_sum']]

'''
   Baseline methods
'''
res_baseline = calculate_baseline_stats_satisfaction( df_alt_votes, max_grade, crowd_ids, 
                                          expert_ids ,stats = ['np.nanmean', 'np.nanmedian', 'mode'])

res_baseline = res_baseline[['question_id', 'crowd_mean', 'expert_mean', 'mean', 'crowd_median',
       'expert_median', 'median', 'crowd_majority', 'expert_majority',
       'majority', 'expert_sat-crowd_mean', 'crowd_sat-crowd_mean',
       'satisfaction_area-crowd_mean', 'satisfaction_sum-crowd_mean',
       'expert_sat-expert_mean', 'crowd_sat-expert_mean',
       'satisfaction_area-expert_mean', 'satisfaction_sum-expert_mean',
       'expert_sat-mean', 'crowd_sat-mean', 'satisfaction_area-mean',
       'satisfaction_sum-mean', 'expert_sat-crowd_median',
       'crowd_sat-crowd_median', 'satisfaction_area-crowd_median',
       'satisfaction_sum-crowd_median', 'expert_sat-expert_median',
       'crowd_sat-expert_median', 'satisfaction_area-expert_median',
       'satisfaction_sum-expert_median', 'expert_sat-median',
       'crowd_sat-median', 'satisfaction_area-median',
       'satisfaction_sum-median', 'expert_sat-crowd_majority',
       'crowd_sat-crowd_majority', 'satisfaction_area-crowd_majority',
       'satisfaction_sum-crowd_majority', 'expert_sat-expert_majority',
       'crowd_sat-expert_majority', 'satisfaction_area-expert_majority',
       'satisfaction_sum-expert_majority', 'expert_sat-majority',
       'crowd_sat-majority', 'satisfaction_area-majority',
       'satisfaction_sum-majority']]

_, _, ref_satisfaction =  get_min_and_max_satisfactions(result_optm_abs, alt_attribute = 'question_id')


#sat_col = [col for col in res_kalai.columns if 'sat' in col and 'diff' not in col]
#maxsat_col = [col for col in max_satisfaction.columns if 'sat' in col]

###### add relative satisfection by each alternative
res_nash = relative_detail_satisfaction_nash(res_nash, ref_satisfaction)
res_kalai = relative_detail_satisfaction_kalai(res_kalai, ref_satisfaction)
res_baseline = relative_detail_satisfaction_baseline(res_baseline, ref_satisfaction)


'''
    Result Analysis
'''

#from Data_Prepare import all_mathods_optimal_grades
from Evaluate_and_Results import add_method_name

res_nash = add_method_name(res_nash, '-nash', keep_same = {'question_id'})
res_kalai = add_method_name(res_kalai, '-kalai', keep_same = {'question_id'})

df_list = [res_nash, res_kalai, res_baseline ]    
#df_crowd_sample = df_crowd.groupby('vote', group_keys = False).apply(lambda x: x.sample(min(len(x),3)))
from functools import reduce
df_all_detail = reduce(lambda left,right: pd.merge(left,right,on='question_id'), df_list)

#df_all_detail['expert_std']  = np.std(df_alt_votes[expert_ids], axis = 1)
#df_all_detail['crowd_std']  = np.std(df_alt_votes[crowd_ids], axis = 1)

exp_median = np.array(df_all_detail['expert_median']).reshape(len(df_all_detail), 1)
crd_median = np.array(df_all_detail['crowd_median']).reshape(len(df_all_detail), 1)

df_all_detail['expert_median_diff']  = np.mean(np.abs(np.array(df_alt_votes[expert_ids]) - exp_median), axis = 1, keepdims=True)
df_all_detail['crowd_median_diff']  = np.mean(np.abs(np.array(df_alt_votes[crowd_ids]) - crd_median), axis = 1, keepdims=True)


crowd_sat = [col for col in df_all_detail.columns if col.startswith('crowd_sat')] 
expert_sat = [col for col in df_all_detail.columns if col.startswith('expert_sat')] 

max_expert = ['expert_sat-expert_median']
max_crowd = ['crowd_sat-crowd_median']


for c , e  in zip(crowd_sat, expert_sat):
    name = e.split('-')[1]
    assert(e.split('-')[1] == c.split('-')[1])
    #print('crowd_sat: ', c, '; ' 'expert_sat: ',   e, ';' ,'method: ', name)
    
    
    n = len(df_all_detail)
    
    method_cols = [e, c]
    
    max_df = df_all_detail[['expert_sat-expert_median', 'crowd_sat-crowd_median']].rename(columns = {'expert_sat-expert_median':'max_expert_sat', 'crowd_sat-crowd_median':'max_crowd_sat'})
    
    help_df = df_all_detail[method_cols] 
    help_df = pd.merge(help_df, max_df, left_index=True, right_index=True)
    
    

    help_df['maxmax'] = max_df.idxmax(axis = 1)    

    def kalai_score(row, name = name):
        if row['maxmax'] == 'max_expert_sat':
            return (row['expert_sat-'+name]/row['max_expert_sat'] - row['crowd_sat-'+name]/row['max_crowd_sat'] )
        else:
            return (row['crowd_sat-'+name]/row['max_crowd_sat'] - row['expert_sat-'+name]/row['max_expert_sat'])
    
   
    df_all_detail['sum_gain-' + name] =  help_df.apply(kalai_score, axis = 1) #+ 1) #+  df_all_detail['satisfaction_sum-' + name]
    
   
    
df_all_detail['median-diff'] = np.abs(df_all_detail['expert_median'] - df_all_detail['crowd_median'])


df_all_detail.to_csv('results_tx/results_detail_all_'+ expert_type +'.csv')


measure_cols = [col for col in df_all_detail.columns if 'sat' in col or 'area' in col or 'sum_gain' in col]
#measure_cols = [col for col in measure_cols if  'rel_' not in col]

crd_sat = [col for col in measure_cols if  'crowd_sat'  in col and 'rel' not in col]
exp_sat = [col for col in measure_cols if  'expert_sat'  in col and 'rel' not in col]
sum_sat = [col for col in measure_cols if  'satisfaction_sum'  in col and 'rel' not in col]
area_sat = [col for col in measure_cols if  'area'  in col and 'rel' not in col]
gain_sat = [col for col in measure_cols if  'sum_gain'  in col]
rel_sat = [col for col in measure_cols if 'rel_sat' in col]
rel_area = [col for col in rel_sat if 'area' in col]
rel_sum = [col for col in rel_sat if 'sum' in col]

crd = pd.melt(df_all_detail, id_vars=['question_id'], value_vars=crd_sat, var_name = 'method', value_name = 'crowd_sat')
crd['method'] = crd['method'].str.split('-').apply(lambda x: x[1]) 

exp = pd.melt(df_all_detail, id_vars=['question_id'], value_vars=exp_sat, var_name = 'method', value_name = 'expert_sat')
exp['method'] = exp['method'].str.split('-').apply(lambda x: x[1]) 

tot = pd.melt(df_all_detail, id_vars=['question_id'], value_vars=sum_sat, var_name = 'method', value_name = 'sum_sat')
tot['method'] = tot['method'].str.split('-').apply(lambda x: x[1]) 

area = pd.melt(df_all_detail, id_vars=['question_id'], value_vars=area_sat, var_name = 'method', value_name = 'area_sat')
area['method'] = area['method'].str.split('-').apply(lambda x: x[1]) 

gain = pd.melt(df_all_detail, id_vars=['question_id'], value_vars=gain_sat, var_name = 'method', value_name = 'diff_gain_sat')
gain['method'] = gain['method'].str.split('-').apply(lambda x: x[1]) 

r_area = pd.melt(df_all_detail, id_vars=['question_id'], value_vars=rel_area, var_name = 'method', value_name = 'rel_area_sat')
r_area['method'] = r_area['method'].str.split('-').apply(lambda x: x[1])

r_sum = pd.melt(df_all_detail, id_vars=['question_id'], value_vars=rel_sum, var_name = 'method', value_name = 'rel_sum_sat')
r_sum['method'] = r_sum['method'].str.split('-').apply(lambda x: x[1])

all_detail_results = reduce(lambda left,right: pd.merge(left,right,on=['question_id', 'method']), [crd,exp,tot,area,gain, r_area, r_sum])
all_detail_metrics = pd.merge(all_detail_results, df_all_detail[['question_id', 'median-diff']], on = 'question_id')

df_all_detail.to_csv('results_tx/results_detail_all_'+ expert_type +'.csv')
all_detail_metrics.to_csv('results_tx/results_all_detail_metrics_'+ expert_type +'.csv')
#question_map.to_excel('results_tx/question_map.xlsx')  
'''
                crowd_sat	expert_sat
crowd_median	    2.318797	    1.933333	
expert_median	2.269173 	2.400000

expert_s_1 = 2.400000
expert_s_2 = 1.933333	
crowd_s_1 = 2.269173
crowd_s_2 = 2.318797	

'''
from Optimize_Grades import objective_function_grades_absolute    
lambda_expert = 0.4
expert_type   
org_votes = df_alt_votes[df_alt_votes['question_id'] == 8]
vote = objective_function_grades_absolute(np.array(org_votes[expert_ids]).reshape(len(expert_ids),1), 
                                              np.array(org_votes[crowd_ids]).reshape(len(crowd_ids),1),  
                                              lambda_expert, 1 - lambda_expert)[0]


vote 
