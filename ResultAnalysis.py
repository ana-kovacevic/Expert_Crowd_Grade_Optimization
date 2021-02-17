# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 20:24:29 2021

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

from Data_Prepare import get_aggregated_data
from Data_Prepare import create_ratings_and_mapping

from Data_Prepare import get_user_ids_from_mapping

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

df_selected_expert =  df_journal # df_science # 
expert_type = 'journal' # 'science' #   

#alts_dict = dict(zip(alternative_map['alternative_id'] , alternative_map['alternative_name']))


#### transacional data of expert and crowd that labeled same alternatives as experts
df_expert_crowd = pd.concat([df_selected_expert, df_crowd], ignore_index=True)
#n_crowd = len(df_crowd['voter'].unique())

############# Aggregate data
#crowd_agg = get_aggregated_data(df_crowd, alt_names)
#expert_agg = get_aggregated_data(df_selected_expert, alt_names)
expert_crowd_agg = get_aggregated_data(df_expert_crowd, alt_names)


'''
Create data for factorization
'''

_, alts_lookup, voters_lookup = create_ratings_and_mapping(expert_crowd_agg, alt_names)

#voter_dict = dict(zip(voters_lookup['voter_id'], voters_lookup['voter']))

##### replace voters name with ids in all dataframes
# df_crowd = pd.merge( voters_lookup,df_crowd, how = 'inner', on = 'voter').drop('voter', axis = 1)
# df_expert_crowd = pd.merge( voters_lookup, df_expert_crowd, how = 'inner', on = 'voter').drop('voter', axis = 1)
# df_selected_expert = pd.merge(voters_lookup,  df_selected_expert, how = 'inner', on = 'voter').drop('voter', axis = 1)
# crowd_agg = pd.merge(voters_lookup,  crowd_agg, how = 'inner', on = 'voter').drop('voter', axis = 1)
# expert_agg = pd.merge(voters_lookup,  expert_agg, how = 'inner', on = 'voter').drop('voter', axis = 1)

"""
################### Read factors  
-------------- read factors obtained by previous experiments
"""

#user_factors =np.array( pd.read_csv('results/user_factors_' + expert_type + '.csv').drop('Unnamed: 0', axis = 1))
#alt_factors = np.array( pd.read_csv('results/alts_factors_' + expert_type + '.csv').drop('Unnamed: 0', axis = 1))


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

# ## pivot vote data for each voter
# df_voter_votes = get_aggregated_data(pd.concat([all_crowd_grades, all_exp_grades]), alts_lookup['alternative_id'], 
#                                     index_column = 'voter_id', column= 'alternative_id', value = 'rate')

#df_alt_votes = get_aggregated_data(all_crowd_grades, voters_lookup['voter_id'], index_column = 'alternative_id', column= 'voter_id', value = 'rate')


'''
################################ Results
'''

res_nash = pd.read_csv('results/results_nash_' + expert_type + '.csv').drop('Unnamed: 0', axis = 1)
res_nash = res_nash[['alternative_id','lambda_exp', 'optimal_grade', 'expert_sat', 'crowd_sat', 'satisfaction_area', 'satisfaction_sum']]
###### kalai

res_kalai = pd.read_csv('results/results_kalai_' + expert_type + '.csv').drop('Unnamed: 0', axis = 1)
res_kalai = res_kalai[['alternative_id','lambda_exp', 'vote', 'expert_sat', 'crowd_sat', 'satisfaction_area', 'satisfaction_sum']]
####### Baseline methods

res_baseline = pd.read_csv('results/results_baseline_'+ expert_type +'.csv').drop('Unnamed: 0', axis = 1)
res_baseline = res_baseline[['alternative_id', 'crowd_mean', 'expert_mean', 'mean', 'crowd_median',
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
#### weighted method
res_weighted = pd.read_csv('results/results_weighted_'+ expert_type +'.csv').drop('Unnamed: 0', axis = 1)

###### relative satisfaction calculation

_, _, ref_satisfaction =  get_min_and_max_satisfactions(result_optm_abs)


###### add relative satisfection by each alternative and gains
res_nash = relative_detail_satisfaction_nash(res_nash, ref_satisfaction)
res_kalai = relative_detail_satisfaction_kalai(res_kalai, ref_satisfaction)
res_baseline = relative_detail_satisfaction_baseline(res_baseline, ref_satisfaction)
res_weighted = relative_detail_satisfaction_baseline(res_weighted, ref_satisfaction)




# res_nash.to_csv('results/results_nash_' + expert_type +'.csv')
# res_kalai.to_csv('results/results_kalai_' + expert_type +'.csv')
# res_baseline.to_csv('results/results_baseline_'+ expert_type +'.csv')
# res_weighted.to_csv('results/results_weighted_'+ expert_type +'.csv')


from Data_Prepare import all_mathods_optimal_grades
from Evaluate_and_Results import add_method_name

res_nash = add_method_name(res_nash, '-nash', keep_same = {'alternative_id'})
res_kalai = add_method_name(res_kalai, '-kalai', keep_same = {'alternative_id'})

df_list = [res_nash, res_kalai, res_baseline , res_weighted]    
#df_crowd_sample = df_crowd.groupby('vote', group_keys = False).apply(lambda x: x.sample(min(len(x),3)))
from functools import reduce
df_all_detail = reduce(lambda left,right: pd.merge(left,right,on='alternative_id'), df_list)

#df_all_detail['expert_std']  = np.std(df_alt_votes[expert_ids], axis = 1)
#df_all_detail['crowd_std']  = np.std(df_alt_votes[crowd_ids], axis = 1)

exp_median = np.array(df_all_detail['expert_median']).reshape(len(df_all_detail), 1)
crd_median = np.array(df_all_detail['crowd_median']).reshape(len(df_all_detail), 1)

df_all_detail['expert_median_diff']  = np.mean(np.abs(np.array(df_alt_votes[expert_ids]) - exp_median), axis = 1, keepdims=True)
df_all_detail['crowd_median_diff']  = np.mean(np.abs(np.array(df_alt_votes[crowd_ids]) - crd_median), axis = 1, keepdims=True)




#df_all_detail['expert_iqr'] = iqr(df_alt_votes[expert_ids], axis=1, keepdims=True)
#df_all_detail['crowd_iqr'] = iqr(df_alt_votes[crowd_ids], axis = 1, keepdims=True)

crowd_sat = [col for col in df_all_detail.columns if col.startswith('crowd_sat')] 
expert_sat = [col for col in df_all_detail.columns if col.startswith('expert_sat')] 

max_expert = ['expert_sat-expert_median']
max_crowd = ['crowd_sat-crowd_median']

#from scipy.stats import iqr
#q75, q25 = np.percentile(df_alt_votes[expert_ids], [75 ,25], axis = 1)
#iqre = q75 - q25


for c , e  in zip(crowd_sat, expert_sat):
    name = e.split('-')[1]
    assert(e.split('-')[1] == c.split('-')[1])
    #print('crowd_sat: ', c, '; ' 'expert_sat: ',   e, ';' ,'method: ', name)
    
    #sat_gain = np.array(df_all_detail[c] / df_all_detail[e]).reshape(len(df_all_detail),1)
    #max_gain = np.array(df_all_detail[max_crowd])/ np.array(df_all_detail[max_expert])
    
    # df_all_detail['newgain-' + name] = np.abs(sat_gain  - max_gain)
    # df_all_detail['racio_gain-' + name] = sat_gain/max_gain
    
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
    
    #np.max(help_df[max_expert + max_crowd], axis = 1)
    
    #a = help_df.iloc[:, [1,3]].idxmax(axis=1).str.split('-').apply(lambda x: x[0]) + '-' + name
    #a = a.tolist()
    
    
    # crd_gain = np.array(np.exp(df_all_detail[c])/ np.exp(df_all_detail['crowd_iqr'])).reshape(n,1)
    # exp_gain = np.array(np.exp(df_all_detail[e])/ np.exp(df_all_detail['expert_iqr'])).reshape(n,1)
    
    # df_all_detail['sum_gain-' + name] = crd_gain + exp_gain
    # df_all_detail['sum_gain_log-' + name] = np.log(crd_gain + exp_gain)
    
    # crd_gain = np.array(np.exp(df_all_detail[c])/ np.exp(df_all_detail['crowd_median_diff'])).reshape(n,1)
    # exp_gain = np.array(np.exp(df_all_detail[e])/ np.exp(df_all_detail['expert_median_diff'])).reshape(n,1)
    
    # df_all_detail['sum_gain-' + name] = crd_gain + exp_gain
    # df_all_detail['sum_gain_log-' + name] = np.log(crd_gain) + np.log(exp_gain)
    
    #df_all_detail['sum_gain_norm-' + name] = df_all_detail['sum_gain-' + name]/np.sum(df_all_detail['sum_gain-' + name])
    
    # crd_max = np.array(df_all_detail[max_crowd]).reshape(n,1)
    # crd_sat = np.array(df_all_detail[c]).reshape(n,1)
    
    # crd_gain = crd_max / crd_sat
    
    # exp_max = np.array(df_all_detail[max_expert]).reshape(n,1)
    # exp_sat = np.array(df_all_detail[e]).reshape(n,1)
    
    # exp_gain = exp_max/ exp_sat
    
    # df_all_detail['sum_gain-' + name] = crd_gain + exp_gain
    
df_all_detail['median-diff'] = np.abs(df_all_detail['expert_median'] - df_all_detail['crowd_median'])



# for i in df_all_detail.columns:
#     print(i)

df_all_detail.to_csv('results/results_detail_all_'+ expert_type +'.csv')


measure_cols = [col for col in df_all_detail.columns if 'sat' in col or 'area' in col or 'sum_gain' in col]
#measure_cols = [col for col in measure_cols if  'rel_' not in col]

crd_sat = [col for col in measure_cols if  'crowd_sat'  in col and 'rel' not in col]
exp_sat = [col for col in measure_cols if  'expert_sat'  in col]
sum_sat = [col for col in measure_cols if  'satisfaction_sum'  in col]
area_sat = [col for col in measure_cols if  'area'  in col]
gain_sat = [col for col in measure_cols if  'sum_gain'  in col]
rel_sat = [col for col in measure_cols if 'rel_sat' in col]

crd = pd.melt(df_all_detail, id_vars=['alternative_id'], value_vars=crd_sat, var_name = 'method', value_name = 'crowd_sat')
crd['method'] = crd['method'].str.split('-').apply(lambda x: x[1]) 

exp = pd.melt(df_all_detail, id_vars=['alternative_id'], value_vars=exp_sat, var_name = 'method', value_name = 'expert_sat')
exp['method'] = exp['method'].str.split('-').apply(lambda x: x[1]) 

tot = pd.melt(df_all_detail, id_vars=['alternative_id'], value_vars=sum_sat, var_name = 'method', value_name = 'sum_sat')
tot['method'] = tot['method'].str.split('-').apply(lambda x: x[1]) 

area = pd.melt(df_all_detail, id_vars=['alternative_id'], value_vars=area_sat, var_name = 'method', value_name = 'area_sat')
area['method'] = area['method'].str.split('-').apply(lambda x: x[1]) 

gain = pd.melt(df_all_detail, id_vars=['alternative_id'], value_vars=gain_sat, var_name = 'method', value_name = 'diff_gain_sat')
gain['method'] = gain['method'].str.split('-').apply(lambda x: x[1]) 

all_detail_results = reduce(lambda left,right: pd.merge(left,right,on=['alternative_id', 'method']), [crd,exp,tot,area,gain])
all_detail_metrics = pd.merge(all_detail_results, df_all_detail[['alternative_id', 'median-diff']], on = 'alternative_id')


all_detail_metrics.to_csv('results/results_all_detail_metrics_'+ expert_type +'.csv')
# df_all_detail = pd.read_csv('results/results_detail_all_'+ expert_type +'.csv').drop('Unnamed: 0', axis = 1)





sums_sat = [col for col in df_all_detail.columns if 'rel_sum' in col and 'kalai' not in col and 'nash' not in col]

df_all_detail['Max_Methods'] = np.max(df_all_detail[sums_sat], axis = 1)
df_all_detail['Kalai-Max_sum'] = df_all_detail['rel_satisfaction_sum-kalai'] - df_all_detail['Max_Methods']

df_all_detail['Kalai-Max_sum'].sort_values(ascending = True)
#df_all_detail['rel_sum-w_crowd_grades'].sort_values(ascending = False)
 ## ----------------------------------------------------------------------------
res_baseline['median_diff'] = np.abs(res_baseline['expert_median'] - res_baseline['crowd_median'])

diff_med = 1.5
alts_diff = list(res_baseline[res_baseline['median_diff']>= diff_med]['alternative_id'])
print(len(alts_diff))

df_all = df_all_detail[df_all_detail['alternative_id'].isin(alts_diff)]

df_all.sort_values(by = 'Kalai-Max_sum', ascending = True)


import matplotlib.pyplot as plt
plt.hist(df_all['Kalai-Max_sum'], bins = 100, density=True)

a = df_all_detail[df_all_detail.alternative_id == 1420]
a.columns
a[['rel_expert_sat-kalai', 'rel_crowd_sat-kalai', 'rel_satisfaction_sum-kalai']]
b = a[sums_sat]




df_all_detail['Kalai-Max_sum'].sort_values()
a = pd.merge(res_baseline[['alternative_id','expert_sat-crowd_median', 'expert_sat-expert_median', 'crowd_sat-expert_median', 'crowd_sat-crowd_median']], ref_satisfaction[['alternative_id', 'min_expert_sat', 'max_expert_sat', 'min_crowd_sat', 'max_crowd_sat']], how = 'inner', on = 'alternative_id')
bas = 'crowd_sat-crowd_median'
ref = 'max_crowd_sat'

b = a[ a[bas] != a[ref]]
b[['alternative_id',bas, ref]]
b['diff'] = b[bas] -  b[ref]
b.sort_values(by = 'diff', ascending = False)


######### SUMMARIZE RESULTS

####### select alternatives with certan level of difference between groups
res_baseline['median_diff'] = np.abs(res_baseline['expert_median'] - res_baseline['crowd_median'])

diff_med = 1.0
alts_diff = list(res_baseline[res_baseline['median_diff']>= diff_med]['alternative_id'])
print(len(alts_diff))

kalai = res_kalai[res_kalai['alternative_id'].isin(alts_diff)]
nash = res_nash[res_nash['alternative_id'].isin(alts_diff)]
baseline = res_baseline[res_baseline['alternative_id'].isin(alts_diff)]
weighted = res_weighted[res_weighted['alternative_id'].isin(alts_diff)]

res_overal_sat = avg_satisfaction_by_group(kalai, nash, baseline, weighted).reset_index()
res_overal_sat.to_csv('results/results_overall_avg_satisfaction_'+ expert_type +'.csv')

diff_grade = 1.0
alts_diff = list(res_baseline[res_baseline['median_diff']>= diff_med]['alternative_id'])
print(len(alts_diff))

kalai = res_kalai[res_kalai['alternative_id'].isin(alts_diff)]
nash = res_nash[res_nash['alternative_id'].isin(alts_diff)]
baseline = res_baseline[res_baseline['alternative_id'].isin(alts_diff)]
weighted = res_weighted[res_weighted['alternative_id'].isin(alts_diff)]

res_overal_sat = avg_satisfaction_by_group(kalai, nash, baseline, weighted).reset_index()



res_relative_sat = relative_overall_satisfaction(nash, kalai, baseline, weighted, ref_satisfaction)
res_relative_sat = res_relative_sat.rename(columns = ({'crowd_sat': 'rel-crowd_sat', 'expert_sat':'rel-expert_sat', 
                                                       'satisfaction_area' : 'rel-satisfaction_area',
                                                       'satisfaction_sum': 'rel-satisfaction_sum'}))


all_sum_res = pd.merge(res_relative_sat, res_overal_sat, on = 'method')
all_sum_res = all_sum_res.drop('index', axis = 1)

all_sum_res.to_csv('results/results_overall_relative_'+ expert_type +'.csv')



###### Statistic

# Mann-Whitney U test

from scipy.stats import mannwhitneyu
# seed the random number generator
for i in range(len(df_alt_votes)):
    #print('Alternative: ', str(i))
# generate two independent samples
    data1 = df_alt_votes[expert_ids].iloc[i,:]
    data2 = df_alt_votes[crowd_ids].iloc[i,:]
# compare samples
    stat, p = mannwhitneyu(data1, data2)
    #print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
    alpha = 0.01
    if p > alpha:
        continue #print('Same distribution (fail to reject H0)')
    else:
        print('Alternative: ', str(i))
        print('Statistics=%.3f, p=%.3f' % (stat, p))
        print('Different distribution (reject H0)')
        print('------------------------------------------------')

#################### Result analysis - differences

from Data_Prepare import all_mathods_optimal_grades
from Evaluate_and_Results import add_method_name

res_nash = add_method_name(res_nash, '-nash', keep_same = {'alternative_id'})
res_kalai = add_method_name(res_kalai, '-kalai', keep_same = {'alternative_id'})

df_list = [res_nash, res_kalai, res_baseline, res_weighted]    
#df_crowd_sample = df_crowd.groupby('vote', group_keys = False).apply(lambda x: x.sample(min(len(x),3)))
from functools import reduce
df_all_detail = reduce(lambda left,right: pd.merge(left,right,on='alternative_id'), df_list)
df_all_detail.to_csv('results/results_detail_all_'+ expert_type +'.csv')
#### Take optimal grade for each method
df_all_detail = pd.read_csv('results/results_detail_all_'+ expert_type +'.csv').drop('Unnamed: 0', axis = 1)

sums_sat = [col for col in df_all_detail.columns if 'rel_sum' in col and 'kalai' not in col and 'nash' not in col]

df_all_detail['Max_Methods'] = np.max(df_all_detail[sums_sat], axis = 1)
df_all_detail['Kalai-Max_sum'] = df_all_detail['rel_satisfaction_sum-kalai'] - df_all_detail['Max_Methods']

df_all_detail['Kalai-Max_sum'].sort_values(ascending = False)
df_all_detail['rel_sum-w_crowd_grades'].sort_values(ascending = False)

import matplotlib.pyplot as plt
plt.hist(df_all_detail['Kalai-Max_sum'], bins = 100, density=True)

df_all_detail.to_csv('results/results_detail_all_'+ expert_type +'.csv')


all_method_votes = all_mathods_optimal_grades(df_list)



###### merge all votes in one dataframe
df_final = reduce(lambda left,right: pd.merge(left,right,on='alternative_id'), all_method_votes)
df_final.columns = [x[-1] if len(x)>1 else x[0] for x in list(df_final.columns.str.split('-'))]


a[['expert_sat-mean', 'expert_sat-kalai']]
a[['crowd_sat-mean', 'crowd_sat-kalai']]





ref_satisfaction[ref_satisfaction.alternative_id == 1420][['min_expert_sat', 'max_expert_sat']]
ref_satisfaction[ref_satisfaction.alternative_id == 1420][['min_crowd_sat', 'max_crowd_sat']]




#from Evaluate_and_Results import   satisfaction_gain_derivative
from Evaluate_and_Results import calculate_gain_gradient

#satisfaction_gain_derivative(e_votes, c_votes, grade)
res_derivative = calculate_gain_gradient(df_alt_votes, df_final, expert_ids, crowd_ids)

res_derivative = res_derivative.abs()
met_cols = ['gain2-nash', 'gain2-kalai', 'gain2-crowd_mean',
       'gain2-expert_mean', 'gain2-mean', 'gain2-crowd_median',
       'gain2-expert_median', 'gain2-median', 'gain2-crowd_majority',
       'gain2-expert_majority', 'gain2-majority', 'gain2-w_crowd_grades',
       'gain2-w_all_grades']

#melt for visual analysis
#der = pd.melt(res_derivative, id_vars=['alternative_id'], value_vars=met_cols, var_name = 'method', value_name = 'gain2')
#der['method'].str.split('-')[1]

res_derivative[met_cols] = 0 - res_derivative[met_cols]

res_derivative = res_derivative.abs()



derivative_sum = res_derivative[met_cols].mean(axis = 0)
derivative_sum = derivative_sum.reset_index()

res_derivative.to_csv('results/results_derivative_detail_'+ expert_type +'.csv')

cols = list(df_final.columns)
try:
    cols.remove('alternative_id')
except ValueError:
    print("Given Element Not Found in List")
    
pd.melt(df_final, id_vars=['alternative_id'], value_vars=cols, var_name = 'method', value_name = 'grade')

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