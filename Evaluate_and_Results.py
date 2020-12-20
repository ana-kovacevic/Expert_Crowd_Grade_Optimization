# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 12:50:22 2020

@author: akovacevic
"""
import pandas as pd
import numpy as np

from scipy.optimize import minimize #,optimize
#from scipy.stats import mode

#from Optimize_Grades import expectation_maximization_nash
from Optimize_Grades import maximization_kalai_smorodinsky
from Optimize_Grades import nash_solution
from Optimize_Grades import calculate_satisfaction_absolute


def nash_results(df_alt_votes, max_grade, crowd_ids, expert_ids, cons, bnds, lambda_expert = 0.5):

    #res_nash = pd.DataFrame(columns=(['alternative_id', 'lambda_exp', 'vote', 'expert_sat', 'crowd_sat', 'area']))
    res_nash = pd.DataFrame(columns=(['alternative_id', 'lambda_exp', 'optimal_grade']))
    #res_nash[['lambda_exp', 'vote', 'area']] = np.nan
   
    w_lambda_1 = lambda_expert
    w_lambda_2 = 1 - lambda_expert
    w_vote = 1
    w = [w_lambda_1, w_lambda_2, w_vote]
    # i = 0
    for i in list(df_alt_votes.alternative_id.unique()):
        #res = optimal_grades[(optimal_grades['alternative_id'] == i) & (optimal_grades['alpha'] == lambda_expert)]
        votes = df_alt_votes[df_alt_votes['alternative_id'] == i]
        v_expert = np.array(votes[expert_ids])[0]
        v_crowd = np.array(votes[crowd_ids])[0]
        
        if i%100==0:
            print('Alternative to optimize: ', str(i))
    
        # n =  expectation_maximization_nash(res, lambda_expert, votes, crowd_ids, expert_ids,max_grade, num_iter = 100, verbose = False)
        n = minimize(nash_solution, w, constraints= cons, bounds=bnds,
                     args = (v_expert, v_crowd, max_grade), method = 'SLSQP')
        n = n.x[[0,2]]
        n = (i,) + tuple(n)
        
        res_nash = res_nash.append(pd.Series(list(n), index=res_nash.columns ), ignore_index=True)
    
    res_nash = calculate_satisfaction_absolute(df_alt_votes, res_nash, max_grade, expert_ids, crowd_ids)
    
    res_nash['satisfaction_area'] = res_nash['expert_sat'] * res_nash['crowd_sat']
    res_nash['satisfaction_sum'] = res_nash['expert_sat'] + res_nash['crowd_sat']
    res_nash['crowd_mean']=df_alt_votes[crowd_ids].apply(lambda x: np.mean(x), axis =1)
    res_nash['expert_mean']=df_alt_votes[expert_ids].apply(lambda x: np.mean(x), axis =1)
    res_nash['mean_diff'] = res_nash['expert_mean'] - res_nash['crowd_mean']
    res_nash['crowd_std']=df_alt_votes[crowd_ids].apply(lambda x: np.std(x), axis =1)
    res_nash['expert_std']=df_alt_votes[expert_ids].apply(lambda x: np.std(x), axis =1)
    res_nash['diff_sat'] =  res_nash['expert_sat'] - res_nash['crowd_sat']
    #res_nash.to_csv('results/results_nash.csv')   
    return res_nash
###### kalai
# df_alt_votes = df_alt_votes[df_alt_votes['alternative_id'] == 0]
# optimal_grades =  result_optm_abs
def kalai_results(df_alt_votes, optimal_grades, max_grade, crowd_ids, expert_ids):
    res_kalai = pd.DataFrame(columns=(['alternative_id', 'lambda_exp', 'vote', 'expert_sat', 'crowd_sat', 'satisfaction_area'])) 
    #i = 0
    for i in list(df_alt_votes.alternative_id.unique()):
        res = optimal_grades[(optimal_grades['alternative_id'] == i) &(optimal_grades['alpha'].isin([0,1]))]
        votes = df_alt_votes[df_alt_votes['alternative_id'] == i]
        
        if i%100==0:
            print('Alternative to optimize: ', str(i))
        
        k = maximization_kalai_smorodinsky(res, max_grade, votes, crowd_ids, expert_ids)
        
        k = (i,) + k
        res_kalai = res_kalai.append(pd.Series(list(k), index=res_kalai.columns ), ignore_index=True)
    
    res_kalai['satisfaction_sum'] = res_kalai['expert_sat'] + res_kalai['crowd_sat']
    res_kalai['crowd_mean']=df_alt_votes[crowd_ids].apply(lambda x: np.mean(x), axis =1)
    res_kalai['expert_mean']=df_alt_votes[expert_ids].apply(lambda x: np.mean(x), axis =1)
    res_kalai['mean_diff'] = res_kalai['expert_mean'] - res_kalai['crowd_mean']
    res_kalai['crowd_std']=df_alt_votes[crowd_ids].apply(lambda x: np.std(x), axis =1)
    res_kalai['expert_std']=df_alt_votes[expert_ids].apply(lambda x: np.std(x), axis =1)
    res_kalai['diff_sat'] =  res_kalai['expert_sat'] - res_kalai['crowd_sat']
    #res_kalai.to_csv('results/results_kalai.csv')
    return res_kalai

#### base line

def calculate_baseline_stats_satisfaction(df_alt_votes, max_grade, crowd_ids, 
                                          expert_ids ,stats = ['np.mean', 'np.median', 'mode']):
    """
    

    Parameters
    ----------
    df_alt_votes : TYPE
        DESCRIPTION.
    crowd_ids : TYPE
        DESCRIPTION.
    expert_ids : TYPE
        DESCRIPTION.
    stats : TYPE, optional
        DESCRIPTION. The default is ['np.mean', 'np.median', 'mode'].

    Returns
    -------
    df_baseline : TYPE
        DESCRIPTION.

    """
    
    data = df_alt_votes.copy() #pd.DataFrame(df_alt_votes['alternative_id'], columns = ['alternative_id'])
    
    #i = 0
    for i in range(len(stats)):
        name = stats[i]
        part_name = stats[i].split('.')[-1]
        
        if part_name != 'mode':
            data['crowd_' + part_name]= data[crowd_ids].apply(lambda x: eval(name)(x), axis =1)
            data['expert_' + part_name] = data[expert_ids].apply(lambda x: eval(name)(x), axis =1)
            data[part_name] = data[crowd_ids + expert_ids].apply(lambda x: eval(name)(x), axis = 1)
        else:
            data['crowd_majority']=data[crowd_ids].round().mode(axis = 1)[0]
            data['expert_majority']= data[expert_ids].round().mode(axis=1)[0]
            data['majority'] = data[crowd_ids + expert_ids].round().mode(axis = 1)[0]
    
    
    # mean
    baseline_col =  [i for i in list(data.columns) if i not in crowd_ids + expert_ids][1:]
    
    #data = pd.merge(df_baseline, df_alt_votes, on= 'alternative_id')
    #col = 'crowd_median'
    for col in baseline_col:
        grade = np.array(data[col]).reshape(len(data),1)
        expert_votes = np.array(data[expert_ids]).reshape(len(data),data[expert_ids].shape[1])
        crowd_votes = np.array(data[crowd_ids]).reshape(len(data),data[crowd_ids].shape[1])
        
        expert_sat = np.mean(max_grade - np.abs(expert_votes - grade), axis = 1)
        crowd_sat = np.mean(max_grade - np.abs(crowd_votes - grade), axis = 1)
        
        sum_sat = expert_sat+crowd_sat
        product_sat = expert_sat*crowd_sat
        
        data['expert_sat-' + col]= expert_sat
        data['crowd_sat-' + col]= crowd_sat
        
        
        data['satisfaction_area-' + col] = product_sat
        data['satisfaction_sum-' + col] = sum_sat
    
    df_baseline = data.drop(crowd_ids+expert_ids, axis = 1)
    return df_baseline

#optimal_grades = weighted_grades
#data = df_alt_votes
def satisfaction_calculation_weighted_methods(df_alt_votes, max_grade, crowd_ids, expert_ids, optimal_grades):
    
    data = df_alt_votes.copy()
    #res_data = pd.DataFrame(columns=(['method', 'expert_sat', 'crowd_sat']))
    
    for key in optimal_grades.keys():
        
        grade = np.array(optimal_grades[key]).reshape(len(data),1)
        expert_votes = np.array(data[expert_ids]).reshape(len(data),data[expert_ids].shape[1])
        crowd_votes = np.array(data[crowd_ids]).reshape(len(data),data[crowd_ids].shape[1])
            
        expert_sat = np.mean(max_grade - np.abs(expert_votes - grade), axis = 1)
        crowd_sat = np.mean(max_grade - np.abs(crowd_votes - grade), axis = 1)
        
        sum_sat = expert_sat + crowd_sat
        product_sat = expert_sat*crowd_sat
        
        data['expert_sat-' + key] = expert_sat
        data['crowd_sat-' + key] = crowd_sat
        
        
        data['satisfaction_area-' + key] = product_sat
        data['satisfaction_sum-' + key] = sum_sat
        #res = pd.DataFrame()
        #res['expert_sat'] = expert_sat
        #res['crowd_sat'] = crowd_sat
        #res['method'] = key
    df_res = data.drop(crowd_ids+expert_ids, axis = 1)
        #res_data = res_data.append(res, ignore_index = True)
    return df_res

def get_min_and_max_satisfactions(result_optm_abs):        
    max_satisfaction = result_optm_abs[['alternative_id', 'crowd_sat', 'expert_sat']].groupby(by='alternative_id' ).agg('max').reset_index()
    max_satisfaction = max_satisfaction.rename(columns = {'crowd_sat':'max_crowd_sat', 'expert_sat' : 'max_expert_sat'})
    max_satisfaction['max_satisfaction_sum'] = max_satisfaction['max_crowd_sat'] + max_satisfaction['max_expert_sat']
    max_satisfaction['max_satisfaction_area'] = max_satisfaction['max_crowd_sat'] * max_satisfaction['max_expert_sat']
    
    
    min_satisfaction = result_optm_abs[['alternative_id', 'crowd_sat', 'expert_sat']].groupby(by='alternative_id' ).agg('min').reset_index()
    min_satisfaction = min_satisfaction.rename(columns = {'crowd_sat':'min_crowd_sat', 'expert_sat' : 'min_expert_sat'})
    min_satisfaction['min_satisfaction_sum'] = min_satisfaction['min_crowd_sat'] + min_satisfaction['min_expert_sat']
    min_satisfaction['min_satisfaction_area'] = min_satisfaction['min_crowd_sat'] * min_satisfaction['min_expert_sat']
        
    ref_satisfaction = pd.merge(max_satisfaction, min_satisfaction, on = 'alternative_id')   

    return min_satisfaction, max_satisfaction, ref_satisfaction     
        

def avg_satisfaction_by_group(res_kalai, res_nash, res_baseline, res_weighted):
    
    ### Kalai results
    sat_cols = [col for col in res_kalai.columns if 'sat' in col or 'area' in col]
    sat_cols = [col for col in sat_cols if  'rel_' not in col and 'diff' not in col]
    kalai = pd.DataFrame(res_kalai[sat_cols].apply(lambda x: np.mean(x), axis = 0), columns = ['satisfaction'])
    kalai['method'] = 'Kalai'
    
    kalai.reset_index(inplace=True)
    kalai = kalai.rename(columns = {'index':'group'})
    kalai.drop(kalai[kalai['group'] == 'diff_sat'].index, inplace = True) 
    
    #### Nash results
    sat_cols = [col for col in res_nash.columns if 'sat' in col or 'area' in col]
    sat_cols = [col for col in sat_cols if  'rel_' not in col and 'diff' not in col]
    nash = pd.DataFrame(res_nash[sat_cols].apply(lambda x: np.mean(x), axis = 0), columns = ['satisfaction'])
    nash['method'] = 'Nash'
    
    nash.reset_index(inplace=True)
    nash = nash.rename(columns = {'index':'group'})
    nash.drop(nash[nash['group'] == 'diff_sat'].index, inplace = True) 

    ### Baseline results
    sat_cols = [col for col in res_baseline.columns if 'sat' in col]
    baseline = pd.DataFrame(res_baseline[sat_cols].apply(lambda x: np.mean(x), axis = 0), columns = ['satisfaction']) 
    
    baseline.reset_index(inplace=True)
    baseline = baseline.rename(columns = {'voter_id':'group-method', 'index':'group-method'})
    
    baseline['group'] = baseline.apply(lambda x:  x['group-method'].split('-')[0], axis = 1)
    baseline['method'] = baseline.apply(lambda x: x['group-method'].split('-')[1], axis = 1)
    baseline = baseline.drop(['group-method'], axis = 1)

    #np.mean(5 - np.abs(df_alt_votes[crowd_ids] - df_baseline['crowd_median']), axis = 1)
    
    ###### Weighted results
    sat_cols = [col for col in res_weighted.columns if 'sat' in col]
    weighted = pd.DataFrame(res_weighted[sat_cols].apply(lambda x: np.mean(x), axis = 0), columns = ['satisfaction']) 
    
    weighted.reset_index(inplace=True)
    weighted = weighted.rename(columns = {'voter_id':'group-method','index':'group-method'})
    
    weighted['group'] = weighted.apply(lambda x:  x['group-method'].split('-')[0], axis = 1)
    weighted['method'] = weighted.apply(lambda x: x['group-method'].split('-')[1], axis = 1)
    weighted = weighted.drop(['group-method'], axis = 1)
    

    res_all = pd.concat([kalai, nash, baseline, weighted])
    res_all= res_all.pivot(index = 'method', columns = 'group', values = 'satisfaction')
    
    res_all.reset_index(inplace=True)
    return res_all

def relative_detail_satisfaction_nash(res_nash, max_satisfaction):
    
    res_nash['rel_expert_sat'] = res_nash['expert_sat'] / max_satisfaction['max_expert_sat']
    res_nash['rel_crowd_sat'] = res_nash['crowd_sat'] / max_satisfaction['max_crowd_sat']
    res_nash['rel_satisfaction_sum'] = res_nash['satisfaction_sum'] / max_satisfaction['max_satisfaction_sum']
    res_nash['rel_satisfaction_area'] = res_nash['satisfaction_area'] / max_satisfaction['max_satisfaction_area']
    
    return res_nash
    
def relative_detail_satisfaction_kalai(res_kalai, max_satisfaction):
    
    res_kalai['rel_expert_sat'] = res_kalai['expert_sat'] / max_satisfaction['max_expert_sat']
    res_kalai['rel_crowd_sat'] = res_kalai['crowd_sat'] / max_satisfaction['max_crowd_sat']
    res_kalai['rel_satisfaction_sum'] = res_kalai['satisfaction_sum'] / max_satisfaction['max_satisfaction_sum']
    res_kalai['rel_satisfaction_area'] = res_kalai['satisfaction_area'] / max_satisfaction['max_satisfaction_area']
    
    return res_kalai
#res_baseline=res_weighted
def relative_detail_satisfaction_baseline(res_baseline, max_satisfaction):
    s = [col for col in res_baseline.columns if 'sat' in col]
    exp = [col for col in s if col.startswith('expert')]    
    crd = [col for col in s if col.startswith('crowd')] 
    adds = [col for col in s if 'satisfaction' in col and 'sum' in col] 
    area = [col for col in s if 'satisfaction' in col and 'area' in col]
    
    
    
    for c in exp:
        name = c.split('-')[1]
        res_baseline['rel_expert-' + name] = res_baseline[c]/max_satisfaction['max_expert_sat']
        
    for c in crd:
        name = c.split('-')[1]
        res_baseline['rel_crowd-' + name] = res_baseline[c]/max_satisfaction['max_crowd_sat']
        
    for c in adds:
        name = c.split('_')[1]
        res_baseline['rel-' + name] = res_baseline[c]/max_satisfaction['max_satisfaction_sum']
        
    for c in area:
        name = c.split('_')[1]
        res_baseline['rel-' + name] = res_baseline[c]/max_satisfaction['max_satisfaction_area']
    
    rel_exp = [col for col in res_baseline.columns if col.startswith('rel_expert')]    
    rel_crd = [col for col in res_baseline.columns if col.startswith('rel_crowd')]
    
    for e, c in zip(rel_exp, rel_crd):
        name = e.split('-')[1]
        assert(e.split('-')[1] == c.split('-')[1])
        
        res_baseline['gain-' + name] = np.abs(res_baseline[e] - res_baseline[c])

    
    return res_baseline
#relative_overall_satisfaction(res_nash, res_kalai, res_baseline, res_weighted, ref_satisfaction)
#max_satisfaction = ref_satisfaction
def relative_overall_satisfaction(res_nash, res_kalai, res_baseline, res_weighted, max_satisfaction):
    
    #### Max values 
    total_max = pd.DataFrame(max_satisfaction.iloc[:,1:].apply(lambda x: np.sum(x), axis = 0), columns = ['total_satisfaction'])   
    total_max.reset_index(inplace=True)
    total_max = total_max.rename(columns = {'index':'group'})
    total_max =total_max.sort_values(['group']).reset_index()
    
    ##### nash
    sat_cols = [col for col in res_nash.columns if 'rel' not in col and 'sat' in col and 'diff' not in col ]
    total_nash = pd.DataFrame(res_nash[sat_cols].apply(lambda x: np.sum(x), axis = 0), columns=['total_satisfaction'] )
    total_nash.reset_index(inplace = True)
    total_nash = total_nash.rename(columns={'index': 'group'})
    total_nash['method'] = 'Nash'
    total_nash = total_nash.sort_values(['group']).reset_index()
    total_nash = total_nash.drop('index', axis = 1)
    
    total_nash['rel_satisfaction'] = total_nash['total_satisfaction']/ total_max['total_satisfaction']
    total_nash['avg_gain'] = np.mean(res_nash['gain_ratio'])
    
    ###### kalai
    sat_cols = [col for col in res_kalai.columns if 'rel' not in col and 'sat' in col and 'diff' not in col] 
 
    total_kalai = pd.DataFrame(res_kalai[sat_cols].apply(lambda x: np.sum(x), axis = 0), columns = ['total_satisfaction'])   
    total_kalai.reset_index(inplace=True)
    total_kalai = total_kalai.rename(columns = {'index':'group'})
    total_kalai['method'] = 'Kalai'
    total_kalai =  total_kalai.sort_values(['group']).reset_index()
    total_kalai = total_kalai.drop('index', axis = 1)
    

    total_kalai['rel_satisfaction'] = total_kalai['total_satisfaction']/total_max['total_satisfaction']
    total_kalai['avg_gain'] = np.mean(res_kalai['gain_ratio'])
    
    ###### baseline
    sat_cols = [col for col in res_baseline.columns if 'rel' not in col and 'sat' in col ] 
    base = pd.DataFrame(res_baseline[sat_cols].apply(lambda x: np.sum(x), axis = 0), columns = ['total_satisfaction'])
    base.reset_index(inplace = True)
    base = base.rename(columns = {'voter_id':'group-method', 'index':'group-method'})
    base['group'] = base.apply(lambda x:  x['group-method'].split('-')[0], axis = 1)
    base['method'] = base.apply(lambda x: x['group-method'].split('-')[1], axis = 1)
    base = base.drop(['group-method'], axis = 1)
    
    gain_cols = [col for col in res_baseline.columns if 'gain'  in col ] 
    base_gain =   pd.DataFrame(res_baseline[gain_cols].apply(lambda x: np.mean(x), axis = 0), columns = ['avg_gain']) 
    base_gain.reset_index(inplace = True)
    base_gain = base_gain.rename(columns = {'voter_id':'group-method', 'index':'group-method'})
    base_gain['group'] = base_gain.apply(lambda x:  x['group-method'].split('-')[0], axis = 1)
    base_gain['method'] = base_gain.apply(lambda x: x['group-method'].split('-')[1], axis = 1)
    base_gain = base_gain.drop(['group-method', "group"], axis = 1)
    
    
    
    max_exp = total_max[total_max['group'] == 'max_expert_sat']['total_satisfaction'].item()
    max_crowd = total_max[total_max['group'] == 'max_crowd_sat']['total_satisfaction'].item()
    max_area = total_max[total_max['group'] == 'max_satisfaction_area']['total_satisfaction'].item()
    max_sum = total_max[total_max['group'] == 'max_satisfaction_sum']['total_satisfaction'].item()
    
    base['rel_satisfaction'] = np.select(
        [base.group == 'expert_sat' , base.group == 'crowd_sat',  base.group == 'satisfaction_area',  base.group == 'satisfaction_sum'],
        [base.total_satisfaction/max_exp, base.total_satisfaction/max_crowd,  base.total_satisfaction/max_area, base.total_satisfaction/max_sum], default=0)
    
    base = pd.merge(base, base_gain, on = 'method')
    
    ###### weighted
    sat_cols = [col for col in res_weighted.columns if 'rel' not in col and 'sat' in col ] 
    weight = pd.DataFrame(res_weighted[sat_cols].apply(lambda x: np.sum(x), axis = 0), columns = ['total_satisfaction'])
    weight.reset_index(inplace = True)
    weight = weight.rename(columns = {'voter_id':'group-method', 'index':'group-method'})
    weight['group'] = weight.apply(lambda x:  x['group-method'].split('-')[0], axis = 1)
    weight['method'] = weight.apply(lambda x: x['group-method'].split('-')[1], axis = 1)
    weight = weight.drop(['group-method'], axis = 1)
    
    gain_cols = [col for col in res_weighted.columns if 'gain'  in col ] 
    weight_gain =   pd.DataFrame(res_weighted[gain_cols].apply(lambda x: np.mean(x), axis = 0), columns = ['avg_gain']) 
    weight_gain.reset_index(inplace = True)
    weight_gain = weight_gain.rename(columns = {'voter_id':'group-method', 'index':'group-method'})
    weight_gain['group'] = weight_gain.apply(lambda x:  x['group-method'].split('-')[0], axis = 1)
    weight_gain['method'] = weight_gain.apply(lambda x: x['group-method'].split('-')[1], axis = 1)
    weight_gain = weight_gain.drop(['group-method', "group"], axis = 1)
    

    weight['rel_satisfaction'] = np.select(
        [weight.group == 'expert_sat' , weight.group == 'crowd_sat',  weight.group == 'satisfaction_area',  weight.group == 'satisfaction_sum'],
        [weight.total_satisfaction/max_exp, weight.total_satisfaction/max_crowd,  weight.total_satisfaction/max_area, weight.total_satisfaction/max_sum], default=0)
    
    weight = pd.merge(weight, weight_gain, on = 'method')
    
    all_res = pd.concat([total_nash, total_kalai, base, weight])
    gain = all_res.groupby(['method', 'avg_gain']).size().reset_index(name='Freq')
    
    res_relative_sat = all_res.pivot(index = 'method', columns = 'group', values = 'rel_satisfaction')
    
    res_relative_sat.reset_index(inplace = True)
    
    res_relative_sat = pd.merge(res_relative_sat, gain[['method', 'avg_gain']], on = 'method')
    
    #res_relative_sat['avg_gain'] = np.where(res_relative_sat['method'].isin(base_methods), np.abs(res_relative_sat['expert_sat'] - res_relative_sat['crowd_sat'] ), 0)

    return res_relative_sat

def add_median_variation(df_votes, n_repetitions, n_samples):
    '''
    
    Parameters
    ----------
    df_votes : pandas data frame
        Dataframe with grades of voters, where rows are alternatives and columns are voters.
    n_repetitions : int
        Number that represents experiment repetition.
    n_samples : int
        Number that represents thse size of sample.

    Returns
    -------
    res : pandas dataframe
        Standard deviation for each alternative (rows) and for different sample sizes (columns).

    '''
    sel = df_votes.apply(lambda x: np.random.choice(x, (n_repetitions, n_samples)), axis = 1)
    sel2 = sel.apply(lambda x: np.median(x, axis = 1))
    res =sel2.apply(lambda x: np.std(x))
    return res

'''
def nash_results_old(df_alt_votes, optimal_grades , crowd_ids, expert_ids, lambda_expert = 0.5):

    res_nash = pd.DataFrame(columns=(['alternative_id', 'lambda_exp', 'vote', 'expert_sat', 'crowd_sat', 'area']))  
    #res_nash[['lambda_exp', 'vote', 'area']] = np.nan
    #df_alt_votes = df_alt_votes.iloc[0:5,:]
    
    for i in list(df_alt_votes.alternative_id.unique()):
        res = optimal_grades[(optimal_grades['alternative_id'] == i) & (optimal_grades['alpha'] == lambda_expert)]
        votes = df_alt_votes[df_alt_votes['alternative_id'] == i]
        
        print('Alternative to optimize: ', str(i))
    
        n =  expectation_maximization_nash(res, 
                                          lambda_expert, 
                                          votes, 
                                          crowd_ids, expert_ids, 
                                          num_iter = 100, verbose = False)
        
        n = (i,) + n
        res_nash = res_nash.append(pd.Series(list(n), index=res_nash.columns ), ignore_index=True)
    
    #res_nash.to_csv('results/results_nash.csv')   
    return res_nash
'''