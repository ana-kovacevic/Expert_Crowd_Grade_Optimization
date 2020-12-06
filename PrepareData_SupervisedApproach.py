# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 18:01:16 2020

@author: akovacevic
"""
import pandas as pd
import numpy as np


import itertools 
#from itertools import permutations  

#df = df_expert
def aggregate_statistics_per_alternatives(df, voter_map, split_char = '-' ):
    """
    Procedure aggregate expert rating 
    """
    stat_data = df.copy()
    
    stat_data['voter_type'] = stat_data['voter'].str.split(split_char, expand = True)[0]
    stat_data['rate'] = stat_data['rate'].astype(float)
   
  
    one_expert = stat_data.groupby("vote").agg(average_rate=pd.NamedAgg(column='rate', aggfunc=np.mean)).reset_index()
    group_expert = stat_data.groupby(["vote", "voter_type"]).agg(group_rate = pd.NamedAgg(column='rate', aggfunc=np.mean)).reset_index()

    stat_data = pd.merge(stat_data, one_expert, how='left',left_on = ['vote'], right_on = ['vote'])
    stat_data = pd.merge(stat_data, group_expert, how='left',left_on = ['vote', 'voter_type'], right_on = ['vote', 'voter_type'])
    
    std_one = stat_data.groupby("vote").agg(std_rate=pd.NamedAgg(column='rate', aggfunc=np.std)).reset_index()
    #std_group = stat_data.groupby(["vote", "expert_type"]).agg(std_group_rate=pd.NamedAgg(column='rate', aggfunc=np.std )).reset_index()
    #std_group = stat_data.groupby(["vote", "expert_type"]).agg({'rate': np.std }).reset_index()
    std_group = stat_data.groupby(["vote", "voter_type"]).agg({'rate': lambda x: np.std(x, ddof=0)}).reset_index() 
    std_group = std_group.rename(columns = {'rate' : 'std_group_rate'})

    stat_data = pd.merge(stat_data, std_one, how = 'left', left_on= ['vote'], right_on= ['vote'])
    stat_data = pd.merge(stat_data, std_group, how = 'left', left_on= ['vote', 'voter_type'], right_on= ['vote', 'voter_type'])
    
    stat_data = pd.merge(stat_data, voter_map, how = 'inner', left_on= 'voter', right_on= 'voter')
    stat_data = stat_data.rename(columns=({'rate' : 'expert_rate', 'voter_id' : 'expert_id'}))
    
    
    # expert_rated_df =  pd.merge(expert_rates_mean, expert_rated_alternatives, how='inner', on='vote', #left_on=None, right_on=None,
    #      left_index=False, right_index=False, #sort=True,  #suffixes=('_x', '_y'), copy=True, indicator=False,
    #      validate=None)
    
    return stat_data



def calculate_differences(df_votes):
    
    data = df_votes.copy()
    
    data['abs_diff_each'] = np.absolute(data['rate'] - data['expert_rate'])
    data['abs_diff_one'] = np.absolute(data['rate'] - data['average_rate'])
    data['abs_diff_group'] = np.absolute(data['rate'] - data['group_rate'])
    
    
    return data
    
#### exert - alternative ratings (on expert level aggregated to one expert or aggregated to group expert)
def create_data_for_predicting_differences(df_expert, df_crowd, user_factors, alt_factors, 
                                           voters_lookup, alts_lookup, voter_map, 
                                           crowd_ids, num_factors):
    df_expert_stats = aggregate_statistics_per_alternatives(df_expert, voter_map)
    #crowd_stats = aggregate_statistics_per_alternatives(df_crowd_2020, voter_map)

    df_votes = pd.merge(df_crowd, df_expert_stats.drop('voter', axis=1), how = "inner", left_on = 'vote', right_on ='vote' )
    df_votes = pd.merge(df_votes, voter_map, how = 'inner', left_on= 'voter', right_on= 'voter')

    #cols = ['voter_id', 'voter', 'vote', 'expert_id', 'expert_type',  'rate', 'expert_rate',  'average_rate', 'group_rate']
    df_votes = df_votes.reindex(columns=['voter_id', 'voter', 'vote', 'expert_id', 'expert_type',  'rate', 'expert_rate',  'average_rate', 'group_rate'])
    # df_crowd_alt =  pd.merge(df_expert_stats, df_crowd_2020, how='inner', on='vote', #left_on=None, right_on=None,
    #          left_index=False, right_index=False, #sort=True,  #suffixes=('_x', '_y'), copy=True, indicator=False,
    #          validate=None)

    df_crowd_alt =  pd.merge(df_votes, voters_lookup, how='inner', on='voter', #left_on=None, right_on=None,
         left_index=False, right_index=False,suffixes=('_map', '_lookup'), #sort=True,  # copy=True, indicator=False,
         validate=None)


    user_factors_df = pd.DataFrame(user_factors[crowd_ids])
    col_users = ['UF' + str(x) for x in range(1, num_factors+1)]

    user_factors_df.columns = col_users
    user_factors_df['voter_id'] = crowd_ids

    alt_factors_df = pd.DataFrame(alt_factors)
    col_alts = ['AF' + str(x) for x in range(1, num_factors+1)]
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

    final_data = calculate_differences(df_crowd_alt)
    
    return final_data

def create_train_data_for_predicting_rates(df_expert_crowd, user_factors, alt_factors, voters_lookup, alts_lookup):
    
    final_data = df_expert_crowd.copy()
    num_factors = user_factors.shape[1]
    
    user_factors_df = pd.DataFrame(user_factors)
    col_users = ['UF' + str(x) for x in range(1,num_factors+1)]

    user_factors_df.columns = col_users
    user_factors_df['voter_id'] = voters_lookup['voter_id']

    alt_factors_df = pd.DataFrame(alt_factors)
    col_alts = ['AF' + str(x) for x in range(1,num_factors+1)]
    alt_factors_df.columns = col_alts

    alt_factors_df['alternative_id'] = alts_lookup['alternative_id']

    #np.sum(alts_lookup['alternative_id'] != alts_lookup['alternative'])
    
    final_data =  pd.merge(df_expert_crowd, voters_lookup, how='inner', on='voter_id', #left_on=None, right_on=None,
         left_index=False, right_index=False,suffixes=('_map', '_lookup'), #sort=True,  # copy=True, indicator=False,
         validate=None)

    final_data =  pd.merge(final_data, user_factors_df, how='inner', left_on='voter_id', right_on='voter_id',  #on='voter_id',
                             left_index=False, right_index=False, #sort=True,  #suffixes=('_x', '_y'), copy=True, indicator=False,
                             validate=None)
    #len(df_crowd_alt['vote'].unique())
    final_data =  pd.merge(final_data, alt_factors_df, how='inner', # on='voter_id', 
                             left_on='alternative_id', right_on='alternative_id',
                             left_index=False, right_index=False, #sort=True,  #suffixes=('_x', '_y'), copy=True, indicator=False,
                             validate=None)
    final_data = final_data.astype({'rate': 'int32'})#.dtypes

    return final_data 

def create_test_data_for_predicting_rates(final_data, user_factors, alt_factors, voters_lookup, alts_lookup):
    
    num_factors = user_factors.shape[1]
    user_factors_df = pd.DataFrame(user_factors)
    col_users = ['UF' + str(x) for x in range(1, num_factors+1)]

    user_factors_df.columns = col_users
    user_factors_df['voter_id'] = voters_lookup['voter_id']

    alt_factors_df = pd.DataFrame(alt_factors)
    col_alts = ['AF' + str(x) for x in range(1, num_factors+1)]
    alt_factors_df.columns = col_alts

    alt_factors_df['alternative_id'] = alts_lookup['alternative_id']

    
    unique_combinations = list(itertools.product(voters_lookup['voter_id'], alts_lookup['alternative_id']))
    all_combinations = pd.DataFrame(unique_combinations, columns= ['voter_id', 'alternative_id'] )  

    
    filter_data = pd.merge(all_combinations, final_data, how='left', left_on=['voter_id', 'alternative_id'], right_on=['voter_id', 'alternative_id'],  #on='voter_id',
                             left_index=False, right_index=False, indicator=True, #sort=True,  #suffixes=('_x', '_y'), copy=True, 
                             validate=None)  
    filter_data = filter_data[filter_data._merge == 'left_only'][['voter_id', 'alternative_id']]
    

    filter_data =  pd.merge(filter_data, user_factors_df, how='inner', left_on='voter_id', right_on='voter_id',  #on='voter_id',
                             left_index=False, right_index=False, #sort=True,  #suffixes=('_x', '_y'), copy=True, indicator=False,
                             validate=None)
    #len(df_crowd_alt['vote'].unique())
    filter_data =  pd.merge(filter_data, alt_factors_df, how='inner', # on='voter_id', 
                             left_on='alternative_id', right_on='alternative_id',
                             left_index=False, right_index=False, #sort=True,  #suffixes=('_x', '_y'), copy=True, indicator=False,
                             validate=None)
    
    return filter_data

def prepare_data_for_grade_optimization(all_pred, test_combinations, df_expert, df_crowd, voters_lookup, expert_ids, crowd_ids):
    test_combinations['rate'] = all_pred

    test_pred_expert = test_combinations[test_combinations['voter_id'].isin(expert_ids)]
    test_pred_crowd = test_combinations[test_combinations['voter_id'].isin(crowd_ids)]

    train_crowd = pd.merge(df_crowd, voters_lookup, on = 'voter_id')[['voter_id', 'alternative_id', 'rate']]
    #train_crowd = train_crowd.rename(columns=({ 'vote' : 'alternative_id'}))
    
    train_expert = pd.merge(df_expert, voters_lookup, on = 'voter_id')[['voter_id', 'alternative_id', 'rate']]
    #train_expert = train_expert.rename(columns=({ 'vote' : 'alternative_id'}))
        
    
    
    all_exp_grades = pd.concat([train_expert, test_pred_expert], axis = 0, ignore_index=True)
    all_crowd_grades = pd.concat([train_crowd, test_pred_crowd], axis = 0, ignore_index=True)
    
    all_exp_grades['rate'] = all_exp_grades['rate'].astype(float)
    
    return all_exp_grades, all_crowd_grades
    
    
        