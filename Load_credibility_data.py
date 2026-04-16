# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 13:40:37 2020

@author: akovacevic
"""
import pandas as pd
import numpy as np

from Prepare_Data import crete_alternatives_map
from Prepare_Data import prepare_crowd_data
from Prepare_Data import remap_answers
from Prepare_Data import prepare_expert_data



def read_data_credibility():
    '''
    Procedure is used to transform and read Credibility data.
    
    Returns
    -------
    alternative_map : pandas dataframe
        Generate id for each alternative name (media_url) and map create map between id and url.
    alt_names : list
        List of all alternatives.
    df_crowd : pandas dataframe
         Transactional data of all crowd grades (votes) .
    df_expert : pandas dataframe
        Transactional data of all expert grades (votes).
    df_crowd_2020 : pandas dataframe
        Transactional data of crowd grades (votes) for alternatives that were evaluated by experts.
    df_science : pandas dataframe
        Transactional data of science expert grades (votes).
    df_journal : pandas dataframe
        Transactional data of journal expert grades (votes).

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
    crowd_2019 = pd.read_csv(data_folder_2019 + crowd_file_2019)
    
    df_crowd = prepare_crowd_data(crowd_2020, crowd_2019)
    
    
    
    
    #### Read and prepare expert data
    df_expert, df_science, df_journal = prepare_expert_data(data_folder_2020)
    exp_urls = list(df_expert['URL'].unique()) # alts that experts gave opinion on
    
   
    #### filter alternatives same as experts
    #crowd_2020 = crowd_2020[crowd_2020['media_url'].isin(exp_urls)]
    #merged_crowd = pd.merge(crowd_all, crowd_2020, how='left',left_on = ['annotator', 'media_url'], right_on = ['annotator', 'media_url'], indicator = True)
    #### take only users that are not part of earlier 
    #crowd_rest = crowd_all[merged_crowd['_merge'] == 'left_only']

    #### ALL DATA
    df_all = pd.concat([df_crowd, df_expert], ignore_index=True)
    # remove duplicates
    df_all = df_all.drop_duplicates().reset_index(drop=True)
    
    ##### save map of all alternatives
    alternative_map = crete_alternatives_map(df_all)
    #alternative_map =  alternative_map.rename(columns={"media_url": "alternative_name"})
    alts_dict = dict(zip(alternative_map['alternative_id'] , alternative_map['alternative_name']))
    
    alt_names = list(alternative_map['alternative_id'].unique())
    alt_names.sort()
    
    df_all = pd.merge(df_all, alternative_map, how = 'inner', left_on = 'URL', right_on= 'alternative_name')[['voter', 'URL', 'rate', 'alternative_id']]
    df_crowd = pd.merge(df_crowd, alternative_map, how = 'inner', left_on = 'URL', right_on= 'alternative_name')[['voter', 'URL', 'rate', 'alternative_id']]
    df_expert = pd.merge(df_expert, alternative_map, how = 'inner', left_on = 'URL', right_on= 'alternative_name')[['voter', 'URL', 'rate', 'alternative_id']]
    df_science = pd.merge(df_science, alternative_map, how = 'inner', left_on = 'URL', right_on= 'alternative_name')[['voter', 'URL', 'rate', 'alternative_id']]
    df_journal = pd.merge(df_journal, alternative_map, how = 'inner', left_on = 'URL', right_on= 'alternative_name')[['voter', 'URL', 'rate', 'alternative_id']]
    
    #crowd_rest = prepare_crowd_data(crowd_rest, alternative_map)[0] 
    #crowd_rest = remap_answers(crowd_rest)
    #alternative_map, alt_names, df_crowd, df_expert, df_crowd_2020 = read_data_credibility()
    
    
    return  df_crowd, df_science, df_journal, alternative_map,  alts_dict, alt_names, df_expert,  df_all

        
