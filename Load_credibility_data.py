# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 13:40:37 2020

@author: akovacevic
"""
import pandas as pd
import numpy as np

from Data_Prepare import crete_alternatives_map
from Data_Prepare import prepare_crowd_data
from Data_Prepare import remap_answers
from Data_Prepare import prepare_expert_data



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
    crowd_all = pd.read_csv(data_folder_2019 + crowd_file_2019)
    crowd_all = remap_answers(crowd_all)

    ##### save map of all alternatives
    alternative_map = crete_alternatives_map(crowd_all)
    #alternative_map =  alternative_map.rename(columns={"media_url": "alternative_name"})
    alts_dict = dict(zip(alternative_map['alternative_id'] , alternative_map['media_url']))

    #### Read and prepare expert data
    df_expert, df_science, df_journal = prepare_expert_data(data_folder_2020, alternative_map)
    exp_alt = list(df_expert['vote'].unique()) # alts that experts gave opinion on
    exp_urls = [alts_dict.get(e,'') for e in exp_alt]  # alts that experts gave opinion on

    #### filter alternatives same as experts
    crowd_2020 = crowd_2020[crowd_2020['media_url'].isin(exp_urls)]
    #merged_crowd = pd.merge(crowd_all, crowd_2020, how='left',left_on = ['annotator', 'media_url'], right_on = ['annotator', 'media_url'], indicator = True)
    #### take only users that are not part of earlier 
    #crowd_rest = crowd_all[merged_crowd['_merge'] == 'left_only']

    df_crowd, alt_names = prepare_crowd_data(crowd_all, alternative_map)
    df_crowd = remap_answers(df_crowd)

    df_crowd_2020, _ = prepare_crowd_data(crowd_2020, alternative_map)
    df_crowd_2020 = remap_answers(df_crowd_2020)
    #crowd_rest = prepare_crowd_data(crowd_rest, alternative_map)[0] 
    #crowd_rest = remap_answers(crowd_rest)
    #alternative_map, alt_names, df_crowd, df_expert, df_crowd_2020 = read_data_credibility()
    return alternative_map, alt_names, df_crowd, df_expert, df_crowd_2020, df_science, df_journal

        
