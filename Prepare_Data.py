# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 20:11:26 2020

@author: akovacevic
"""
import pandas as pd
import numpy as np
import glob
import os
import re

import scipy.sparse as sparse

#### task_1_question ----- 'Rate your impression of the credibility of this article'
qu1 = {1 : 'Very low credibility', 
2 : 'Somewhat low credibility', 
3 : 'Medium credibility', 
4 : 'Somewhat high credibility', 
5 : 'Very high credibility' }

#### task_2_question ---- 'Is the language of the headline extremely negative, extremely positive, or somewhere in the middle?'
qu2 = {1 : 'Extremely negative', 
2 : 'Somewhat negative' , 
3 : 'Neither negative nor positive', 
4 : 'Somewhat positive' ,
5 : 'Extremely positive' }

#### task_3_question  ----- 'Rate the degree to which the headline is clickbait'  
qu3 = {1 : 'Not at all clickbaity',
2 : 'A little bit clickbaity' ,
3 : 'Medium clickbaity',
4 : 'Somewhat clickbaity',
5 : 'Very much clickbaity'}

#### task_5_question  ---- 'Rate the representativeness of the article title'
qu5 = {1 : 'Completely Unrepresentative',
2 : 'Somewhat Unrepresentative',
3 : 'Medium Unrepresentative',
4 : 'Somewhat Representative',
5 : 'Completely Representative'} 

ans_dicts = [qu1, qu2, qu3, qu5]
#### task_7_question  ----- 'Is the language of the subheadline extremely negative, extremely positive, or somewhere in the middle?'
#qu7 = {1 : 'Extremely negative',
#2 : 'Somewhat negative',
#3 : 'Neither negative nor positive',
#4 : 'Somewhat positive',
#5 : 'Extremely positive'}
#
##### task_8_question  ----  'Is the article you are reading a news piece or an opinion piece?'
##['Not sure' 'News' 'Opinion' nan]
#
#### task_9_question ----  'Rate your impression of the credibility of this article (2)'
#qu9 = {1 : 'Very low credibility',
#2 : 'Somewhat low credibility',
#3 :  'Medium credibility',
#4 : 'Somewhat high credibility',
#5 : 'Very high credibility'}

#df_expert = df_expert.rename(columns={'Awarding_country':'voter', 'Receiving_country':'vote', 'Jury_points': 'Quantity'})
#df_crowd = df_crowd.rename(columns={'Awarding_country':'voter', 'Receiving_country':'vote', 'Televoting_points': 'Quantity'})


#len(df_crowd['annotator'].unique())
def crete_alternatives_map(data, alternative_name = 'URL'):
    
    data_copy = data.copy()
    
    # Assign IDs (preserve original order)
    data_copy['alternative_id'] = data_copy.groupby(alternative_name, sort=False).ngroup()
    
    # Create mapping
    alternative_map = (
        data_copy[[alternative_name, 'alternative_id']]
        .drop_duplicates()
        .rename(columns={alternative_name: 'alternative_name'})
        .reset_index(drop=True)
    )
    
    return alternative_map

def crete_voter_map(dfs, voter_name = 'voter'):
    '''
      This function takes the list of dataframe and create unique map of all users

    Parameters
    ----------
    dfs : TYPE
        DESCRIPTION.
    voter_name : TYPE, optional
        DESCRIPTION. The default is 'voter'.

    Returns
    -------
    voter_map : TYPE
        DESCRIPTION.

    '''
  
    
    data_copy = pd.concat(dfs, ignore_index = True)
    
    data_copy['voter_id'] = data_copy.groupby(voter_name).ngroup()
    
    voter_map = data_copy[['voter', 'voter_id']].drop_duplicates().reset_index().drop('index', axis = 1)
    
    return voter_map

#df_crowd1 = crowd_2020
#df_crowd2 = crowd_2019
def prepare_crowd_data(df_crowd1, df_crowd2):
    """
    Return:
       Crowd data in transactional form for task_1_answer only.
    Input:
       df_crowd1, df_crowd2: original crowd dataframes
    """

    crowd_1 = df_crowd1[['annotator', 'media_url', 'task_1_answer']].copy()
    crowd_2 = df_crowd2[['annotator', 'media_url', 'task_1_answer']].copy()

    crowd_1 = remap_answers(crowd_1)
    crowd_2 = remap_answers(crowd_2)

    # Union
    df_crowd = pd.concat([crowd_1, crowd_2], ignore_index=True).drop_duplicates()

    # Remove rows with missing essential values
    df_crowd = df_crowd[df_crowd['annotator'].notna()]
    df_crowd = df_crowd[df_crowd['task_1_answer'].notna()]
    df_crowd = df_crowd[df_crowd['media_url'].notna()]

    # Add crowd suffix
    df_crowd['annotator'] = df_crowd['annotator'].astype(str) + '_crowd'

    # Rename columns
    df_crowd = df_crowd.rename(columns={
        'annotator': 'voter',
        'task_1_answer': 'rate',
        'media_url': 'URL'
    })

    # Ensure rate is numeric
    df_crowd['rate'] = pd.to_numeric(df_crowd['rate'], errors='coerce')
    df_crowd = df_crowd[df_crowd['rate'].notna()]
    
    df_crowd = df_crowd.reset_index(drop=True)

    return df_crowd    
def remap_answers(data):
    """
    Maps textual answers to numeric values.
    """

    qu1 = {
        'Very low credibility': 1,
        'Somewhat low credibility': 2,
        'Medium credibility': 3,
        'Somewhat high credibility': 4,
        'Very high credibility': 5
    }

    qu2 = {
        'Extremely negative': 1,
        'Somewhat negative': 2,
        'Neither negative nor positive': 3,
        'Somewhat positive': 4,
        'Extremely positive': 5
    }

    qu3 = {
        'Not at all clickbaity': 1,
        'A little bit clickbaity': 2,
        'Medium clickbaity': 3,
        'Somewhat clickbaity': 4,
        'Very much clickbaity': 5
    }

    qu5 = {
        'Completely Unrepresentative': 1,
        'Somewhat Unrepresentative': 2,
        'Medium Unrepresentative': 3,
        'Somewhat Representative': 4,
        'Completely Representative': 5
    }

    df = data.copy()

    # Apply mapping per column (adjust column names!)
    if 'task_1_answer' in df.columns:
        df['task_1_answer'] = df['task_1_answer'].map(qu1)

    if 'task_2_answer' in df.columns:
        df['task_2_answer'] = df['task_2_answer'].map(qu2)

    if 'task_3_answer' in df.columns:
        df['task_3_answer'] = df['task_3_answer'].map(qu3)

    if 'task_5_answer' in df.columns:
        df['task_5_answer'] = df['task_5_answer'].map(qu5)

    return df



def remap_answers_tx(df):
    """
    The procedure contains a dictionary that maps answers to numeric values
    Return:
        Data set with remapped answers
    Input:
        Data set with textual answers
    """
    #### task_1_question ----- 'Rate your impression of the credibility of this article'
    ans_dicts = {1:'Важно је', 2:'Неутралан', 3: 'Није ми важно' }
    
    #### task_2_question ---- 'Is the language of the headline extremely negative, extremely positive, or somewhere in the middle?'
    data = df.copy()
    
    for key, values in ans_dicts.items():
        data.replace({values : key}, inplace = True) 
            
    return data

def prepare_expert_data(data_folder):
    """
    Read and prepare expert answers.

    Parameters
    ----------
    data_folder : str
        Path where all expert files are stored.
    alternative_map : pandas.DataFrame
        Map created using crowd data to assign same ids to media URLs.

    Returns
    -------
    df_expert : pandas.DataFrame
        Combined expert dataset in transactional form.
    df_science : pandas.DataFrame
        Science expert dataset in transactional form.
    df_journal : pandas.DataFrame
        Journalism expert dataset in transactional form.
    """

    expert_cols = [ 'URL', 'Score', 'voter']

    # -------------------------
    # Science files
    # -------------------------
    sci_files = [
        os.path.join(data_folder, f)
        for f in os.listdir(data_folder)
        if 'science' in f.lower() and f.endswith('.csv')
    ]

    if not sci_files:
        raise ValueError("No matching science files found.")

    sci_dfs = []

    for file in sci_files:
        sci_df = pd.read_csv(file)
        filename = os.path.basename(file)

        match = re.search(r'#(\d+)', filename)
        num = match.group(1) if match else "unknown"

        sci_df['voter'] = f"Science_{num}_expert"
        sci_dfs.append(sci_df)

    sci_final = pd.concat(sci_dfs, ignore_index=True).drop_duplicates()
    science = sci_final.dropna(subset=['URL', 'Score', 'voter']).copy()

    # -------------------------
    # Journalism files
    # -------------------------
    jrn_files = [
        os.path.join(data_folder, f)
        for f in os.listdir(data_folder)
        if 'journalism' in f.lower() and f.endswith('.csv')
    ]

    if not jrn_files:
        raise ValueError("No matching journalism files found.")

    jrn_dfs = []

    for file in jrn_files:
        jrn_df = pd.read_csv(file)
        filename = os.path.basename(file)

        match = re.search(r'#([A-Za-z])', filename)
        if match:
            letter = match.group(1).upper()
            num = ord(letter) - ord('A') + 1
        else:
            num = "unknown"

        jrn_df['voter'] = f"Journalism_{num}_expert"
        jrn_dfs.append(jrn_df)

    jrn_final = pd.concat(jrn_dfs, ignore_index=True).drop_duplicates()
    journal = jrn_final.dropna(subset=['URL', 'Score', 'voter']).copy()

    # -------------------------
    # Select columns
    # -------------------------
    df_science = science[expert_cols].copy()
    df_journal = journal[expert_cols].copy()

    # Make score numeric
    df_science['Score'] = pd.to_numeric(df_science['Score'], errors='coerce')

    df_journal['Score'] = pd.to_numeric(
        df_journal['Score'].astype(str).str.extract(r'(\d+)')[0],
        errors='coerce'
    )

    # -------------------------
    # Merge with alternative map
    # -------------------------
    
    # -------------------------
    # Combine
    # -------------------------
    df_expert = pd.concat([df_science, df_journal], ignore_index=True)
    #df_expert = df_expert[['voter', 'alternative_id', 'Score']]
    df_expert = df_expert.rename(columns={'Score': 'rate'})
    df_expert = df_expert[df_expert['rate'].notna()].drop_duplicates()

    # Separate outputs
    df_science = df_expert[df_expert['voter'].str.contains('Science', na=False)].copy()
    df_journal = df_expert[df_expert['voter'].str.contains('Journalism', na=False)].copy()

    return df_expert, df_science, df_journal           

# df_trans = pd.concat([all_crowd_grades, all_exp_grades]) #.reset_index() #df_expert
# df_trans = df_crowd_2020
# alternative_space = list(voters_lookup['voter_id'])  alt_names  

def get_aggregated_data(df_trans, column_order, index_column = 'voter', column= 'alternative_id', value = 'rate'):
    """
    Returns:
        Aggregated DF: Sparse Data Frame where every column represents alternatives and every row is a voter. 
        names: List Names of alternatives as columns
    Input:
        df_trans: Data Frame of votes given in the transactional form with quantity
        index_column: String representing column name to be  
        columns: String of column name to be pivoted
        values: String of column name for values 
    """
    #df_trans = pd.merge(df_trans, voter_map, how = "left", on = "voter")
    #df_trans = df_trans.drop('voter', axis = 1)
    #df_trans = df_trans.rename(columns={'voter_id' : 'voter'})
    vote_data = df_trans.pivot(index = index_column, columns = column, values = value).reindex(column_order, axis=1)
    #vote_data = vote_data.fillna(0)
    #vote_sample = vote_data.iloc[0:6, 0:5]
    
    #a = vote_sample.groupby([index_column]).sum(axis = 0).reset_index()

    data_agg = vote_data.groupby(index_column).sum().reset_index()
    
    #data_agg = vote_data.rename_axis(None)
    
    #data_agg = data_agg.reset_index()
    #data_agg = data_agg.rename(columns={index: 'voter'})
    #data_agg = data_agg.sort_index(axis=1)
    
    #names = list(data_agg.columns)
    #names.remove('voter')
    
    return data_agg #, names #, OHE  

#all_crowd_grades.pivot(index = index_column, columns = column, values = value)
#all_exp_grades.pivot(index = index_column, columns = column, values = value)

# data = expert_crowd_agg
def create_ratings_and_mapping(data, alt_names, voter_col = 'voter'):
    #data = expert_crowd_agg
    ratings = np.array(data[alt_names], dtype=np.float64)
    
    mapa_user = pd.DataFrame(columns = ['voter', 'voter_id'])
    mapa_user['voter'] = data[voter_col]
    mapa_user['voter_id'] = data.groupby(voter_col).ngroup()
    
    mapa_alt = pd.DataFrame(alt_names, columns = ['alternative'])
    mapa_alt['alternative_id'] = mapa_alt.groupby('alternative').ngroup()
    
    return ratings, mapa_alt, mapa_user

# size = 0.1 mask_test_size
def train_test_split(ratings, size, restricted_count):
    
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    
    # i,j = np.nonzero(ratings)

    # ix = np.random.choice(len(i), int(np.floor(size * len(i))), replace=False)
   
    # train[i[ix], j[ix]] = 0.
    # test[i[ix], j[ix]] = ratings[i[ix], j[ix]]
    for alt in range(ratings.shape[1]):
        alt_num =  len([a for a in ratings[:, alt] if a != 0])
        num_size = np.round(alt_num * size).astype('int')
        if alt_num < restricted_count:
            continue
        else:
            test_ratings = np.random.choice(ratings[:, alt].nonzero()[0], size=num_size, replace=False)
            train[test_ratings, alt] = 0.
            test[test_ratings, alt] = ratings[test_ratings, alt]
        
    #user = 0
    # for user in range(ratings.shape[0]):
    #     alt_num =  len([a for a in ratings[user, :] if a != 0])
    #     num_size = np.round(alt_num * size).astype('int')
    #     if alt_num < restricted_count:
    #         continue
    #     else:
    #         test_ratings = np.random.choice(ratings[user, :].nonzero()[0], size=num_size, replace=False)
    #         train[user, test_ratings] = 0.
    #         test[user, test_ratings] = ratings[user, test_ratings]
        
    # Test and training are truly disjoint
    assert(np.all((train * test) == 0)) 
    return train, test


def calculate_sparsity(df_sparse):
        
    # Check sparsity of data
    matrix_size = df_sparse.shape[0]*df_sparse.shape[1] # Number of possible interactions in the matrix
    num_purchases = len(df_sparse.nonzero()[0]) # Number of items interacted with
    sparsity = 100*(1 - (num_purchases/matrix_size))
    return sparsity

def get_user_ids_from_mapping(data_lookup, string_lookup, name_col = 'voter', id_col= 'voter_id' ):
    
    #### extract expert ids for similarity
    data_filterd = data_lookup[data_lookup[name_col].str.contains(string_lookup)]
    ids = data_filterd.apply(lambda x: x[id_col] , axis = 1) 
    ids = ids.tolist()
    return ids

def all_mathods_optimal_grades(df_list):
    
    res_list = list()
    for i in range(len(df_list)):
        df = df_list[i]
        grade_cols = [col for col in df.columns if 'optimal_grade' in col or 'vote' in col]
        if  len(grade_cols) == 0:
            grade_cols = [col for col in df.columns if '-' not in col and 'alt' not in col ]
        grade_cols.insert(0, 'alternative_id')    
        
        df_filterd = df[grade_cols]  
        res_list.append(df_filterd)
        
    return res_list
            

def calculate_crowd_exper_diff(df_crowd_sample, df_selected_expert, df_alt_votes, crowd_ids):    
    
    crowd_original_medians = df_crowd_sample.groupby('alternative_id').agg('median').reset_index()
    crowd_original_medians = crowd_original_medians.rename(columns = { 'rate':'crowd_original_median'})
    
    
    expert_original_medians = df_selected_expert.groupby('alternative_id').agg('median').reset_index()
    expert_original_medians = expert_original_medians.rename(columns = {'rate':'expert_original_median'})
    
    original_medians = pd.merge(expert_original_medians, crowd_original_medians, how = 'inner', on = 'alternative_id')
    
    
    round_medians = df_alt_votes.copy().round()
    #round_medians['crowd_median'] = df_alt_votes.round()[crowd_ids].apply(lambda x: np.median(x), axis = 1)
    round_medians['crowd_median'] = round_medians[crowd_ids].apply(lambda x: np.median(x), axis = 1)
    round_medians = round_medians[['alternative_id', 'crowd_median']]
    
    original_all = pd.merge(original_medians, round_medians, how = 'inner', on = 'alternative_id')
    
    original_all['original_diff'] = np.abs(np.array(original_all['expert_original_median']) - np.array(original_all['crowd_original_median']))
    original_all['estimated_diff'] = np.abs(np.array(original_all['expert_original_median']) - np.array(original_all['crowd_median']))
    
    orig_diff = np.mean(np.array(original_all['original_diff']))
    estm_diff = np.mean(np.array(original_all['estimated_diff']))
    
    return orig_diff, estm_diff


