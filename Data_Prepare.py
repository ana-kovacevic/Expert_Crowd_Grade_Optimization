# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 20:11:26 2020

@author: akovacevic
"""
import pandas as pd
import numpy as np
import glob

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
def crete_alternatives_map(data, alternative_name = 'media_url'):
    
    data_copy = data.copy()
    
    data_copy['alternative_id'] = data_copy.groupby(alternative_name).ngroup()
    
    alternative_map = data_copy[['media_url', 'alternative_id']].drop_duplicates().reset_index().drop('index', axis = 1)
    alternative_map = alternative_map.rename(columns = {'media_url' : 'alternative_name'})
    
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

#df_crowd = crowd_rest
def prepare_crowd_data(df_crowd, alternative_map):
    """
    Return:
       Crowd data in the transactional form, only for one question, in this case, 'task_1_answer'
       List of attributes (alternatives) names
       Data frame that contains a map of alternative description (media URL to the article) and their ids
    Input:
       df_crowd: data frame, the original form of crowd data
   """
    #cols = ['annotator', 'contributing_users',  'media_url',   'project_id', 'report_id', 'report_title',   'tasks_resolved_count', 'type']
    #quest_ans_cols = ['task_1_question', 'task_1_answer',  'task_2_question', 'task_2_answer', 'task_3_question', 'task_3_answer', 'task_5_question', 'task_5_answer',
    #              'task_7_question', 'task_7_answer',  'task_8_question', 'task_8_answer',   'task_9_question','task_9_answer']

    #complite_cols = cols + quest_ans_cols
    ######## create ids for alternatives values
    data = df_crowd.copy()
    #data['alternative_id'] = data.groupby('media_url').ngroup()
    #data['alternative_id']  = data['alternative_id'].astype("str")
    #data = data.astype({"alternative_id": str})
    # data.dtypes
    #alternative_map = data[['media_url', 'alternative_id']].drop_duplicates().reset_index().drop('index', axis = 1)
    #alternative_map['alternative_id']  = alternative_map['alternative_id'].astype(str)
    #alternative_map.dtypes
    data = pd.merge(data, alternative_map, how = 'inner', left_on = 'media_url', right_on= 'alternative_name')
    
    ###### Prepare crow data / select atributes, rename attributes and create label for crowd row
    final_col = ['annotator', 'alternative_id', 'task_1_answer']
    data = data[final_col]
    data['annotator'] =  data['annotator'] + '_crowd'  #data.annotator.str.replace('CredCo-', 'crowd_')

    
    data = data.rename(columns={'annotator':'voter', 'task_1_answer': 'rate'})
    data = data[data.rate.notnull()]
    data = data.drop_duplicates()
    
    names = list(data['alternative_id'].unique())
    names.sort()
    
    return data, names
    
def remap_answers(df_crowd):
    """
    The procedure contains a dictionary that maps answers to numeric values
    Return:
        Data set with remapped answers
    Input:
        Data set with textual answers
    """
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
    
    data = df_crowd.copy()
    
    for di in ans_dicts:
        for key, values in di.items():
            data.replace({values : key}, inplace = True) 
            
    return data

def prepare_expert_data(data_folder, alternative_map):
    """
    The procedure is used to read and prepare experts answers

    Parameters
    ----------
    data_folder : String
        The path where all expert files are stored.
    alternative_map : pandas dataframe, 
        Alternative map created using crowd data to assign same ids to URL media.

    Returns
    -------
    df_expert : pandas df
        Expert data set in transactional form
    df_science : TYPE
        DESCRIPTION.
    df_journal : TYPE
        DESCRIPTION.

    """  

    expert_cols = ['ID', 'URL', 'Score', 'voter']
    
    science = pd.DataFrame()
    i = 0
    for file in glob.glob( data_folder + "*Science*.csv"):
        i = i + 1
        df = pd.read_csv(file)
        df['voter'] = 'Science-' + str(i) + '_expert'
        if science.empty:
            science = df
        else:
            science = pd.concat([science, df], ignore_index = True)
            
    journal = pd.DataFrame()
    i = 0
    for file in glob.glob( data_folder + "*Journalism*.csv"):
        i = i + 1
        df = pd.read_csv(file)
        df['voter'] = 'Journalism-' + str(i) + '_expert'
        if journal.empty:
            journal = df
        else:
            journal = pd.concat([journal, df], ignore_index = True)
            
    df_science = science[expert_cols]
    df_journal = journal[expert_cols]
    
    df_journal['Score'] = journal['Score'].str.split('\)', expand = True)[0]
    
    df_science = pd.merge(alternative_map, df_science, how = 'inner', left_on = 'alternative_name', right_on =  'URL')[['alternative_id', 'URL', 'Score', 'voter']]
    df_journal = pd.merge(alternative_map, df_journal, how = 'inner', left_on = 'alternative_name', right_on =  'URL')[['alternative_id', 'URL', 'Score', 'voter']]
    
    df_expert = pd.concat([df_science, df_journal]).reset_index().drop('index', axis = 1)
    df_expert = df_expert[['voter', 'alternative_id', 'Score']]
    
    df_expert = df_expert.rename(columns={ 'Score': 'rate'})
    df_expert = df_expert[df_expert.rate.notnull()]
    df_expert = df_expert.drop_duplicates()
    
    df_science = df_expert[df_expert['voter'].str.contains('Science')]
    df_journal = df_expert[df_expert['voter'].str.contains('Journalism')]
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


