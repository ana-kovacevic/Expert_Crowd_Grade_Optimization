# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 18:56:13 2020

@author: akovacevic
"""


### Set working directory
import os
os.chdir('F:\PROJEKTI\ONR_FON\Experiments\Credibility-Factors2020')
#### Import libreries
import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np

data_folder = 'credibilitycoalition-credibility-factors2020\\'
file = 'CredCo 2019 Study - CplusJ 2020 subset.csv'

data_folder_2019 = 'credibilitycoalition-2019-study\\'
full_file = 'CredCo_2019_Crowd Annotators-FULL.csv'



crowd_2020 = pd.read_csv(data_folder+file)
crowd_map = pd.read_csv(data_folder + 'CredCo Study 2019 Crowd Annotators -simple.csv')
crowd_all = pd.read_csv(data_folder_2019 + full_file)

sci_1 = pd.read_csv(data_folder + 'No Notes Copy of CredCo Study 2019 Expert Scores - Science #1.csv')
sci_2 = pd.read_csv(data_folder + 'No Notes Copy of CredCo Study 2019 Expert Scores - Science #2.csv' )
sci_3 = pd.read_csv(data_folder + 'No Notes Copy of CredCo Study 2019  Expert Scores - Science #3.csv')

jour_1 = pd.read_csv(data_folder + 'No Notes Copy of CredCo Study 2019 Expert Scores - Journalism #A.csv')
jour_2 = pd.read_csv(data_folder + 'No Notes Copy of CredCo Study 2019 Expert Scores - Journalism #B.csv')
jour_3 = pd.read_csv(data_folder + 'No Notes Copy of CredCo Study 2019 Expert Scores - Journalism #C.csv')

### questions column
crowd = crowd_all
crowd = crowd_2020
                     
question_cols = [col for col in crowd.columns if 'question' in col]
answer_cols = [col for col in crowd.columns if 'answer' in col and len(col) < 18]

for col in question_cols:
    print(' ------------------ Unique question of ' + col + ' are: -----------------')
    print(crowd_2020[col].unique())
                     
for col in answer_cols:
    print(' ------------------ Unique answer of ' + col + ' are: -----------------')
    print(crowd_2020[col].unique())
    

for i in range(len(answer_cols)):
    quest = question_cols[i]
    ans = answer_cols[i]
    print(' ------------------ Question of ' + quest + ' is: -----------------')
    print(crowd_2020[quest].unique())
    print('Possible answers to the question are: ')
    print(crowd_2020[ans].unique())

#cols = ['annotator', 'contributing_users', 'media_content', 'media_url', 'notes_count', 'notes_ugc_count', 'project_id', 
#        'report_date', 'report_id', 'report_title', 'tags', 'tasks_count', 'tasks_resolved_count', 'time_delta_to_first_status', 
#        'time_delta_to_last_status', 'time_original_media_publishing', 'type']

cols = ['annotator', 'contributing_users',  'media_url',   'project_id', 'report_id', 'report_title',   'tasks_resolved_count', 'type']

quest_ans_cols = ['task_1_question', 'task_1_answer',  'task_2_question', 'task_2_answer', 'task_3_question', 'task_3_answer', 'task_5_question', 'task_5_answer',
                  'task_7_question', 'task_7_answer',  'task_8_question', 'task_8_answer',   'task_9_question','task_9_answer']



complite_cols = cols + question_cols + answer_cols
crowd['annotator'].value_counts()

crowd[complite_cols]

print(list(crowd.columns))
print(question_cols)


import glob

main_df = pd.DataFrame()
for file in glob.glob( data_folder + "*Science*.csv"):
    df = pd.read_csv(file)
    if main_df.empty:
        main_df = df
    else:
        #main_df = main_df.join(df, how='outer')
        main_df = pd.merge(main_df, df, on = 'ID')
        
 
import glob       
science = pd.DataFrame()
i = 0
for file in glob.glob( data_folder + "*Science*.csv"):
    i = i + 1
    df = pd.read_csv(file)
    df['voter'] = 'Science_' + str(i) + '_expert'
    if science.empty:
        science = df
    else:
        #science_df = science_df.join(df, how='outer')
        science = pd.concat([science, df], ignore_index = True)
        
journal = pd.DataFrame()
i = 0
for file in glob.glob( data_folder + "*Journalism*.csv"):
    i = i + 1
    df = pd.read_csv(file)
    df['voter'] = 'Journalism_' + str(i) + '_expert'
    if journal.empty:
        journal = df
    else:
        #journal_df = journal_df.join(df, how='outer', on = 'ID')
        journal = pd.concat([journal, df], ignore_index = True)


science.join(df_crowd, how = 'outer', on = 'URL')

a = pd.merge(science, df_crowd, how = 'right', left_on = 'URL', right_on =  'media_url')
a.isnull().sum()
b = a[a['URL'].isnull()]
len(b['media_url'].unique())


art_50= list(journal_df['URL'].unique())
ec = pd.merge(c, crowd_map,  how = 'inner', left_on = 'annotator', right_on =  'Annotator ID')

len(ec['media_url'].unique())
isd = crowd_all[crowd_all['report_id'] == 32967]
isd['media_url'].unique()
len(crowd_all['report_id'].unique())

agg = crowd_all.groupby('report_id').size()

a = pd.merge(crowd_all[complite_cols], crowd_2020[complite_cols], how = 'inner', left_on = ['media_url', 'annotator'], right_on =  ['media_url','annotator'])

expert_cols = ['ID', 'URL', 'Score', 'voter']
df_science = science[expert_cols]
df_journal = journal[expert_cols]

df_journal['Score'] = journal['Score'].str.split('\)', expand = True)[0]

    

df_trans = df_expert
#alternative_space, 
index_column = 'voter'
column= 'alternative_id'
value = 'Score'

df_trans.pivot(index = index_column, columns = column, values = value)#.reindex_axis(alternative_space, axis=1)


print (vote_data)

print (data_agg.index.name)

print (vote_data.rename_axis(None))
a = data_agg.reset_index()

for col in data_agg:
    print(data_agg[col].dtypes)
    
non_num = []
for col in data_agg:
    if data_agg[col].dtypes not in [ "float64", "int64"]:
         non_num.append(col)
         
         

a = expert_agg[alt_names]>0
a = a.astype(int)
a.sum(axis=1)
a.sum(axis = 0)
ea = expert_agg[alt_names]
ea = ea.astype(float)
ea.mean(axis = 0)

data = crowd.copy()
data['alternative_id'] = data.groupby('media_url').ngroup()
#data['alternative_id']  = data['alternative_id'].astype("str")
#data = data.astype({"alternative_id": str})
# data.dtypes
alternative_map = data[['media_url', 'alternative_id']].drop_duplicates().reset_index().drop('index', axis = 1)
#alternative_map['alternative_id']  = alternative_map['alternative_id'].astype(str)
#alternative_map.dtypes

###### Prepare crow data / select atributes, rename attributes and create label for crowd row
final_col = ['annotator', 'alternative_id', 'task_1_answer']
data = data[final_col]
data['annotator'] =  data['annotator'] + '_crowd'  #data.annotator.str.replace('CredCo-', 'crowd_')


data = data.rename(columns={'annotator':'voter', 'alternative_id':'vote', 'task_1_answer': 'rate'})
data = data[data.rate.notnull()]
data = data.drop_duplicates()

names = list(data['vote'].unique())
names.sort()

return data, names, alternative_map


df = pd.DataFrame({'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
                    'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
                    'C': np.random.randn(8),
                    'D': np.random.randn(8)})
    
df
grouped = df.groupby('A').sum().reset_index()

grouped = df.groupby(['A', 'B'])

grouped.sum()
final_data
animals = pd.DataFrame({'kind': ['cat', 'dog', 'cat', 'dog'], 'height': [9.1, 6.0, 9.5, 34.0], 'weight': [7.9, 7.5, 9.9, 198.0]})
animals
 



a = final_data.groupby("vote").agg(average_rate=pd.NamedAgg(column='rate', aggfunc=np.mean)).reset_index()
b = final_data.groupby(["vote", "expert_type"]).agg(group_rate = pd.NamedAgg(column='rate', aggfunc=np.mean)).reset_index()

final_data = pd.merge(final_data, a, how='left',left_on = ['vote'], right_on = ['vote'])
final_data = pd.merge(final_data, b, how='left',left_on = ['vote', 'expert_type'], right_on = ['vote', 'expert_type'])

animals.groupby("kind").agg(
     min_height=('height', 'min'),
     max_height=('height', 'max'),
     average_weight=('weight', np.mean), )
 
g = final_data.groupby(["expert_type","alternative_id"])

animals.groupby("kind").agg(**{     'total weight': pd.NamedAgg(column='weight', aggfunc=sum),})

df_expert_stats.groupby(["vote", "expert_type"]).agg(std_rate=pd.NamedAgg(column='expert_rate', aggfunc=np.std)).reset_index()
df_expert_stats.dtypes

final_data.groupby("alternative_id").groups
g.describe()
df_expert.groupby("vote").agg({'rate': lambda x: np.std(x, ddof=0)})    

df_expert.groupby("vote").agg({'rate': 'std'})  

df_expert[['vote', 'rate']].groupby("vote").agg(np.std)

final_data['alternative_id'].unique()

df_expert = df_expert.astype({'rate': 'int'})
df_expert.dtypes

final_data.dtypes

np.std([5,3,1,2,1,2])
np.std([5,3,1], ddof = 0)
np.std([2,1,2])


exp_alt = list(df_expert['vote'].unique())
crowd_rest_non_expert = crowd_rest[~crowd_rest['vote'].isin(exp_alt)]
crowd_2020_non_expert = df_crowd_2020[~df_crowd_2020['vote'].isin(exp_alt)]

non_exp_alts = crowd_2020_non_expert['vote'].unique()

crowd_voters_expert = crowd_rest[crowd_rest['vote'].isin(exp_alt)]['voter'].unique()

a =  crowd_rest[crowd_rest['voter'].isin(crowd_voters_expert)]#['voter'].unique()
a[~a['vote'].isin(exp_alt)]



#grades = 

def objective_function_grades(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    return x1*x4*(x1+x2+x3)

def constraint1(x):
    return x[0] * x[1] * x[2] * x[3] - 25.0

def constraint2(x):
    
    sum_sq = 40
    for i in range(4):
        sum_sq = sum_sq - x[i]**2
    return sum_sq

x0 = [1, 5, 5, 1]

objective_function_grades(x0)

b = (1,5)    
bnds = (b,b,b,b)    
con1 = {'type': 'ineq', 'fun':constraint1 } 
con2 = {'type': 'eq', 'fun':constraint2 }    

cons = [con1, con2]

sol = minimize(objective_function_grades, x0, method= 'SLSQP', bounds = bnds, constraints=cons)
print(sol)
