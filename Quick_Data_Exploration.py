# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 21:57:45 2020

@author: akovacevic
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
#sns.set_theme(style="ticks", color_codes=True)
sns.set_theme()

from Load_credibility_data import read_data_credibility

from Data_Prepare import crete_voter_map

from PrepareData_SupervisedApproach import aggregate_statistics_per_alternatives


'''
    Read Data
'''
alternative_map, alt_names, df_crowd, df_expert, df_crowd_2020 = read_data_credibility()
voter_map = crete_voter_map([df_expert, df_crowd])


##### check if crowd have changed their opinion on last question
def compare_opinion_consistency(data):
    comparison_column = np.where(data["task_1_answer"] != data["task_9_answer"], True, False)
    comparison_column.sum()
    comparison_column.sum()/len(comparison_column)*100

    comparison = data[comparison_column]
    len(comparison["annotator"].unique())
    len(data["annotator"].unique())
    b = comparison[["annotator","media_url"]].groupby("annotator").count()
    
    
    print(b)
    print(comparison.groupby('annotator').count()[["annotator"]])


###### Plot


def plot_overall_variability_bar(df_expert, df_crowd, voter_map): 
#### exert - alternative ratings (on expert level aggregated to one expert or aggregated to group expert)
    df_expert_stats = aggregate_statistics_per_alternatives(df_expert, voter_map)
    crowd_stats = aggregate_statistics_per_alternatives(df_crowd, voter_map)


    data = df_expert_stats[['expert_type', 'vote', 'expert_rate', 'std_group_rate']]
    data = data[['expert_type', 'vote', 'std_group_rate']].drop_duplicates()

    cd = crowd_stats[['expert_type', 'vote', 'std_group_rate']].drop_duplicates()

    data = pd.concat([data, cd])
    #data = pd.merge(data, cd, how = 'inner', on = 'vote')

    plt.figure(figsize=(30,10))
    std_rate = cd['std_group_rate']
    bars = (cd['vote'])
    y_pos = np.arange(len(bars))
 
# Create bars
    plt.bar(y_pos, std_rate)
 
# Create names on the x-axis
    plt.xticks(y_pos, bars)
 
# Show graphic
    plt.show()

plot_overall_variability_bar(df_expert, df_crowd_2020, voter_map)

def plot_expert_group_variability_bar(df_expert, df_crowd, voter_map):
    
    df_expert_stats = aggregate_statistics_per_alternatives(df_expert, voter_map)
    #crowd_stats = aggregate_statistics_per_alternatives(df_crowd, voter_map)


    data = df_expert_stats[['voter_type', 'vote', 'expert_rate', 'std_group_rate']]
    data = data[['expert_type', 'vote', 'std_group_rate']].drop_duplicates()
    # data to plot
    data = data.sort_values(by = 'vote')
    means_sci = (data[data.expert_type == 'Science']['std_group_rate'])
    means_jor = (data[data.expert_type == 'Journalism']['std_group_rate'])
    #means_crw = (data[data.expert_type == 'CredCo']['std_group_rate'])
    n_sci = len(means_sci)
    n_jor = len(means_jor)
    #n_crw = len(means_crw)
    # create plot
    #plt.figure(figsize=(30,10))
    fig, ax = plt.subplots(figsize=(35,10))
    index1 = np.arange(n_sci)
    index2 = np.arange(n_jor)
    #index3 = np.arange(n_crw)
    bar_width = 0.1
    opacity = 0.9
    
    rects1 = plt.bar(index1, means_sci, bar_width,
    alpha=opacity,
    color='b',
    label='Science')
    # rects1 =  plt.plot(index1, means_sci, 'o', color='b', label='Science')
    #rects2 = plt.plot(index2, means_jor, 'o', color='g', label='Journal')
    rects2 = plt.bar(index2 + bar_width, means_jor, bar_width,
    alpha=opacity,
    color='g',
    label='Journal')
    
    # #rects3 = plt.plot(index3, means_crw, 'o', color='g', label='Crowd')
    # rects3 = plt.bar(index3 + bar_width, means_crw, bar_width,
    # alpha=opacity,
    # color='r',
    # label='Crowd')
    
    plt.xlabel('Alternative')
    plt.ylabel('Standard deviation')
    plt.title('Standard deviation by Groups')
    plt.xticks(index1 + bar_width, (index1))
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_expert_group_variability_bar(df_expert, df_crowd_2020)

#df_crowd = df_crowd_2020
def plot_variability_scatter(df_expert, df_crowd, voter_map, group_by = 'voter_type', measure = 'std_group_rate', split_char = '-'):
    # plot_variability_scatter(df_expert, df_crowd, voter_map, group_by = 'voter', measure = 'std_rate', split_char = '_')
    # plot_variability_scatter(df_expert, df_crowd, voter_map, group_by = 'voter_type', measure = 'std_group_rate', split_char = '-')
    df_expert_stats = aggregate_statistics_per_alternatives(df_expert, voter_map)
    crowd_stats = aggregate_statistics_per_alternatives(df_crowd, voter_map)
    
    data = df_expert_stats[[group_by, 'vote', 'expert_rate', measure]]
    data = data[[group_by, 'vote', measure]].drop_duplicates()
    
    cd = crowd_stats[[group_by, 'vote', measure]].drop_duplicates()
    
    
    data = pd.concat([data, cd])
    
    if split_char == '-':
        data['voter_type'] = data[group_by].str.split(split_char, expand = True)[0]
    else:
        data['voter_type'] = data[group_by].str.split(split_char, expand = True)[1]
    
    data = data.drop('voter', axis = 1, errors = 'ignore')
    data = data.drop_duplicates()
    num = len(data['voter_type'].unique())
    

    a = data.groupby('vote').count()
    votes = a[a['voter_type'] == num].index
    votes = list(votes)


    data = data[data['vote'].isin(votes)]
    data = data.sort_values(by = 'vote')
    
    groups = list(data['voter_type'].unique())
    shapes = ['o', 'v', '*']
    colors = ['g', 'b', 'r']

    x_data = range(0, len(a[a['voter_type'] == num].index)) #range(0, a.shape[0]) #
    

    #zip(groups, shapes[0:num_groups], colors[0:num_groups])

    fig, ax = plt.subplots()
    # plot each column
    for column in zip(groups, shapes[0:num], colors[0:num]):#[('Science', 'o', 'g'), ('Journalism', 'v', 'b'), ('CredCo', '*' ,'r')]:
        y_data = data[data['voter_type'] == column[0] ][measure]
        #ax.plot(x_data, y_data, label = column)
        plt.plot(x_data, y_data, column[1], color=column[2], label=column[0])
    # set title and legend
    ax.set_title('Variability within group')
    ax.legend()


data = result_data[result_data['alpha']== 0.5]

    #data = pd.merge(data, cd, how = 'inner', on = 'vote')
data.dtypes
data.shape

data["alternative_id"] = data["alternative_id"].astype('category')

data = data.sort_values(by=['fun_val'] , ascending=False)

plt.figure(figsize=(40,12))
sns.catplot(x="alternative_id", y="fun_val", kind="bar", color = 'b', data=data)
#sns.barplot(x = "alternative_id", y = "fun_val",order= data['alternative_id'], color = 'b' , data= data)
plt.show()



plt.figure(figsize=(40,10))
std_rate = data['fun_val']
bars = (data['alternative_id'])
y_pos = np.arange(len(bars))
 
# Create bars
plt.bar(y_pos, std_rate, color = 'b')
 
# Create names on the x-axis
plt.xticks(y_pos, bars)
 
# Show graphic
plt.show()
