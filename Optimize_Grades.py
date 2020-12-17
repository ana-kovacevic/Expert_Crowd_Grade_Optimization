"""
Created on Tue Nov  3 11:42:27 2020

@author: akovacevic
"""
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize


from scipy.optimize import linprog

import warnings
warnings.filterwarnings('ignore')

#from Data_Prepare import get_aggregated_data


#grades = [1.0, 2.0, 3.0, 4.0, 5.0]

# alpha = 0.5
# v_crowd = [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0 ,3.0]
# v_experts = [1.0, 2.0, 5.0]

# v_crowd = [1, 2, 3, 4, 5, 1, 2 ,3]
# v_experts = [1, 2, 5]

'''##########################################################################
########################     Square differences optimization   ##############
#############################################################################
'''

# x = x0
def objective_function_grades_square(x, v_crowd, v_experts, alpha):
    
    np_crowd = np.array(v_crowd)
    np_expert = np.array(v_experts)
    
    nc = len(v_crowd)
    ne = len(v_experts)
    
    w_crowd = (1 - alpha) / nc
    w_expert = alpha / ne
    
    crd_dist = np.sum(np.power(np_crowd - x, 2))
    exp_dist = np.sum(np.power(np_expert - x, 2))
    
    final = w_crowd * crd_dist + w_expert * exp_dist
    return final



#objective_function_grades_square(x0, v_crowd, v_experts, 0.5)


#def optimize_grade(alt_names, all_exp_grades, all_crowd_grades, alphas ):

def optimize_grade_square_dist(alt_names, all_exp_grades, all_crowd_grades, alphas, x0, bnds ):    
    result_data = pd.DataFrame(columns=['alternative_id', 'alpha', 'optimal_grade', 'fun_val'])
    print("-------------Data for result is crated------------------")
    print(result_data)
    for alt in alt_names:
        print('Alternative for optimization is: ' + str(alt))
        
        result_list = list(np.zeros(result_data.shape[1]))
        print('----Createtd Result list----------')
        print(result_list)
        result_list[0]= alt
        #print('---------Added alternative in result list------------')
        #print(result_list)
        v_experts = all_exp_grades[all_exp_grades.alternative_id == alt]['rate'].astype('float')
        v_crowd = all_crowd_grades[all_crowd_grades.alternative_id == alt]['rate'].astype('float')
    
        
        print('--------  Vectors of grades are created ----------------------')
        #print(v_experts)
        #print(v_crowd)
        
        for alp in alphas:
            print('Alpha for optimization is: ' + str(alp))
            result_list[1] = alp
            
           # print('Alpha added to list')
           # print(result_list)
            sol = minimize(lambda x: objective_function_grades_square(x, v_crowd, v_experts, alp), x0, method= 'SLSQP', bounds = bnds)
            result_list[2] = sol.x[0]
            result_list[3] = sol.fun
            
            print('---Full result list----')
            
            print(result_list)
            result_data = result_data.append(pd.Series(result_list, index=result_data.columns ), ignore_index=True)
    result_data.to_csv('results/square_optimization_grades.csv')
    
    return result_data


######!!!!!!!!!!!!!! Change into: Satisfaction_calculation_square_diff
def calculate_avg_distance_from_optimal(expert_rates, crowd_rates, optimal_grades, alphas): 
    
    res_data = optimal_grades.copy()
    
    n_experts = len(expert_rates['voter_id'].unique())
    n_crowd = len(crowd_rates['voter_id'].unique())
    
    
    expert_data = pd.merge(res_data, expert_rates, on = 'alternative_id')
    expert_data['expert_diff'] = np.power(expert_data['optimal_grade'] - expert_data['rate'], 2)
    
    grouped_expert = expert_data.groupby(["alternative_id", "alpha"]).agg({'expert_diff': np.sum}).reset_index() 
    grouped_expert['expert_diff'] = (grouped_expert['expert_diff'] / n_experts)
    grouped_expert['expert_diff']= np.sqrt(grouped_expert['expert_diff'])
    
    crowd_data = pd.merge(res_data, crowd_rates, on = 'alternative_id')
    crowd_data['crowd_diff'] = np.power(crowd_data['optimal_grade'] - crowd_data['rate'], 2)
    
    grouped_crowd = crowd_data.groupby(["alternative_id", "alpha"]).agg({'crowd_diff' : np.sum}).reset_index()
    grouped_crowd['crowd_diff'] = (grouped_crowd['crowd_diff'] / n_crowd)
    grouped_crowd['crowd_diff'] = np.sqrt(grouped_crowd['crowd_diff'])

    res_data = pd.merge(res_data, grouped_expert, on = ['alternative_id', 'alpha'])
    res_data = pd.merge(res_data, grouped_crowd, on = ['alternative_id', 'alpha'])
    
    return res_data

''' ##############################################################################
    ######################## Abolute distance optimization   #####################
    ##############################################################################
'''
#def get_vote(expert_votes, crowd_votes, lambda_expert, lambda_crowd):
    
# expert_votes = np.array(df_votes[expert_ids]).reshape(len(expert_ids),1)
#expert_votes.shape
#crowd_votes.shape
# crowd_votes =  np.array(df_votes[crowd_ids]).reshape(len(crowd_ids),1)
# lambda_crowd = (1 - lambda_expert)
def objective_function_grades_absolute(expert_votes, crowd_votes, lambda_expert, lambda_crowd):
    # GOAL FUNCTION
    w_expert = np.repeat(lambda_expert/len(expert_votes), len(expert_votes))
    w_crowd = np.repeat(lambda_crowd/len(crowd_votes), len(crowd_votes))
    w_grade = np.array([0])
    
    w = np.concatenate([w_expert, w_crowd, w_grade])
    
    # ABSOLUTE CONSTRAINTS - EXPERT
    lhs_ineq_expert_ind = (len(w_expert), len(w))
    lhs_ineq_expert_1 = np.zeros(lhs_ineq_expert_ind)
    lhs_ineq_expert_2 = np.zeros(lhs_ineq_expert_ind)
    
    rhs_ineq_expert_1 = expert_votes
    rhs_ineq_expert_2 = -expert_votes
    
    for i in range(lhs_ineq_expert_1.shape[0]):
        lhs_ineq_expert_1[i, i] = -1
        lhs_ineq_expert_1[i, len(w) - 1] = 1
        
        lhs_ineq_expert_2[i, i] = -1
        lhs_ineq_expert_2[i, len(w) - 1] = -1
   
    # ABSOLUTE CONSTRAINTS - CROWD
    lhs_ineq_crowd_ind = (len(w_crowd), len(w))
    lhs_ineq_crowd_1 = np.zeros(lhs_ineq_crowd_ind)
    lhs_ineq_crowd_2 = np.zeros(lhs_ineq_crowd_ind)
    
    rhs_ineq_crowd_1 = crowd_votes
    rhs_ineq_crowd_2 = -crowd_votes
    
    for i in range(lhs_ineq_crowd_1.shape[0]):
        lhs_ineq_crowd_1[i, len(expert_votes) + i] = -1
        lhs_ineq_crowd_1[i, len(w) - 1] = 1
        
        lhs_ineq_crowd_2[i, len(expert_votes) + i] = -1
        lhs_ineq_crowd_2[i, len(w) - 1] = -1
        
    # COMPILE
    lhs_ineq = np.concatenate((lhs_ineq_expert_1, lhs_ineq_expert_2, lhs_ineq_crowd_1, lhs_ineq_crowd_2))
    rhs_ineq = np.concatenate((rhs_ineq_expert_1, rhs_ineq_expert_2, rhs_ineq_crowd_1, rhs_ineq_crowd_2))
    
#     bnd = [(1, 5)] * len(w)
    
#     opt = linprog(c=w, A_ub=lhs_ineq, b_ub=rhs_ineq, bounds=bnd, method='interior-point')
    opt = linprog(c=w, A_ub=lhs_ineq, b_ub=rhs_ineq, method='interior-point')
    
    return opt.x[len(w) - 1], opt.fun

def nash_bargaining_solution(expert_satisfaction, crowd_satisfaction):
    lambda_importance = crowd_satisfaction/(crowd_satisfaction + expert_satisfaction)
#     lambda_importance = 1/(1 + expert_satisfaction)
    return lambda_importance

def nash_solution(variables, expert_votes, crowd_votes, max_grade):
    expert_side = np.log(variables[0]) + np.log(max_grade - np.mean(np.abs(expert_votes - variables[2])))
    crowd_side = np.log(variables[1]) + np.log(max_grade - np.mean(np.abs(crowd_votes - variables[2])))
    
    return -1*(expert_side + crowd_side)

def lambda_const(variables):
    return variables[0] + variables[1] - 1


def kalai_smorodinsky_solution(expert_s_1, crowd_s_1, expert_s_2, crowd_s_2):
    max_expert = np.max([expert_s_1, expert_s_2])
    max_crowd = np.max([crowd_s_1, crowd_s_2])
    
    if np.abs(expert_s_1 - expert_s_2) < 0.0001 and np.abs(crowd_s_1 - crowd_s_2) < 0.0001:
        return (0.5)
    
    lambda_importance = (max_expert * crowd_s_2 - max_crowd * expert_s_2)/(max_crowd * (expert_s_1 - expert_s_2) - max_expert * (crowd_s_1 - crowd_s_2))
    
    return lambda_importance
                                 
                                      
                                     
def expectation_maximization_nash(optimal_grades, lambda_expert, df_votes, crowd_ids, expert_ids, max_grade, num_iter = 100, verbose = False):
    #lambda_expert = 0.5
    to_continue = True
    iterator = 1
    
    while to_continue:
        # FIND VOTE
        if lambda_expert == 0.5:
            vote = optimal_grades['optimal_grade'].item()
        else:
            vote = objective_function_grades_absolute(np.array(df_votes[expert_ids]).reshape(len(expert_ids),1), 
                                                  np.array(df_votes[crowd_ids]).reshape(len(crowd_ids),1), 
                                                  lambda_expert = lambda_expert, 
                                                  lambda_crowd = (1 - lambda_expert))[0]
      
        # FIND LAMBDA
        expert_satisfaction = np.mean(max_grade - np.abs(df_votes[expert_ids] - vote), axis = 1).item() #optimal_grades['expert_sat']
        crowd_satisfaction = np.mean(max_grade - np.abs(df_votes[crowd_ids] - vote), axis = 1).item() #optimal_grades['crowd_sat']
    
        new_lambda = nash_bargaining_solution(expert_satisfaction, crowd_satisfaction)
        
        if verbose:
            print(new_lambda)
        
        # STOP CRITERIA
        if lambda_expert == new_lambda:
            to_continue = False
        else:
            lambda_expert = new_lambda
            
        iterator = iterator + 1
        if iterator == num_iter:
            to_continue = False
            
    area = expert_satisfaction * crowd_satisfaction
    
    return lambda_expert, vote, expert_satisfaction, crowd_satisfaction, area

# optimal_grades= res
# df_votes = votes


def maximization_kalai_smorodinsky(optimal_grades, max_grade, df_votes, crowd_ids, expert_ids ): #, expert_votes, crowd_votes
    # FIND VOTE
    #vote_1 = objective_function_grades_absolute(expert_votes, crowd_votes, lambda_expert = 1, lambda_crowd = 0)[0]
    #vote_1 = optimal_grades[optimal_grades['alpha'] == 1]['optimal_grade'].item()
    #vote_2 = objective_function_grades_absolute(expert_votes, crowd_votes, lambda_expert = 0, lambda_crowd = 1)[0]
    #vote_2 = optimal_grades[optimal_grades['alpha'] == 0]['optimal_grade'].item()
    # FIND LAMBDA
    expert_s_1 = optimal_grades[optimal_grades['alpha'] == 1]['expert_sat'].item() # np.mean(5 - np.abs(expert_votes - vote_1))
    expert_s_2 = optimal_grades[optimal_grades['alpha'] == 0]['expert_sat'].item() #np.mean(5 - np.abs(expert_votes - vote_2))
    crowd_s_1 = optimal_grades[optimal_grades['alpha'] == 1]['crowd_sat'].item() #np.mean(5 - np.abs(crowd_votes - vote_1))
    crowd_s_2 = optimal_grades[optimal_grades['alpha'] == 0]['crowd_sat'].item() #np.mean(5 - np.abs(crowd_votes - vote_2))
    
    lambda_expert = kalai_smorodinsky_solution(expert_s_1, crowd_s_1, expert_s_2, crowd_s_2)
    
    vote = objective_function_grades_absolute(np.array(df_votes[expert_ids]).reshape(len(expert_ids),1), 
                                              np.array(df_votes[crowd_ids]).reshape(len(crowd_ids),1),  
                                              lambda_expert, 1 - lambda_expert)[0]
    
    expert_satisfaction = np.mean(max_grade - np.abs(df_votes[expert_ids] - vote), axis = 1).item() 
    crowd_satisfaction = np.mean(max_grade - np.abs(df_votes[crowd_ids] - vote), axis = 1).item()
    
    
    area = expert_satisfaction * crowd_satisfaction
    
    return lambda_expert, vote, expert_satisfaction, crowd_satisfaction, area


def optimize_grade_absolute_dist(df_alt_votes, expert_ids, crowd_ids, alphas):
    
    result_data = pd.DataFrame(columns=['alternative_id', 'alpha', 'optimal_grade', 'fun_val'])

    #data = data.iloc[0:5, :]
    num_alt = len(df_alt_votes['alternative_id'].unique())
    start_idex = 0

    for alp in alphas:
        end_index = start_idex + num_alt

        result_data = pd.concat([result_data, pd.DataFrame(df_alt_votes['alternative_id'])],ignore_index=True)
        #df.apply(lambda x: func(x['col1'],x['col2']),axis=1)
        res_list = list(
           df_alt_votes.apply(lambda x: objective_function_grades_absolute(
                                     x[expert_ids], x[crowd_ids], alp, 1 - alp), axis = 1) )
       
        result_data.iloc[start_idex:end_index, 1] = alp   
       
        opt = pd.DataFrame( res_list, columns=(['optimal_grade','fun_val']))
       
        result_data.iloc[start_idex:end_index, 2] = list(opt['optimal_grade'])
        result_data.iloc[start_idex:end_index, 3] = list(opt['fun_val'])
       
        start_idex = start_idex + num_alt 
    
    return result_data
       
       

# optimal_grades = result_optimization_abs

def calculate_satisfaction_absolute(df_alt_votes, optimal_grades, max_grade, expert_ids, crowd_ids): 
    
    res_data = optimal_grades.copy()
    
    #n_experts = len(expert_ids)
    #n_crowd = len(crowd_ids)
    
    data = pd.merge(res_data, df_alt_votes, on = 'alternative_id')
    
    data['expert_sat'] = np.mean(
        max_grade - np.abs(np.array(data[expert_ids]) - np.array(data['optimal_grade']).reshape(data.shape[0],1) )
        , axis = 1)
    data['crowd_sat'] = np.mean(
        max_grade - np.abs(np.array(data[crowd_ids]) - np.array(data['optimal_grade']).reshape(data.shape[0], 1) )
        , axis = 1)

    if 'alpha' in list(data.columns):
        res_data = data[['alternative_id', 'alpha', 'optimal_grade', 'expert_sat', 'crowd_sat']]
    else:
        res_data = data[['alternative_id', 'lambda_exp', 'optimal_grade', 'expert_sat', 'crowd_sat']]
    #data[['alternative_id', 'alpha', 'optimal_grade', 'fun_val', 'expert_sat', 'crowd_sat']]
    
    return res_data



'''
def optimize_grade_absolute_dist(alt_names, all_exp_grades, all_crowd_grades , alphas ):    
    result_data = pd.DataFrame(columns=['alternative_id', 'alpha', 'optimal_grade', 'fun_val'])
    print("-------------Data for result is crated------------------")
    print(result_data)
    for alt in alt_names:
        print('Alternative for optimization is: ' + str(alt))
        
        result_list = list(np.zeros(result_data.shape[1]))
        print('----Createtd Result list----------')
        print(result_list)
        result_list[0]= alt
        #print('---------Added alternative in result list------------')
        #print(result_list)
        v_experts = np.array(all_exp_grades[all_exp_grades.alternative_id == alt]['rate'].astype('float'))
        v_crowd = np.array(all_crowd_grades[all_crowd_grades.alternative_id == alt]['rate'].astype('float'))
    
        print('--------  Vectors of grades are created ----------------------')
        #print(v_experts)
        #print(v_crowd)
        #alp = 0.5
        for alp in alphas:
            print('Alpha for optimization is: ' + str(alp))
            result_list[1] = alp
            
           # print('Alpha added to list')
           # print(result_list)
            sol = objective_function_grades_absolute(v_experts, v_crowd, alp, 1 - alp)
            result_list[2] = sol[0]
            result_list[3] = sol[1]
            
            print('---Full result list----')
            
            print(result_list)
            result_data = result_data.append(pd.Series(result_list, index=result_data.columns ), ignore_index=True)
    result_data.to_csv('results/absolute_optimization_grades.csv')
    
    return result_data

def calculate_rate_differences_from_optimal(v_rates, x_optimal):
    
    
    np_rates = np.array(v_rates)
    
    n = len(v_rates)
    
    
    crd_dist = np.sum(np.power(np_rates - x_optimal, 2)) * (1/n)

    final = np.sqrt(crd_dist)
    return final

#optimize_grade(alt_names, all_exp_grades, all_crowd_grades, alphas )
crowd_opt_df = pd.DataFrame(columns=['alternative_id', 'alpha', 'crowd_diff'])
al = res_crowd['alternative_id'].unique()
#a = al[0]
for a in al:
    print(a)
    result_list_crowd = list(np.zeros(crowd_opt_df.shape[1]))
    result_list_crowd[0] = a
    crowd_opt = res_crowd[res_crowd['alternative_id'] ==  a]
    print(crowd_opt)
    for alp in alphas:
        #print(alp)
        result_list_crowd[1] = alp
        crowd_alp = crowd_opt[crowd_opt['alpha'] == alp]
        e_rates = crowd_alp['rate']
        x_optimal = crowd_alp['optimal_grade'].unique()
        crowd_diff = calculate_rate_differences_from_optimal(e_rates, x_optimal)
        result_list_crowd[2] =  crowd_diff
        
        crowd_opt_df = crowd_opt_df.append(pd.Series(result_list_crowd, index=crowd_opt_df.columns ), ignore_index=True)

result_data = pd.merge(result_data, expert_opt_df, left_on=['alternative_id', 'alpha'], right_on=['alternative_id', 'alpha'])
result_data = pd.merge(result_data, crowd_opt_df, left_on=['alternative_id', 'alpha'], right_on=['alternative_id', 'alpha'])
'''
#result_data.to_csv('results/optimal_grads_diff_square.csv')

