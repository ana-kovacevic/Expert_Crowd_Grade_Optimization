# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:59:13 2020

@author: akovacevic
"""
import numpy as np
#import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from Matrix_Factorization import ExplicitMF

#MF_ALS = ExplicitMF(train, n_factors=5, user_reg=0.0, item_reg=0.0)
#iter_array = [1, 2, 5, 10, 25, 50, 100]
#
#
#MF_ALS.calculate_learning_curve(iter_array, test)


#latent_factors = [2, 3, 5, 7, 10, 15, 20]
#regularizations = [0.01, 0.1, 1., 10., 100.]
#regularizations.sort()
#iter_array = [1, 2, 5, 10, 25, 50, 100]

def find_best_parms_for_ALS(latent_factors, regularizations, iter_array, train, test):
        
    best_params = {}
    best_params['n_factors'] = latent_factors[0]
    best_params['reg'] = regularizations[0]
    best_params['n_iter'] = 0
    best_params['train_mse'] = np.inf
    best_params['test_mse'] = np.inf
    best_params['model'] = None

    for fact in latent_factors:
        #print(( 'Factors: {}').format(fact))
        for reg in regularizations:
            #print(( 'Regularization: {}').format(reg))
            MF_ALS = ExplicitMF(train, n_factors=fact, user_reg=reg, item_reg=reg)
            MF_ALS.calculate_learning_curve(iter_array, test)
            min_idx = np.argmin(MF_ALS.test_mse)
            if MF_ALS.test_mse[min_idx] < best_params['test_mse']:
                best_params['n_factors'] = fact
                best_params['reg'] = reg
                best_params['n_iter'] = iter_array[min_idx]
                best_params['train_mse'] = MF_ALS.train_mse[min_idx]
                best_params['test_mse'] = MF_ALS.test_mse[min_idx]
                best_params['model'] = MF_ALS
                print ('New optimal hyperparameters for ALS')
                #print( pd.Series(best_params))

    #best_als_model = best_params['model']
    return best_params


sns.set()

def plot_learning_curve(iter_array, model):
    plt.plot(iter_array, model.train_mse, \
             label='Training', linewidth=5)
    plt.plot(iter_array, model.test_mse, \
             label='Test', linewidth=5)

    plt.xticks(fontsize=16);
    plt.yticks(fontsize=16);
    plt.xlabel('iterations', fontsize=30);
    plt.ylabel('MAE', fontsize=30);
    plt.legend(loc='best', fontsize=20);

#plot_learning_curve(iter_array, best_als_model)










