# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 12:37:17 2020

@author: akovacevic
"""
import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt 
# import seaborn as sns


# import sklearn
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler

# from sklearn.preprocessing import scale
# from sklearn.feature_selection import RFE

# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
#from sklearn.metrics import neg_root_mean_squared_error



import pickle
#from sklearn.metrics import  

#sklearn.metrics.SCORERS.keys()

# filter only area and price
# df = final_train_data #.drop(['voter_id_map', 'alternative_id'], axis=1)
# df.head()

def optimize_predictive_model_and_predict(df_tr, df_ts, folds = 3):

    param_grid_lr = { }    
    
    param_grid_rf = {
        'n_estimators' : [1, 3, 5, 7, 10, 50, 100, 200, 500]
        ,'max_depth':[5, 10, 15]
        #, 'min_samples_split': [5, 50, 100]
        } 
    
    param_grid_gbr = {
        'n_estimators' : [1, 3, 5, 7, 10, 50, 100, 200, 500]
        ,'max_depth':[1, 3, 5, 7, 10, 15, 30]
        ,'learning_rate':[0.01, 0.1, 0.05, 0.25, 0.5, 1]
        #, 'min_samples_split': [5, 50, 100]
        } 
    
    # param_grid_dtr = {
    #     'max_depth' : [2, 4, 6]
    #     # ,'min_samples_split' : [5, 50, 100]
    #     # ,'min_samples_leaves': [15, 30, 50],
    #     ,'max_features': [ 10, 50, 300]
    #  }    
       
    lm = LinearRegression()
    rf = RandomForestRegressor()
    gbr =  GradientBoostingRegressor()
    
    folds=folds;    
    
    gs_LR_wc = GridSearchCV(estimator = lm, param_grid = param_grid_lr, 
                            n_jobs = 1,cv= folds, scoring = 'neg_root_mean_squared_error', verbose = 1, return_train_score=False)
    gs_RF_wc = GridSearchCV(estimator = rf, param_grid = param_grid_rf, 
                            n_jobs = 1,cv= folds, scoring = 'neg_root_mean_squared_error', verbose = 1, return_train_score=False)
    gs_GBM_wc = GridSearchCV(estimator= gbr, param_grid = param_grid_gbr, 
                             n_jobs = 1,cv=folds, scoring = 'neg_root_mean_squared_error', verbose = 1, return_train_score=False)
#                             n_jobs = 1,cv=folds, scoring = my_scorer.prc_score, return_train_score=False)
# split into train and test
    X = df_tr.loc[:, df_tr.columns != 'rate']
    y = df_tr['rate']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 10)


    grids = [gs_LR_wc ,gs_RF_wc, gs_GBM_wc]  #,  gs_GBM_wc, gs_GBM_im, gs_RF_im, gs_LR_im ]
    grid_dict = {0:'LinearRegression', 1: 'RandomForest', 2: 'GradientBoostingRegressor'} # , 3: 'GBMWithImb', 4:'RandomForestWithImb', 5: 'LogisticWithImb'}
    

    best_mse = 100
    best_mod = 0
    best_gs = ''

    for idx, gs in enumerate(grids):
        print('\nEstimator: %s' % grid_dict[idx])		
        print(gs)
        model = gs.fit(X_train, y_train)
        print('Best params: %s' % model.best_params_)
        print('Best training neg_root_mean_squared_error: %.3f' % model.best_score_)
        #model.best_estimator_
        y_pred = model.predict(X_test)
        #y_probas = model.predict_proba(X_test)
        #y_probas = y_probas[:,1]
        print('Test set sq root mse score for best params: %.3f ' % np.sqrt(mean_squared_error(y_test, y_pred)))
    	   # Track best (highest test accuracy) model
        mn = grid_dict[idx]
        model_name = mn  + '.sav'
        model_filename = 'Models/' + model_name
        
        pickle._dump(model.best_estimator_, open(model_filename, 'wb'))
        dict_name = 'CV_results/cv_results_' + mn  + '.pkl' 
        pickle._dump(gs.cv_results_, open(dict_name, "wb"))
        
        if mean_squared_error(y_test, y_pred) < best_mse:
            best_mse = mean_squared_error(y_test, y_pred)
            best_gs = gs
            best_mod = idx
    print('\nModel with best test set metrics: %s' % grid_dict[best_mod])


    all_pred = grids[best_mod].predict(df_ts)
    
    return all_pred, grids, best_mod

# mod = 'RandomForest.sav'            

# loaded_model = pickle.load(open( 'Models/' + mod, 'rb')) 
# #loaded_model.predict(non_rated_combinations)
# pred = loaded_model.predict(non_rated_combinations)
    
# prediction_dict.update({mod:pred})




'''
gs_LR_wc.fit(X_train, y_train)
gs_RF_wc.fit(X_train, y_train)

cv_results_LR = pd.DataFrame(gs_LR_wc.cv_results_)
cv_results_LR


cv_results_RF = pd.DataFrame(gs_RF_wc.cv_results_)
cv_results_RF

lr_pred = gs_LR_wc.predict(X_test)
rf_pred = gs_RF_wc.predict(X_test)

mse_lr = sklearn.metrics.mean_squared_error(y_test, lr_pred)
print(mse_lr)

mse_rf = sklearn.metrics.mean_squared_error(y_test, rf_pred)
print(mse_rf)
'''

'''

"""
    Hyperparameter Tuning Using Grid Search Cross-Validation - Linear Regression
"""
# step-1: create a cross-validation scheme
folds = KFold(n_splits = 5, shuffle = True, random_state = 100)

# step-2: specify range of hyperparameters to tune
hyper_params = [{'n_features_to_select': list(range(1, 42))}]


# step-3: perform grid search
# 3.1 specify model
lm = LinearRegression()
lm.fit(X_train, y_train)
rfe = RFE(lm)             

# 3.2 call GridSearchCV()
model_cv = GridSearchCV(estimator = rfe, 
                        param_grid = hyper_params, 
                        scoring= 'r2', 
                        cv = folds, 
                        verbose = 1,
                        return_train_score=True)      

# fit the model
model_cv.fit(X_train, y_train)

# cv results
cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results

# plotting cv results
plt.figure(figsize=(16,6))

plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_test_score"])
plt.plot(cv_results["param_n_features_to_select"], cv_results["mean_train_score"])
plt.xlabel('number of features')
plt.ylabel('r-squared')
plt.title("Optimal Number of Features")
plt.legend(['test score', 'train score'], loc='upper left')

"""
    Linear Regression
"""

from sklearn.model_selection import train_test_split
X = df.loc[:, df.columns != 'abs_diff_one']
y = df['abs_diff_one']
   
    ###### train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33)



from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

print(regr.coef_)  
 # The mean square error
np.mean((regr.predict(X_test) - y_test)**2)

# Explained variance score: 1 is perfect prediction
# and 0 means that there is no linear relationship
# between X and y.
regr.score(X_test, y_test)



# recaling the variables (both)
df_columns = df.columns
#scaler = MinMaxScaler()
#df = scaler.fit_transform(df)

# rename columns (since now its an np array)
#df = pd.DataFrame(df)
#df.columns = df_columns

#df.head()
len(df.columns)
# visualise area-price relationship
for col in df.columns[0:41]:
   sns.regplot(x=col, y="rate", data=df, fit_reg=False)
   plt.show()
   
   
# split into train and test
X = df.loc[:, df.columns != 'rate']
y = df['rate']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 10)


# first model with an arbitrary choice of n_features
# running RFE with number of features=10

lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, n_features_to_select=10)             
rfe = rfe.fit(X_train, y_train)

# tuples of (feature name, whether selected, ranking)
# note that the 'rank' is > 1 for non-selected features
list(zip(X_train.columns,rfe.support_,rfe.ranking_))

# predict prices of X_test
y_pred = rfe.predict(X_test)

# evaluate the model on test set
r2 = sklearn.metrics.r2_score(y_test, y_pred)
print(r2)


# try with another value of RFE
lm = LinearRegression()
lm.fit(X_train, y_train)

rfe = RFE(lm, n_features_to_select=6)             
rfe = rfe.fit(X_train, y_train)

# predict prices of X_test
y_pred = rfe.predict(X_test)
r2 = sklearn.metrics.r2_score(y_test, y_pred)
print(r2)

# k-fold CV (using all the 13 variables)
lm = LinearRegression()
scores = cross_val_score(lm, X_train, y_train, scoring='r2', cv=5)
scores 

# the other way of doing the same thing (more explicit)

# create a KFold object with 5 splits 
folds = KFold(n_splits = 3, shuffle = True, random_state = 100)
scores = cross_val_score(lm, X_train, y_train, scoring='r2', cv=folds)
scores   

# can tune other metrics, such as MSE
scores = cross_val_score(lm, X_train, y_train, scoring='neg_mean_squared_error', cv=3)
scores
'''
