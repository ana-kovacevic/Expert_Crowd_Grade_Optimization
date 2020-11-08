# -*- coding: utf-8 -*-
"""
Created on Sun May  3 11:31:46 2020

@author: akovacevic
"""

from sklearn.ensemble import IsolationForest
import numpy as np

#np.random.seed(1)
#random_data = np.random.randn(50000,2)  * 20 + 20

def detect_outliers(data, max_samples, random_state = 1):
    clf = IsolationForest( max_samples = max_samples, random_state = random_state)
    preds = clf.fit_predict(data)
    return preds

def remove_outliers(data, preds, outlier_value):
    
    out  = preds != outlier_value 
    data_without_outliers = data[out]
    
    return data_without_outliers


