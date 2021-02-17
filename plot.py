# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:19:02 2021

@author: akovacevic
"""
import pickle
import pandas as pd
# Load from file
pkl_filename = 'CV_results/cv_results_RandomForest.pkl'
with open(pkl_filename, 'rb') as file:
    pickle_file = pickle.load(file)
    
rand_forest = pd.DataFrame.from_dict(pickle_file)
    