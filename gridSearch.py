import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from datetime import datetime

from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

airsites=['GR8']
for site in airsites:
    print('Processing Site: '+site)
    inputValues = pd.read_csv('prepareddata/inputs-'+site+'.csv')
    results = pd.read_csv('prepareddata/results-'+site+'.csv')
    
    #only consider complete records,disgard incomplete records
    inputValues=inputValues[inputValues['Complete']==True]
    results = results[results['Complete']==True]
    
    ### experiement after feature selection
    
    inputValues=inputValues.reset_index(drop=True)
    results=results.reset_index(drop=True)
    
    #get rid of Incomplete and 'Date' columns
    inputValues=inputValues.iloc[:,3:]
    results=results.iloc[:,3:].values
    
    #generate training and testing data
    x_train,x_test,y_train,y_test=train_test_split(inputValues,results,test_size=0.2,random_state=42)
        
    # grid_search for parameters
    paramgrid = {'max_depth': list(range(15, 20, 1)), 'min_samples_split':list(range(100, 301, 100)),'min_samples_leaf':[50,60,70]}
    grid_search = GridSearchCV(estimator = GradientBoostingRegressor(n_estimators=2000,learning_rate=0.0875,
                               subsample=1, max_features='sqrt',random_state=42),
                               param_grid = paramgrid,cv=5)
                               
    # Fit the grid search model
    grid_search.fit(x_train,y_train)
    
    # Estimating the optimized value
    print(grid_search.best_estimator_)
    print(grid_search.best_params_)
    print(grid_search.best_score_)
