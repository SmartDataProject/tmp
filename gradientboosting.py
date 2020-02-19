import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import math
from datetime import datetime

from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def generateSummary(inputs, y_actual,y_pred):
    summary=inputs.copy()
    summary['NO2Actual']=y_actual
    summary['NO2Predict']=y_pred
    return summary


airsites=['GR8','GR4','GR5','GR7','GN0', 'GR9','GN2','GN3','GN4']
    
for site in airsites:
    print('/n' )
    print(datetime.now())
    print('Processing Site: '+site)
    inputValues = pd.read_csv('prepareddata/inputs-'+site+'.csv')
    results = pd.read_csv('prepareddata/results-'+site+'.csv')
    
    #only consider complete records,disgard incomplete records
    inputValues=inputValues[inputValues['Complete']==True]
    results = results[results['Complete']==True]
    inputValues=inputValues.reset_index(drop=True)
    results=results.reset_index(drop=True)
    
    #get rid of Incomplete and 'Date' columns
    inputValues=inputValues.iloc[:,3:]
    results=results.iloc[:,3:]

    #generate training and testing data
    x_train,x_test,y_train,y_test=train_test_split(inputValues,results,test_size=0.2,random_state=42)
    
    """
    #use Cross Validation (KFold) on training set
    #index needs to be reset otherwise will result in NA values after the train_test_split
    x_train=x_train.reset_index(drop=True)
    y_train=y_train.reset_index(drop=True)
    #print(x_train.head())
    
    i=1
    kf = KFold(n_splits=2)
    for train_index,cv_index in kf.split(x_train):
        print('\n{} of kfold {}'.format(i,kf.n_splits))
        xtr,xvl = x_train.loc[train_index],x_train.loc[cv_index]
        ytr,yvl = y_train.loc[train_index],y_train.loc[cv_index]
        print(xtr.shape)
        print(xvl.shape)
        i=i+1
        
        regr = GradientBoostingRegressor(
                               n_estimators=2000,
                               max_features='auto',
                                max_depth=5,
                                loss='ls',
                                verbose=True,
                                criterion='mse',
                                learning_rate= 0.0875,
                               random_state=42)
        
        regr.fit(xtr, ytr)
        pred_cv = regr.predict(xvl)
        score = r2_score(yvl,pred_cv)
        print('R-squared of training data: ',score)
    """
    
    regr = GradientBoostingRegressor(
                               n_estimators=2000,
                               max_features='auto',
                                max_depth=5,
                                loss='ls',
                                verbose=True,
                                criterion='friedman_mse',
                                learning_rate= 0.0875,
                               random_state=42)
    regr.fit(x_train, y_train)
    y_pred = regr.predict(x_train)

    y_mean=y_train.mean()
    mse = math.sqrt(mean_squared_error(y_train.values, y_pred))

    print('Mean squared error ratio of training data: {}'.format(mse/y_mean))
    print('R-squared of training data: {}'.format(r2_score(y_train, y_pred)))
        
    #on test data
    y_test_pred = regr.predict(x_test)
    factors=pd.DataFrame(columns=['Attribute', 'Importance'])
    for i in range(0, len(x_test.columns)):
        factors.loc[i, 'Attribute']=x_test.columns[i]
        factors.loc[i, 'Importance']=regr.feature_importances_[i]
    factors.to_csv('results/factors-gradientboosting-'+site+'.csv')
        
    y_test_mean=np.mean(y_test_pred)
    mse_test=mean_squared_error(y_test.values, y_test_pred)

    print('Mean of test data: {}'.format(y_test_mean))
    print('MSE of test data: {}'.format(mse_test))
    print('Mean squared error ratio of test data: {}'.format(math.sqrt(mse_test)/y_test_mean))
    print('R-squared of test data: {}'.format(r2_score(y_test, y_test_pred)))   
    
    #saving results
    summary=generateSummary(x_test, y_test['NO2Level'].values, y_test_pred)
    summary.to_csv('results/gradientboosting-'+site+'.csv')
    