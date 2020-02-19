import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
import math
from datetime import datetime

from sklearn import tree
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
import xgboost

#airsites=['GR8','GR9','GR7','GR5','GR4','GN0','GN2','GN3','GN4']
airsites=['GR8']
for site in airsites:
    print('Processing Site: '+site)
    inputValues = pd.read_csv('prepareddata/inputs-'+site+'.csv')
    results = pd.read_csv('prepareddata/results-'+site+'.csv')
    
    #only consider complete records,disgard incomplete records
    inputValues=inputValues[inputValues['Complete']==True]
    results = results[results['Complete']==True]

    #get rid of Incomplete and 'Date' columns
    inputValues=inputValues.iloc[:,3:]
    results=results.iloc[:,3:]
    
    #generate training and testing data
    x_train,x_test,y_train,y_test=train_test_split(inputValues,results,test_size=0.2,random_state=42)
    
    ### screening various algorithms
    #linear regression
    lnenabled = False
    if lnenabled:
        print('Running Liner Regression:')
        regr = LinearRegression()
        regr.fit(x_train, y_train)
        #y_pred = regr.predict(x_train)
        #print('Coefficients: {}'.format(regr.coef_))
        #y_mean=y_train.mean()
        #mse = math.sqrt(mean_squared_error(y_train.values, y_pred))
        #print(y_mean)
        #print(mse)
        print('Mean squared error ratio: {}'.format(mse/y_mean))
        print('R-squared: {}'.format(r2_score(y_train, y_pred)))
        
        #on test data
        y_test_pred = regr.predict(x_test)
        print('Coefficients: {}'.format(regr.coef_))
        y_test_mean=np.mean(y_test.values)
        mse_test=mean_squared_error(y_test.values, y_test_pred)
        #mse_test = np.mean((y_test - y_test_pred)**2)
        print(y_test_mean)
        print(mse_test)
        print('Mean squared error ratio: {}'.format(math.sqrt(mse_test)/y_test_mean))
        print('R-squared: {}'.format(r2_score(y_test, y_test_pred)))
        
        #statistic testing
        import statsmodels.api as sm
        x_train2 = sm.add_constant(x_train)
        model = sm.OLS(y_train, x_train2)
        result = model.fit()
        print(result.summary())


    #Ridge and Lasso regression
    rlrenabled = False
    if rlrenabled:
        print('Running Ridge Regression:')
        regr = Ridge(alpha =0.5)
        #regr = Lasso(alpha = 0.1)
        regr.fit(x_train, y_train)
        #y_pred = regr.predict(x_train)
        #print('Coefficients: {}'.format(regr.coef_))
        #y_mean=y_train.mean()
        #mse = math.sqrt(mean_squared_error(y_train.values, y_pred))
        #print(y_mean)
        #print(mse)
        print('Mean squared error ratio: {}'.format(mse/y_mean))
        print('R-squared: {}'.format(r2_score(y_train, y_pred)))
        
        #on test data
        y_test_pred = regr.predict(x_test)
        #print('Coefficients: {}'.format(regr.coef_))
        y_test_mean=np.mean(y_test.values)
        mse_test=mean_squared_error(y_test.values, y_test_pred)
        #mse_test = np.mean((y_test - y_test_pred)**2)
        print(y_test_mean)
        print(mse_test)
        print('Mean squared error ratio: {}'.format(math.sqrt(mse_test)/y_test_mean))
        print('R-squared: {}'.format(r2_score(y_test, y_test_pred)))

    
    #support vector
    svmenabled = False
    if svmenabled:
        print('Running Support Vector Machine:')
        #regr = SVR(kernel='rbf', C=1e3, gamma=0.1)
        #regr = SVR(kernel='linear')
        regr = SVR(kernel='gaussian')
        regr.fit(x_train, y_train)
        y_pred = regr.predict(x_train)
        #rint('Coefficients: {}'.format(regr.coef_))
        y_mean=np.mean(y_train.values)
        mse = math.sqrt(mean_squared_error(y_train.values, y_pred))
        #print(y_mean)
        #print(mse)
        print('Mean squared error ratio: {}'.format(mse/y_mean))
        print('R-squared: {}'.format(r2_score(y_train, y_pred)))
        
        #on test data
        y_test_pred = regr.predict(x_test)
        #print('Coefficients: {}'.format(regr.coef_))
        y_test_mean=np.mean(y_test.values)
        mse_test=mean_squared_error(y_test.values, y_test_pred)
        #mse_test = np.mean((y_test - y_test_pred)**2)
        print(y_test_mean)
        print(mse_test)
        print('Mean squared error ratio: {}'.format(math.sqrt(mse_test)/y_test_mean))
        print('R-squared: {}'.format(r2_score(y_test, y_test_pred)))
   

    #neural net
    from keras.models import Sequential
    from keras.layers import Dense
    from sklearn.preprocessing import MinMaxScaler
    nrenabled = False
    if nrenabled:
        print('Running Neural Network:')
        #preprocessing the input data
        inputValues=inputValues.reset_index(drop=True)
        results=results.reset_index(drop=True)
        values_input = inputValues.values
        values_result = results.values
        #normalize the features
        scaler1 = MinMaxScaler()
        scaler2 = MinMaxScaler()
        values_input=values_input.astype('float32')
        values_input=scaler1.fit_transform(values_input)
        values_result=values_result.astype('float32')
        values_result=scaler2.fit_transform(values_result)
        train_hours=365*24*4
        x_train = values_input[:train_hours,:]
        y_train = values_result[:train_hours,:]
        x_test = values_input[train_hours:,:]
        y_test = values_result[train_hours:,:]
        #design network
        model = Sequential()
        #add layers one by one
        #keras use the default uniform distribution('uniform') to make the weight initialize between 0 and 0.05
        #12 is the number of neurons in the first layer,with 8 input variables
        model.add(Dense(12,input_dim=33,activation='relu'))#relu is the rectifier activation function
        #8 is the num of neuron in second hidden layer
        model.add(Dense(8,activation='relu'))#use relu for first two layers
        #1 is the num of neuron at the output
        model.add(Dense(1,activation='sigmoid'))#use sigmoid function for output
    
        #compile the model
        #use TensorFLow as backend
        #backend will choose the best way to train the model on CPU or GPU
        #train the network means to find the best weights in the NN
        #use efficient gradient descent algorithm "adam" for search through different weights for the NN
        #use "binary_crossentropy" as logrithmic loss function
        #loss function is used to evaluate a set of weights
        model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
        
        model.fit(x_train,y_train,epochs=50,batch_size=10)
        scores=model.evaluate(x_train,y_train)
        print("\n%s: %.2f%%" %(model.metrics_names[1],scores[1]*100))#print the loss and the accuracy for each epoch
        
        y_test_pred=model.predict(x_test)
        
        # invert scaling for prediction 
        inv_yhat = scaler2.inverse_transform(y_test_pred)
        inv_ytest = scaler2.inverse_transform(y_test)
        
        y_test_mean=np.mean(inv_ytest)
        mse_test=mean_squared_error(inv_ytest, inv_yhat)
  
        print(y_test_mean)
        print(mse_test)
        print('Mean squared error ratio: {}'.format(math.sqrt(mse_test)/y_test_mean))
        print('R-squared: {}'.format(r2_score(inv_ytest, inv_yhat)))
   

    #Decision Tree
    dtenabled = False
    if dtenabled:
        print('Running Decision Tree: ')
        regr = tree.DecisionTreeRegressor(max_depth=20)
        regr.fit(x_train, y_train)
        y_pred = regr.predict(x_train)
        y_mean=np.mean(y_train.values)
        mse = math.sqrt(mean_squared_error(y_train.values, y_pred))
        #print(y_mean)
        #print(mse)
        print('Mean squared error ratio: {}'.format(mse/y_mean))
        print('R-squared: {}'.format(r2_score(y_train, y_pred)))

        #on test data
        y_test_pred = regr.predict(x_test)
        factors=pd.DataFrame(columns=['Attribute', 'Importance'])
        for i in range(0, len(x_test.columns)):
            factors.loc[i, 'Attribute']=x_test.columns[i]
            factors.loc[i, 'Importance']=regr.feature_importances_[i]
        factors.to_csv('results/factors-decisiontree-'+site+'.csv')
        
        y_test_mean=np.mean(y_test.values)
        mse_test=mean_squared_error(y_test.values, y_test_pred)
        #mse_test = np.mean((y_test - y_test_pred)**2)
        print(y_test_mean)
        print(mse_test)
        print('Mean squared error ratio: {}'.format(math.sqrt(mse_test)/y_test_mean))
        print('R-squared: {}'.format(r2_score(y_test, y_test_pred)))
    
    
    #Bagged Decision Tree
    bdtenabled = False
    if bdtenabled:
        print('Running Bagged Decision Tree: ')
        regr = BaggingRegressor(
                    n_estimators=200,
                    max_samples=1.0,
                    max_features=1.0,
                    bootstrap=True,
                    bootstrap_features=False,
                    oob_score=False)
        regr.fit(x_train, y_train)
        y_pred = regr.predict(x_train)
        y_mean=np.mean(y_train.values)
        mse = math.sqrt(mean_squared_error(y_train.values, y_pred))
        #print(y_mean)
        #print(mse)
        print('Mean squared error ratio: {}'.format(mse/y_mean))
        print('R-squared: {}'.format(r2_score(y_train, y_pred)))

        #on test data
        y_test_pred = regr.predict(x_test)
        factors=pd.DataFrame(columns=['Attribute', 'Importance'])
        for i in range(0, len(x_test.columns)):
            factors.loc[i, 'Attribute']=x_test.columns[i]
            factors.loc[i, 'Importance']=regr.feature_importances_[i]
        #factors.to_csv('results/factors-decisiontree-'+site+'.csv')
        
        y_test_mean=np.mean(y_test.values)
        mse_test=mean_squared_error(y_test.values, y_test_pred)
        #mse_test = np.mean((y_test - y_test_pred)**2)
        print(y_test_mean)
        print(mse_test)
        print('Mean squared error ratio: {}'.format(math.sqrt(mse_test)/y_test_mean))
        print('R-squared: {}'.format(r2_score(y_test, y_test_pred)))
    
    
    #Random Forrest
    rfenabled = False
    if rfenabled:
        print('Running Random Forrest: ')
        regr = RandomForestRegressor(
                               n_estimators=1000,
                               min_impurity_decrease=1e-7,
                               bootstrap=True,
                               oob_score=True,
                               max_features='auto',
                                n_jobs=-1, verbose=True,
                                criterion='mse',
                               random_state=42)
        regr.fit(x_train, y_train)
 
        #on test data
        y_test_pred = regr.predict(x_test)
        factors=pd.DataFrame(columns=['Attribute', 'Importance'])
        for i in range(0, len(x_test.columns)):
            factors.loc[i, 'Attribute']=x_test.columns[i]
            factors.loc[i, 'Importance']=regr.feature_importances_[i]
        #factors.to_csv('results/factors-randomforrest-'+site+'.csv')
        
        y_test_mean=np.mean(y_test.values)
        mse_test=mean_squared_error(y_test.values, y_test_pred)
        #mse_test = np.mean((y_test - y_test_pred)**2)
        print(y_test_mean)
        print(mse_test)
        print('Mean squared error ratio: {}'.format(math.sqrt(mse_test)/y_test_mean))
        print('R-squared: {}'.format(r2_score(y_test, y_test_pred)))
   

    #Extra Tree
    etenabled = False
    if etenabled:
        print('Running Extra Tree: ')
        regr =  ExtraTreesRegressor(
                               n_estimators=1000,
                               min_impurity_decrease=1e-7,
                               bootstrap=True,
                               oob_score=True,
                               max_features='auto',
                                n_jobs=-1, verbose=True,
                                criterion='mse',
                               random_state=42)
        regr.fit(x_train, y_train)
 
        #on test data
        y_test_pred = regr.predict(x_test)
        factors=pd.DataFrame(columns=['Attribute', 'Importance'])
        for i in range(0, len(x_test.columns)):
            factors.loc[i, 'Attribute']=x_test.columns[i]
            factors.loc[i, 'Importance']=regr.feature_importances_[i]
        #factors.to_csv('results/factors-extratree-'+site+'.csv')
        
        y_test_mean=np.mean(y_test.values)
        mse_test=mean_squared_error(y_test.values, y_test_pred)
        #mse_test = np.mean((y_test - y_test_pred)**2)
        print(y_test_mean)
        print(mse_test)
        print('Mean squared error ratio: {}'.format(math.sqrt(mse_test)/y_test_mean))
        print('R-squared: {}'.format(r2_score(y_test, y_test_pred)))
    
    
    #Gradient Boosting
    gbenabled = True
    if gbenabled:
        print("Running GradientBoosting:")
        print(datetime.now())
        regr = GradientBoostingRegressor(
                               n_estimators=2000,
                               max_features='auto',
                                max_depth=5,
                                loss='ls',
                                verbose=True,
                                criterion='mse',
                                learning_rate= 0.0875,
                               random_state=42)
        regr.fit(x_train, y_train)
        y_pred = regr.predict(x_train)
        #print('Coefficients: {}'.format(regr.coef_))
        y_mean=y_train.mean()
        mse = math.sqrt(mean_squared_error(y_train.values, y_pred))
        print(y_mean)
        print(mse)
        print('Mean squared error ratio: {}'.format(mse/y_mean))
        print('R-squared: {}'.format(r2_score(y_train, y_pred)))
        
        #on test data
        y_test_pred = regr.predict(x_test)
        factors=pd.DataFrame(columns=['Attribute', 'Importance'])
        #for i in range(0, len(x_test.columns)):
            #factors.loc[i, 'Attribute']=x_test.columns[i]
            #factors.loc[i, 'Importance']=regr.feature_importances_[i]
        #factors.to_csv('results/factors-gradientboosting-'+site+'.csv')
        
        y_test_mean=np.mean(y_test.values)
        mse_test=mean_squared_error(y_test.values, y_test_pred)
        #mse_test = np.mean((y_test - y_test_pred)**2)
        print(y_test_mean)
        print(mse_test)
        print('Mean squared error ratio: {}'.format(math.sqrt(mse_test)/y_test_mean))
        print('R-squared: {}'.format(r2_score(y_test, y_test_pred)))     
        print(datetime.now())
        
    #XGB
    xgbenabled = False
    if xgbenabled:
        print("Running Extreme GradientBoosting:")
        print(datetime.now())
        regr = xgboost.XGBRegressor(n_estimators=2000, learning_rate=0.0875, max_depth=5)
        regr.fit(x_train, y_train)
        y_pred = regr.predict(x_train)
        #print('Coefficients: {}'.format(regr.coef_))
        y_mean=y_train.mean()
        mse = math.sqrt(mean_squared_error(y_train.values, y_pred))
        print(y_mean)
        print(mse)
        print('Mean squared error ratio: {}'.format(mse/y_mean))
        print('R-squared: {}'.format(r2_score(y_train, y_pred)))
        
        #on test data
        y_test_pred = regr.predict(x_test)
        factors=pd.DataFrame(columns=['Attribute', 'Importance'])
        #for i in range(0, len(x_test.columns)):
            #factors.loc[i, 'Attribute']=x_test.columns[i]
            #factors.loc[i, 'Importance']=regr.feature_importances_[i]
        #factors.to_csv('results/factors-gradientboosting-'+site+'.csv')
        
        y_test_mean=np.mean(y_test.values)
        mse_test=mean_squared_error(y_test.values, y_test_pred)
        #mse_test = np.mean((y_test - y_test_pred)**2)
        print(y_test_mean)
        print(mse_test)
        print('Mean squared error ratio: {}'.format(math.sqrt(mse_test)/y_test_mean))
        print('R-squared: {}'.format(r2_score(y_test, y_test_pred)))     
        print(datetime.now())

