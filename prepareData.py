import numpy as np
import pandas as pd

import dateutil.parser
import math
from datetime import datetime,timedelta
import holidays

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import tree
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

def northernVector(row):
    speed = row['ws']
    deg = row['wd']
    result = 0
    if not math.isnan(speed):
        rad = deg/360*2*math.pi
        result = math.cos(rad)*speed
    return result

def easternVector(row):
    speed = row['ws']
    deg = row['wd']
    result = 0
    if not math.isnan(speed):
        rad = deg/360*2*math.pi
        result = math.sin(rad)*speed
    return result
 
    
def prepareData(file, site):
    data = pd.read_csv(file)
    data = data.iloc[:,1:]
    #data.fillna(0,inplace=True)
    
    # featureNames=['Date','DayOfWeek', 'HourOfDay', 'IsWeekday', 'IsSaturday', 'IsSunday', 'IsHoliday',
    #               'DayOfWeekD1', 'HourOfDay','IsWeekdayD1', 'IsSaturdayD1', 'IsSundayD1', 'IsHolidayD1',
    #               'IsWeekdayD2', 'IsSaturdayD2', 'IsSundayD2', 'IsHolidayD2',
    #               'NO2Extrapolation', 'NO2D1', 'NO2D2', 'NO2Delta','Complete', 'WindSpeed', 'WindSpeedD1',
    #               'WindSpeedD2', 'Temperature', 'TemperatureD1', 'TemperatureD2', 'EastWind', 'EastWindD1', 
    #               'EastWindD2', ''NorthWind', 'NorthWindD1', 'NorthWindD2', 'Humidity', 'HumidityD1', 'HumidityD2'
    #               'Pressure', 'PressureD1', 'PressureD2' ]
    

    if site in ['GR9','GN3','GN2','GN4','GN0']: 
        data['date'] = pd.to_datetime(data['date'],format='%Y-%m-%d %H:%M')
    else:
        data['date'] = pd.to_datetime(data['date'],format='%d/%m/%Y %H:%M')

    
    ###treat outliers
    data['no2'] = np.where(data['no2']<0, None, data['no2'])
    data['temp'] = np.where(data['temp']< -30, None,data['temp'])
    data['temp'] = np.where(data['temp'] >50, None,data['temp'])
    data['ws'] = np.where(data['ws']<0, None,data['ws'])
    print(data.describe())
    
    
    ###create more features
    data['northwind']= data.apply(northernVector,axis=1)
    data['eastwind']=data.apply(easternVector,axis=1)
    d = data['date']
    #UK holidays
    from pandas.tseries.holiday import (
    AbstractHolidayCalendar, DateOffset, EasterMonday,
    GoodFriday, Holiday, MO,
    next_monday, next_monday_or_tuesday)
    class EnglandAndWalesHolidayCalendar(AbstractHolidayCalendar):
        rules = [
        Holiday('New Years Day', month=1, day=1, observance=next_monday),
        GoodFriday,
        EasterMonday,
        Holiday('Early May bank holiday',
                month=5, day=1, offset=DateOffset(weekday=MO(1))),
        Holiday('Spring bank holiday',
                month=5, day=31, offset=DateOffset(weekday=MO(-1))),
        Holiday('Summer bank holiday',
                month=8, day=31, offset=DateOffset(weekday=MO(-1))),
        Holiday('Christmas Day', month=12, day=25, observance=next_monday),
        Holiday('Boxing Day',
                month=12, day=26, observance=next_monday_or_tuesday)
        ]
    from datetime import date
    holidays = EnglandAndWalesHolidayCalendar().holidays()
    data['isholiday']=d.dt.date.astype('datetime64').isin(holidays).astype(int)
    data['dayofweek']=d.dt.dayofweek
    data['hourofday']=d.dt.hour
    data['issaturday']=data['dayofweek'].map(lambda x:1 if x==5 else 0).astype(int)
    data['issunday']=data['dayofweek'].map(lambda x:1 if x==6 else 0).astype(int)
    

    ### Derivation
    #get the timestamp of previous 24 and 48 hours
    subtractDay1 = timedelta(days=1)
    subtractDay2 = timedelta(days=2)
    data['dateD1'] = d - subtractDay1
    data['dateD2'] = d - subtractDay2
    

    #transform the data to dictionary
    data_copy = data.copy()
    #data_copy['date']=data_copy['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    #data_copy['date'] = data_copy['date'].apply(lambda x: datetime.strftime(x, '%Y-%m-%d %H:%M:%S'))
    data_copy.set_index('date', inplace=True)
    two_d_dict = data_copy.to_dict('index')
    
    #back count from now to 5 years data
    maxAge = 5*365
    now = datetime.now()
    subtractDay = timedelta(days=maxAge)
    startdatefilter = now-subtractDay 
    
    #start
    inputRows = []
    resultRows = []
    
    for i in range(0, len(data)):
        inputRow = {}
        resultRow = {}
        row = data.iloc[i]
        complete = True
        inputRow['Date'] = row['date']
        
        if (row['temp'] is None or math.isnan(row['temp']) 
            or row['no2'] is None or math.isnan(row['no2'])
            or row['ws'] is None or math.isnan(row['ws'])
            or row['bp'] is None or math.isnan(row['bp'])
            or row['rhum'] is None or math.isnan(row['rhum'])
            or row['wd'] is None or math.isnan(row['wd'])):

            complete=False
        
        inputRow['Temperature']=row['temp']
        inputRow['WindSpeed']=row['ws']
        inputRow['NorthWind']=row['northwind']
        inputRow['EastWind']=row['eastwind']
        inputRow['Pressure']=row['bp']
        inputRow['Humidity']=row['rhum']

        inputRow['DayOfWeek'] = row['dayofweek']
        inputRow['HourOfDay'] = row['hourofday']
        inputRow['IsHoliday'] = row['isholiday']
        inputRow['IsSaturday'] = row['issaturday']
        inputRow['IsSunday'] = row['issunday']
        
        
        d1 = row['dateD1']
        d2 = row['dateD2']
        #cd = dateutil.parser.parse(dk)
 
        #d1='{:%Y-%m-%d %H:%M:%S}'.format(d1)
        #d2='{:%Y-%m-%d %H:%M:%S}'.format(d2)
        
        if d1 in two_d_dict:
           inputRow['NO2D1'] = two_d_dict[d1]['no2']
           inputRow['TempD1'] = two_d_dict[d1]['temp']
           inputRow['WindSpeedD1']= two_d_dict[d1]['wd']
           inputRow['NorthWindD1']=two_d_dict[d1]['northwind']
           inputRow['EastWindD1']=two_d_dict[d1]['eastwind']
           inputRow['PressureD1']=two_d_dict[d1]['bp']
           inputRow['HumidityD1']=two_d_dict[d1]['rhum']
           inputRow['IsHolidayD1'] = two_d_dict[d1]['isholiday']
           inputRow['IsSaturdayD1'] = two_d_dict[d1]['issaturday']
           inputRow['IsSundayD1'] = two_d_dict[d1]['issunday']
            
           if (inputRow['NO2D1'] is None or math.isnan(inputRow['NO2D1']) 
               or inputRow['TempD1'] is None or math.isnan(inputRow['TempD1'])
               or inputRow['WindSpeedD1'] is None or math.isnan(inputRow['WindSpeedD1'])
               or inputRow['NorthWindD1'] is None or math.isnan(inputRow['NorthWindD1'])
               or inputRow['EastWindD1'] is None or math.isnan(inputRow['EastWindD1'])
               or inputRow['PressureD1'] is None or math.isnan(inputRow['PressureD1'])
               or inputRow['HumidityD1'] is None or math.isnan(inputRow['HumidityD1'])):     
               complete = False
        else:
           inputRow['NO2D1']=None
           inputRow['TempD1'] = None
           inputRow['WindSpeedD1']=None
           inputRow['NorthWindD1']=None
           inputRow['EastWindD1']=None
           inputRow['PressureD1']=None
           inputRow['HumidityD1']=None
           inputRow['IsHolidayD1'] = None
           inputRow['IsSaturdayD1'] = None
           inputRow['IsSundayD1'] = None
           complete = False
        
        if d2 in two_d_dict:
           inputRow['NO2D2'] = two_d_dict[d2]['no2']
           inputRow['TempD2'] = two_d_dict[d2]['temp']
           inputRow['WindSpeedD2']=two_d_dict[d2]['wd']
           inputRow['NorthWindD2']=two_d_dict[d2]['northwind']
           inputRow['EastWindD2']=two_d_dict[d2]['eastwind']
           inputRow['PressureD2']=two_d_dict[d2]['bp']
           inputRow['HumidityD2']=two_d_dict[d2]['rhum']
           inputRow['IsHolidayD2'] = two_d_dict[d2]['isholiday']
           inputRow['IsSaturdayD2'] = two_d_dict[d2]['issaturday']
           inputRow['IsSundayD2'] = two_d_dict[d2]['issunday']
           
           if (inputRow['NO2D2'] is None or math.isnan(inputRow['NO2D2']) 
                or inputRow['TempD2'] is None or math.isnan(inputRow['TempD2'])
                or inputRow['WindSpeedD2'] is None or math.isnan(inputRow['WindSpeedD2'])
                or inputRow['NorthWindD2'] is None or math.isnan(inputRow['NorthWindD2'])
                or inputRow['EastWindD2'] is None or math.isnan(inputRow['EastWindD2'])
                or inputRow['PressureD2'] is None or math.isnan(inputRow['PressureD2'])
                or inputRow['HumidityD2'] is None or math.isnan(inputRow['HumidityD2'])):                                                   
                
                complete = False
        else:
           inputRow['NO2D2']=None
           inputRow['TempD2'] = None
           inputRow['WindSpeedD2']=None
           inputRow['NorthWindD2']=None
           inputRow['EastWindD2']=None
           inputRow['PressureD2']=None
           inputRow['HumidityD2']=None
           inputRow['IsHolidayD2'] = None
           inputRow['IsSaturdayD2'] = None
           inputRow['IsSundayD2'] = None
           complete = False
                
        inputRow['Complete']=complete
        
        inputRows.append(inputRow)
        
        resultRow['Date']=row['date']
        resultRow['NO2Level']=row['no2']
        resultRow['Complete']=complete
        resultRows.append(resultRow)
            
    inputValues=pd.DataFrame(inputRows)
    results = pd.DataFrame(resultRows)
    
    inputValues['NO2Delta'] = inputValues['NO2D1']-inputValues['NO2D2']
    inputValues['NO2Extrap'] = 2*inputValues['NO2D1']-inputValues['NO2D2']

    return inputValues, results

print (datetime.now())


### Main
fileKCL = [
    'data/GR8.csv',
    'data/GR4.csv',
    'data/GR5.csv',
    'data/GR7.csv',
    'data/GN0.csv',
    'data/GR9.csv',
    'data/GN2.csv',
    'data/GN3.csv',
    'data/GN4.csv'    
]
sitecode=['GR8','GR4','GR5','GR7','GN0', 'GR9','GN2','GN3','GN4'] #GN2 and GR5 don't have 5 year data

for i in range(0, len(sitecode)):
    print('')
    print('Preparing data for site:{}'.format(sitecode[i]))
    inputValues, results=prepareData(fileKCL[i],sitecode[i])
    inputValues.to_csv('prepareddata/inputs-'+sitecode[i]+'.csv')
    results.to_csv('prepareddata/results-'+sitecode[i]+'.csv')
    print('Site {} Completed'.format(sitecode[i]))
    print('')
    print(datetime.now())
