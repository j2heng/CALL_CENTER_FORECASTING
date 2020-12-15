# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 11:03:37 2018

@author: zhengjie
"""
import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,TimeSeriesSplit
import pickle
import sys
   
class Model(object):
    
    def __init__(self, df, input_cols, output_col, now):     
        df['DATE'] = pd.to_datetime(df['DATE']) 
        
        self.now = now 
        self.dftrain = df[df['DATE'] < self.now].copy()
        self.dftest = df[df['DATE'] >= self.now] .copy()
        self.input_cols =input_cols
        self.output_col = output_col
        self.dftest.fillna(method='ffill',inplace=True )

    def fit(self): 
        param_dist = {"n_estimators":[100,200,300],
                      "max_depth": range(5,20,2),
                      "max_features": ["sqrt",0.33,0.5,0.7,0.9], 
                      "min_samples_leaf": range(2,6), 
                      "criterion": ["mse"],
                      "random_state":[123],
                      "n_jobs": [-1]
                      } 
        RF = RandomForestRegressor()
        index_notnull = self.dftrain[~(self.dftrain.isnull().any(axis=1))].index.values.tolist()
        # cv = list(TimeSeriesSplit(n_splits=5).split(self.dftrain.iloc[index_notnull]))
        cv = 3
        # 1. Classic training 
        #self.model = RandomizedSearchCV(RF, param_distributions=param_dist, n_iter=50, cv = cv, verbose=3, n_jobs=-1)   
        # OR
        self.model = GridSearchCV(RF, param_grid=param_dist, cv = cv , verbose=3, n_jobs=-1)  
        self.model.fit(self.dftrain[self.input_cols].iloc[index_notnull],self.dftrain[self.output_col].iloc[index_notnull])
        
        print(self.model.cv_results_)
        print(self.model.best_estimator_)
        
        self.model = clone(self.model.best_estimator_).fit(self.dftrain[self.input_cols].iloc[index_notnull],self.dftrain[self.output_col].iloc[index_notnull])
        return self.model
        
    def predict(self):
        return self.model.predict(self.dftest[self.input_cols])

    def save(self, name):
        filename = name+'.pkl'
        pickle.dump(self.model, open(filename, 'wb'))
    
    def load(self, name):
        self.model = pickle.load(open(name, 'rb'))
        return self.model

    @property
    def __inputcols__(self):
        return self.input_cols

    @property
    def __outputcol__(self):
        return self.output_col

    

        
class ShortTermModel(Model):
     
    def __init__(self, dataframe,now):
        output_c = 'y'
        # input_c = ['HALF_HOURL', 'HALF_HOURC', 'xMONTH', 'yMONTH', 'xDAY', 'yDAY', 'xWD', 'yWD', 'xHOUR', 'yHOUR',
        #     'TSE', 'JF', 'HolidaysFR', 'AftJF', 'BefJF', 'DAY_LENGTH', 'LIGHT', 'MG_1', 'MG_2', 'MG_3', 'WDG_1', 'WDG_2', 
        #     'WDG_3', 'WDG_4', 'HG_1', 'HG_2', 'HG_3', 'MEAN_LAGGED', 'STD_LAGGED', 'LAGGED_LY_DW_avg']
        # input_c = ['xMONTH', 'yMONTH', 'xDAY', 'yDAY', 'xWD', 'yWD', 'xHOUR', 'yHOUR',
        #     'TSE', 'JF', 'HolidaysFR', 'AftJF', 'BefJF', 'MEAN_LAGGED', 'STD_LAGGED', 'LAGGED_LY_DW_avg']
        input_c = ['xMONTH', 'yMONTH', 'xDAY', 'yDAY', 'xWD', 'yWD', 'xHOUR', 'yHOUR',
            'JF', 'HolidaysFR', 'AftJF', 'BefJF', 'MEAN_LAGGED', 'STD_LAGGED']
        # input_c = ['xMONTH', 'yMONTH', 'xDAY', 'yDAY', 'xWD', 'yWD', 'xHOUR', 'yHOUR',
        #     'JF', 'HolidaysFR', 'AftJF', 'BefJF', 'LAGGED_LY_DW_avg']
        super(ShortTermModel,self).__init__(dataframe,input_c,output_c,now )
       
    def fit(self):
         return super(ShortTermModel,self).fit()
         
    def predict(self):
         return super(ShortTermModel,self).predict()

    def save(self,name):
        return super(ShortTermModel,self).save(name)
    
    def load(self,name):
        return super(ShortTermModel,self).load(name)
        
        
        
class MidTermModel(Model):
     
    def __init__(self, dataframe, now):
        output_c = 'y'
        # input_c =  ['HALF_HOURL', 'HALF_HOURC', 'xMONTH', 'yMONTH', 'xDAY', 'yDAY', 'xWD', 'yWD', 'TSE', 'JF','HolidaysFR', 'AftJF', 'BefJF',
        #            'DAY_LENGTH', 'MG_1', 'MG_2', 'MG_3', 'WDG_1', 'WDG_2', 'WDG_3', 'WDG_4', 'MEAN_LAGGED','STD_LAGGED', 'LAGGED_LY_DW_avg']
        # input_c =  ['xMONTH', 'yMONTH', 'xDAY', 'yDAY', 'xWD','yWD', 'JF','AftJF','BefJF','HolidaysFR','DAY_LENGTH']
        # input_c =  ['MEAN_LAGGED','STD_LAGGED', 'LAGGED_LY_DW_avg']
        # input_c =  ['xMONTH', 'yMONTH', 'xDAY', 'yDAY', 'xWD','yWD', 'JF','AftJF','BefJF','HolidaysFR','DAY_LENGTH', 'MEAN_LAGGED','STD_LAGGED']
        # input_c =  ['xMONTH', 'yMONTH', 'xDAY', 'yDAY', 'xWD','yWD', 'JF','AftJF','BefJF','HolidaysFR','DAY_LENGTH', 'LAGGED_LY_DW_avg']
        # input_c =  ['HALF_HOURL', 'HALF_HOURC', 'TSE', 'JF','HolidaysFR', 'AftJF', 'BefJF',
        #            'DAY_LENGTH', 'MG_1', 'MG_2', 'MG_3', 'WDG_1', 'WDG_2', 'WDG_3', 'WDG_4', 'MEAN_LAGGED','STD_LAGGED']
        input_c =  ['HALF_HOURL', 'HALF_HOURC', 'TSE', 'JF','HolidaysFR', 'AftJF', 'BefJF',
                   'DAY_LENGTH', 'MG_1', 'MG_2', 'MG_3', 'WDG_1', 'WDG_2', 'WDG_3', 'WDG_4', 'LAGGED_LY_DW_avg']
        super(MidTermModel,self).__init__(dataframe,input_c,output_c,now )
       
    def fit(self):
         return super(MidTermModel,self).fit()
         
    def predict(self):
         return super(MidTermModel,self).predict()

    def save(self,name):
        return super(MidTermModel,self).save(name)
    
    def load(self,name):
        return super(MidTermModel,self).load(name)
    