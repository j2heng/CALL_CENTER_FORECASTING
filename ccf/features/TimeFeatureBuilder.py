# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 10:33:28 2018
@author: zhengji
"""

import numpy as np
import pandas as pd
import datetime as dt
import warnings
warnings.filterwarnings("ignore")
from .FEHelper import *
import time
import sys

class BaseFeatureBuilding(object):
     
    def __init__(self, df, ephemeride, start, end, freq):
        self.df = df
        self.ephemeride = ephemeride
        self.start = start
        self.end = end 
        self.freq = freq
        
        # convert df.TIME from string to datetime
        self.df.TIME = pd.to_datetime(self.df.TIME)
        
    def __getitem__(self, key):
        return self.df[key]
       
    
    def fillna(self):
        """Make date index continuous in self.df"""
        if (not(FEHelper.isContinuous(self.df.TIME, self.start, self.end, self.freq))):
            df_missingdate = pd.DataFrame(index=pd.date_range(start=self.start, end=self.end, freq=self.freq))
            df_missingdate = df_missingdate.reset_index()
            df_missingdate = df_missingdate.rename(columns = {"index":"TIME"})

            self.df = df_missingdate.merge(self.df, on='TIME', how='left')
            self.df = self.df.fillna(0)
        
   
    def merge(self):
        """Merge self.df with self.ephemeride"""
        self.ephemeride["HOURL"] = self.ephemeride["LEVER"].apply(lambda x: dt.datetime.strptime(str(x),'%Hh%M').time().hour)  
        self.ephemeride["MINL"] = self.ephemeride["LEVER"].apply(lambda x: dt.datetime.strptime(str(x),'%Hh%M').time().minute/ 60.0)  
        self.ephemeride["HOURC"] = self.ephemeride["COUCHER"].apply(lambda x: dt.datetime.strptime(str(x),'%Hh%M').time().hour)  
        self.ephemeride["MINC"] = self.ephemeride["COUCHER"].apply(lambda x: dt.datetime.strptime(str(x),'%Hh%M').time().minute/ 60.0)  
        self.ephemeride["HALF_HOURL"] = self.ephemeride["HOURL"] + self.ephemeride["MINL"]
        self.ephemeride["HALF_HOURC"] = self.ephemeride["HOURC"] + self.ephemeride["MINC"]
        
        self.df["DATE"] = pd.to_datetime(self.df["TIME"].apply(lambda x : x.date()))       
        self.df["MM_DD"] = self.df["DATE"].map(lambda x: str(x.month).zfill(2) + '-' + str(x.day).zfill(2))       
        self.df =  self.df.merge(self.ephemeride[['HALF_HOURL','HALF_HOURC','MM_DD']], on='MM_DD', how='left').copy()
    
    def addBasicTimeFeatures(self):
        """add basic time features"""
        self.df["YEAR"] = self.df['TIME'].apply(lambda x : x.year)
        self.df["MONTH"] = self.df['TIME'].apply(lambda x : x.month)
        self.df["DAY"] = self.df['TIME'].apply(lambda x : x.day)
        self.df['WEEK_DAY'] = self.df['TIME'].apply(lambda x: x.weekday())
        
    def addPeriodicTimeFeatures(self):
        """add sin/cos time features"""
        self.df['xMONTH'] =  self.df['MONTH'].map(lambda x : np.sin(2*np.pi*x/12))
        self.df['yMONTH'] =  self.df['MONTH'].map(lambda x : np.cos(2*np.pi*x/12))
        self.df['xDAY'] =  self.df['DAY'].map(lambda x : np.sin(2*np.pi*x/31))
        self.df['yDAY'] =  self.df['DAY'].map(lambda x : np.cos(2*np.pi*x/31))
        self.df['xWD'] =  self.df['WEEK_DAY'].map(lambda x : np.sin(2*np.pi*x/7))
        self.df['yWD'] =  self.df['WEEK_DAY'].map(lambda x : np.cos(2*np.pi*x/7))
        
    def addEphemerideFeatures(self):    
        """add ephemeride related features"""
        self.df['DAY_LENGTH'] = self.df['HALF_HOURC'] - self.df['HALF_HOURL']
    
    def addTimeGroupFeatures(self):
        """create time groups and dummy them"""
        self.df['MONTH_GROUP'] = self.df['MONTH'].map(FEHelper.Monthgroup)
        self.df['WD_GROUP'] = self.df['WEEK_DAY'].map(FEHelper.Daygroup)
        self.df = self.df.join(pd.get_dummies(self.df['MONTH_GROUP'], prefix='MG'))
        self.df = self.df.join(pd.get_dummies(self.df['WD_GROUP'], prefix='WDG'))
    
    def addLags(self):
        """add lags"""
        pass

    def addTSE(self):
        """add Time Since Epoch features"""      
        epoch = min(self.df.DATE)
        self.df['TSE'] = self.df.DATE.map(lambda x: x-epoch)
        self.df['TSE'] = self.df['TSE']/ np.timedelta64(1, 'D')
        
            
    def addHolidays(self):
        holidays = FEHelper.lholidays()    
        self.df = self.df.merge(holidays, left_on='DATE', right_on='ds', how='left')
        self.df.drop(['ds'], axis=1,inplace=True)
        self.df['holiday'].fillna(value=0,inplace=True)

        aftholidays = FEHelper.aftholidays()   
        self.df = self.df.merge(aftholidays, left_on='DATE', right_on='ds', how='left')
        self.df.drop(['ds'], axis=1, inplace=True)
        self.df['aftholiday'].fillna(value=0, inplace=True)
        
        list_public_holidays = [x.strftime('%Y-%m-%d') for x in holidays['ds']]
        self.df['DATE'] = self.df['DATE'].apply(lambda x: x.strftime('%Y-%m-%d'))
        self.df['JF'] = self.df['DATE'].apply(lambda x: int(x in list_public_holidays))


        """ 2. Add school holidays"""
        vac_toussaint = ((self.df.DATE >= '2012-10-27') & (self.df.DATE < '2012-11-08')) + \
            ((self.df.DATE >= '2013-10-19') & (self.df.DATE < '2013-11-04')) + \
            ((self.df.DATE >= '2014-10-18') & (self.df.DATE < '2014-11-03')) + \
            ((self.df.DATE >= '2015-10-17') & (self.df.DATE < '2015-11-02')) + \
            ((self.df.DATE >= '2016-10-19') & (self.df.DATE < '2016-11-03')) + \
            ((self.df.DATE >= '2017-10-21') & (self.df.DATE < '2017-11-06')) + \
            ((self.df.DATE >= '2018-10-20') & (self.df.DATE < '2018-11-05')) + \
            ((self.df.DATE >= '2019-10-19') & (self.df.DATE < '2019-11-04')) + \
            ((self.df.DATE >= '2020-10-17') & (self.df.DATE < '2020-11-02'))  
        vac_noel = ((self.df.DATE >= '2012-01-01') & (self.df.DATE < '2012-01-03')) + \
            ((self.df.DATE >= '2012-12-22') & (self.df.DATE < '2013-01-07')) + \
            ((self.df.DATE >= '2013-12-21') & (self.df.DATE < '2014-01-06')) + \
            ((self.df.DATE >= '2014-12-20') & (self.df.DATE < '2015-01-05')) + \
            ((self.df.DATE >= '2015-12-19') & (self.df.DATE < '2016-01-04')) + \
            ((self.df.DATE >= '2016-12-17') & (self.df.DATE < '2017-01-03')) + \
            ((self.df.DATE >= '2017-12-23') & (self.df.DATE < '2018-01-08')) + \
            ((self.df.DATE >= '2018-12-22') & (self.df.DATE < '2019-01-07')) + \
            ((self.df.DATE >= '2019-12-21') & (self.df.DATE < '2020-01-06')) + \
            ((self.df.DATE >= '2020-12-19') & (self.df.DATE < '2021-01-04'))
        vac_hiver = ((self.df.DATE >= '2012-02-11') & (self.df.DATE < '2012-03-12')) + \
            ((self.df.DATE >= '2013-02-16') & (self.df.DATE < '2013-03-18')) + \
            ((self.df.DATE >= '2014-02-15') & (self.df.DATE < '2014-03-17')) + \
            ((self.df.DATE >= '2015-02-07') & (self.df.DATE < '2015-03-09')) + \
            ((self.df.DATE >= '2016-02-07') & (self.df.DATE < '2016-03-07')) + \
            ((self.df.DATE >= '2017-02-04') & (self.df.DATE < '2017-03-06')) + \
            ((self.df.DATE >= '2018-02-10') & (self.df.DATE < '2018-03-12')) + \
            ((self.df.DATE >= '2019-02-08') & (self.df.DATE < '2019-03-09')) + \
            ((self.df.DATE >= '2020-02-08') & (self.df.DATE < '2020-03-09')) + \
            ((self.df.DATE >= '2021-02-06') & (self.df.DATE < '2021-03-08')) 
        vac_printemps = ((self.df.DATE >= '2012-04-07') & (self.df.DATE < '2012-05-07')) + \
            ((self.df.DATE >= '2013-04-13') & (self.df.DATE < '2013-05-13')) + \
            ((self.df.DATE >= '2014-04-12') & (self.df.DATE < '2014-05-28')) + \
            ((self.df.DATE >= '2015-04-11') & (self.df.DATE < '2015-05-11')) + \
            ((self.df.DATE >= '2016-04-02') & (self.df.DATE < '2016-05-02')) + \
            ((self.df.DATE >= '2017-04-01') & (self.df.DATE < '2017-05-02')) + \
            ((self.df.DATE >= '2018-04-07') & (self.df.DATE < '2018-05-07')) + \
            ((self.df.DATE >= '2019-04-04') & (self.df.DATE < '2019-05-04')) + \
            ((self.df.DATE >= '2020-04-04') & (self.df.DATE < '2020-05-04')) + \
            ((self.df.DATE >= '2021-04-10') & (self.df.DATE < '2021-05-10'))
        vac_ete = ((self.df.DATE >= '2012-07-05') & (self.df.DATE < '2012-09-03')) + \
            ((self.df.DATE >= '2013-07-06') & (self.df.DATE < '2013-09-03')) + \
            ((self.df.DATE >= '2014-07-05') & (self.df.DATE < '2014-09-01')) + \
            ((self.df.DATE >= '2015-07-04') & (self.df.DATE < '2015-09-01')) + \
            ((self.df.DATE >= '2016-07-06') & (self.df.DATE < '2016-09-01')) + \
            ((self.df.DATE >= '2017-07-08') & (self.df.DATE < '2017-09-04')) + \
            ((self.df.DATE >= '2018-07-07') & (self.df.DATE < '2018-09-03')) + \
            ((self.df.DATE >= '2019-07-04') & (self.df.DATE < '2019-09-02')) + \
            ((self.df.DATE >= '2020-07-04') & (self.df.DATE < '2020-09-01')) + \
            ((self.df.DATE >= '2021-07-06') & (self.df.DATE < '2021-09-01'))

        self.df['HolidaysFR'] = (vac_toussaint + vac_noel + vac_hiver + vac_printemps + vac_ete).astype('int')

    
    def transform(self):
        self.fillna()
        self.merge()
        self.addBasicTimeFeatures()
        self.addPeriodicTimeFeatures()
        self.addTSE()
        self.addHolidays()
        self.addEphemerideFeatures()  # // proper for J+7 model
        self.addTimeGroupFeatures()   # // proper for J+7 / M+3 model
        self.addLags()  # // proper for J+7 / M+3 model

        return self.df


class DayFeatureBuilding(BaseFeatureBuilding):
     
    def __init__(self, df, ephemeride, start, end):     
        super(DayFeatureBuilding,self).__init__(df, ephemeride, start, end, 'D')
        
    def __getitem__(self, key):    
        return super(DayFeatureBuilding, self).__getitem__(key)  
    
    def fillna(self):
        """Make date index continuous in self.df"""
        super(DayFeatureBuilding, self).fillna()
        
   
    def merge(self):
        """Merge self.df with self.ephemeride"""
        super(DayFeatureBuilding, self).merge()
    
    def addBasicTimeFeatures(self):
        """add basic time features"""
        super(DayFeatureBuilding, self).addBasicTimeFeatures()
        
    def addPeriodicTimeFeatures(self):
        """add sin/cos time features"""
        super(DayFeatureBuilding, self).addPeriodicTimeFeatures()

    def addEphemerideFeatures(self):    
        """add ephemeride related features"""
        super(DayFeatureBuilding, self).addEphemerideFeatures()
    
    def addTimeGroupFeatures(self):
        """create time groups and dummy them"""
        super(DayFeatureBuilding, self).addTimeGroupFeatures()
    
    def addLags(self):
        """add lags"""
        def shifted_mois(x):
            if self.df['MONTH'].iloc[x] > 3:
                shifted = self.df[(self.df['MONTH'] == self.df['MONTH'].iloc[x]-3) & \
                            (self.df['YEAR'] == self.df['YEAR'].iloc[x]) & \
                            (self.df['WEEK_DAY'] == self.df['WEEK_DAY'].iloc[x]) & \
                            (self.df['JF'] == 0)  & (self.df['AftJF'] == 0) ].index.values
            elif self.df['MONTH'].iloc[x] <= 3:
                shifted = self.df[(self.df['MONTH'] == (self.df['MONTH'].iloc[x]-4)%13) & \
                            (self.df['YEAR'] == self.df['YEAR'].iloc[x]-1) & \
                            (self.df['WEEK_DAY'] == self.df['WEEK_DAY'].iloc[x]) & \
                            (self.df['JF'] == 0)  & (self.df['AftJF'] == 0)].index.values
            return (self.df['y'].iloc[shifted].values)        
        
        self.df['LAGGED_3_MONTH_VALUES_WEEKDAY'] = self.df.index.map(lambda x : shifted_mois(x))
        self.df['MEAN_LAGGED'] = self.df.index.map(lambda x : np.nanmean(self.df['LAGGED_3_MONTH_VALUES_WEEKDAY'].iloc[x] ) if (len(self.df['LAGGED_3_MONTH_VALUES_WEEKDAY'].iloc[x]) > 0)  else (self.df['y'].iloc[x] ))        
        
        self.df['LAGGED_1'] = self.df.index.map(lambda x : self.df['LAGGED_3_MONTH_VALUES_WEEKDAY'].iloc[x][len(self.df['LAGGED_3_MONTH_VALUES_WEEKDAY'].iloc[x])-1] if len(self.df['LAGGED_3_MONTH_VALUES_WEEKDAY'].iloc[x]) > 0 else self.df['MEAN_LAGGED'].iloc[x])
        self.df['LAGGED_2'] = self.df.index.map(lambda x : self.df['LAGGED_3_MONTH_VALUES_WEEKDAY'].iloc[x][len(self.df['LAGGED_3_MONTH_VALUES_WEEKDAY'].iloc[x])-2] if len(self.df['LAGGED_3_MONTH_VALUES_WEEKDAY'].iloc[x]) > 1 else self.df['MEAN_LAGGED'].iloc[x])
        self.df['LAGGED_3'] = self.df.index.map(lambda x : self.df['LAGGED_3_MONTH_VALUES_WEEKDAY'].iloc[x][len(self.df['LAGGED_3_MONTH_VALUES_WEEKDAY'].iloc[x])-3] if len(self.df['LAGGED_3_MONTH_VALUES_WEEKDAY'].iloc[x]) > 2 else self.df['MEAN_LAGGED'].iloc[x])
        self.df['LAGGED_4'] = self.df.index.map(lambda x : self.df['LAGGED_3_MONTH_VALUES_WEEKDAY'].iloc[x][len(self.df['LAGGED_3_MONTH_VALUES_WEEKDAY'].iloc[x])-4] if len(self.df['LAGGED_3_MONTH_VALUES_WEEKDAY'].iloc[x]) > 3 else self.df['MEAN_LAGGED'].iloc[x])
        self.df['STD_LAGGED'] = self.df.index.map(lambda x : np.nanstd(self.df['LAGGED_3_MONTH_VALUES_WEEKDAY'].iloc[x] ))    
        self.df['LAGGED_LY_DW_avg'] = self.df.apply(lambda x: np.nanmean(self.df[(self.df['YEAR'] == x['YEAR']-1)  & (self.df['MONTH'] == x['MONTH']) & (self.df['WEEK_DAY'] == x['WEEK_DAY']) & (self.df['JF'] == 0)  & (self.df['AftJF'] == 0)]['y']) if (x['YEAR']>min(self.df.YEAR)) else x['y'] , axis=1)

        # Same holiday of last years        
        holidays = FEHelper.lholidays()
        for h in set(holidays.holiday):
            self.df.loc[self.df['holiday']==h, 'LAGGED_LY_DW_avg'] = self.df.loc[self.df['holiday']==h].apply(lambda x : np.nanmean(self.df[(self.df['holiday']==h) & (self.df['YEAR'] < x['YEAR'])]['y'] if (x['YEAR']>min(self.df.YEAR)) else x['y']), axis=1)

        # Same after holiday of last years
        aftholidays = FEHelper.aftholidays()
        for h in set(aftholidays.aftholiday):
            self.df.loc[self.df['aftholiday']==h, 'LAGGED_LY_DW_avg'] = self.df.loc[self.df['aftholiday']==h].apply(lambda x : np.mean(self.df[(self.df['aftholiday']==h) & (self.df['YEAR'] < x['YEAR'])]['y'] if (x['YEAR']>min(self.df.YEAR)) else x['y']), axis=1)
            
        self.df = self.df.fillna(0)
 
    def addTSE(self):
        """add Time Since Epoch features"""      
        super(DayFeatureBuilding, self).addTSE()
        
            
    def addHolidays(self):
        super(DayFeatureBuilding, self).addHolidays()
        
        """ 3. Add bef/aft public holidays"""
        self.df["AftJF"] = self.df["JF"].shift(1)
        self.df["BefJF"] = self.df["JF"].shift(-1)
        self.df.ix[self.df['JF']==1,'AftJF']=0
        self.df.ix[self.df['JF']==1,'BefJF']=0
        # jour ouvre
        idx_JF_special = self.df[(self.df.JF==1) &(self.df.WEEK_DAY==4) ].index.values
        idx_max = max(self.df.index.values)
        for tmp in idx_JF_special:
            if tmp+1<=idx_max:
                self.df['AftJF'].iloc[tmp+1]=0
            if tmp+3<=idx_max:
                self.df['AftJF'].iloc[tmp+3]=1
    
    def transform(self):
        return super(DayFeatureBuilding, self).transform()


class HourFeatureBuilding(BaseFeatureBuilding):
     
    def __init__(self, df, ephemeride, start, end):
        super(HourFeatureBuilding,self).__init__(df, ephemeride, start, end, 'H')
        
    def __getitem__(self, key):
        return super(HourFeatureBuilding, self).__getitem__(key)    
    
    def fillna(self):
        """Make date index continuous in self.df"""
        super(HourFeatureBuilding, self).fillna()
        
   
    def merge(self):
        """Merge self.df with self.ephemeride"""
        super(HourFeatureBuilding, self).merge()
    
    def addBasicTimeFeatures(self):
        """add basic time features"""
        super(HourFeatureBuilding, self).addBasicTimeFeatures()
        self.df["HOUR"] = self.df['TIME'].apply(lambda x : x.hour) 
        
    def addPeriodicTimeFeatures(self):
        """add sin/cos time features"""
        super(HourFeatureBuilding, self).addPeriodicTimeFeatures()
        self.df['xHOUR'] =  self.df['HOUR'].map(lambda x : np.sin(2*np.pi*x/24))
        self.df['yHOUR'] =  self.df['HOUR'].map(lambda x : np.cos(2*np.pi*x/24))

        
    def addEphemerideFeatures(self):    
        """add ephemeride related features"""
        super(HourFeatureBuilding, self).addEphemerideFeatures()
        self.df['LIGHT'] = self.df.index.map(lambda x : int(self.df['HALF_HOURL'].iloc[x]<=self.df['HOUR'].iloc[x]<=self.df['HALF_HOURC'].iloc[x]))
    
    def addTimeGroupFeatures(self):
        """create time groups and dummy them"""
        super(HourFeatureBuilding, self).addTimeGroupFeatures()
        self.df['HOUR_GROUP'] = self.df['HOUR'].map(FEHelper.Hourgroup)
        self.df = self.df.join(pd.get_dummies(self.df['HOUR_GROUP'], prefix='HG'))
    
    def addLags(self):
        """add lags"""
        if FEHelper.isContinuous(self.df.TIME, self.start, self.end, self.freq):
            
            self.df["LAGGED_1"]=self.df["y"].shift(168) #lag J-7 à la même heure
            self.df["LAGGED_2"]=self.df["y"].shift(336) #lag J-14 à la même heure
            self.df["LAGGED_3"]=self.df["y"].shift(504) #lag J-21 à la même heure
            self.df["LAGGED_4"]=self.df["y"].shift(672) #lag J-28 à la même heure
            
            # Question: whether to handle nan values in lags
            self.df["LAGGED_1"].fillna(self.df['y'], inplace=True)
            self.df["LAGGED_2"].fillna(self.df['y'], inplace=True)
            self.df["LAGGED_3"].fillna(self.df['y'], inplace=True)
            self.df["LAGGED_4"].fillna(self.df['y'], inplace=True)
            
            self.df["MEAN_LAGGED"]=self.df.index.map(lambda x : np.mean([self.df["LAGGED_1"].iloc[x],self.df["LAGGED_2"].iloc[x],self.df["LAGGED_3"].iloc[x],self.df["LAGGED_4"].iloc[x]]))
            self.df["STD_LAGGED"]=self.df.index.map(lambda x : np.std([self.df["LAGGED_1"].iloc[x],self.df["LAGGED_2"].iloc[x],self.df["LAGGED_3"].iloc[x],self.df["LAGGED_4"].iloc[x]]))
            
            # Same hour, weekday, month of last year
            self.df['LAGGED_LY_DW_avg'] = self.df.apply(lambda x: np.mean(self.df[(self.df['YEAR'] == x['YEAR']-1) & (self.df['MONTH'] == x['MONTH']) & (self.df['WEEK_DAY'] == x['WEEK_DAY'])  & (self.df['HOUR'] == x['HOUR']) & (self.df['JF'] == 0)  & (self.df['AftJF'] == 0)]['y']) if (x['YEAR']>min(self.df.YEAR)) else x['y'] , axis=1)
            # Same hour, holiday of last years
            holidays = FEHelper.lholidays()
            for h in set(holidays.holiday):
                self.df.loc[self.df['holiday']==h, 'LAGGED_LY_DW_avg'] = self.df.loc[self.df['holiday']==h].apply(lambda x : np.mean(self.df[(self.df['holiday']==h) & (self.df['YEAR'] < x['YEAR']) & (self.df['HOUR'] == x['HOUR'])]['y']
                                                if (x['YEAR']>min(self.df.YEAR)) else x['y']), axis=1)
    def addTSE(self):
        """add Time Since Epoch features"""      
        super(HourFeatureBuilding, self).addTSE()
        
            
    def addHolidays(self):
        super(HourFeatureBuilding, self).addHolidays()
        
        """ 3. Add bef/aft public holidays"""
        self.df["AftJF"] = self.df["JF"].shift(24)
        self.df["BefJF"] = self.df["JF"].shift(-24)
        self.df.ix[self.df['JF']==1,'AftJF']=0
        self.df.ix[self.df['JF']==1,'BefJF']=0
        # jour ouvre
        idx_JF_special = self.df[(self.df.JF==1) &(self.df.WEEK_DAY==4) ].index.values
        idx_max = max(self.df.index.values)
        for tmp in idx_JF_special:
            if tmp+24<=idx_max:
                self.df['AftJF'].iloc[tmp+24]=0
            if tmp+72<=idx_max:
                self.df['AftJF'].iloc[tmp+72]=1
    
    def transform(self):
        return super(HourFeatureBuilding, self).transform()