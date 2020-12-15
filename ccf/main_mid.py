#!/usr/bin/env python

import sys
import pandas as pd
import numpy as np
import datetime, time
from features.TimeFeatureBuilder import DayFeatureBuilding
import pickle

def inference(now, inputPath):
    # Get Data
    Ephemeride = pd.read_csv("./data/Ephemeride_Paris.csv", sep=';')
    df =  pd.read_csv(inputPath,sep=";") 
    # metter en commentaire ces deux lignes si prediction j a j
    df['time'] = pd.to_datetime(df['time']) 
    df = df.loc[df['time'] <now]
    print("***0")

    index = pd.date_range(now, now+datetime.timedelta(days=91), freq='D')
    df_test = pd.DataFrame({ df.columns[0]: index, df.columns[1]: np.nan })[df.columns].fillna(0)
    df_new = pd.concat([df, df_test], axis=0)[df.columns]
    df_new.reset_index(inplace=True, drop=True)
    df_new.columns = ['TIME', 'y']
    df = df_new.copy()
    print("***1")

    # Feature Engineering      
    df = DayFeatureBuilding(df,Ephemeride, '2012-01-01',(now+datetime.timedelta(days=91)).strftime("%Y-%m-%d"))
    df = df.transform()
    df_pred= df.loc[df['DATE'] >= now.strftime("%Y-%m-%d")]
    print("***2")

    return df, df_test, df_pred


####################################################       
##################  Main function  ####################
####################################################        

def main_mid_call():
    """
    Inference of midterm call 
    """
    # now = datetime.datetime.now().date()
    now = datetime.datetime.strptime('2020-01-01', '%Y-%m-%d')   
    inputPath = "./data/NbcallDay.csv"
    df, df_test, df_pred = inference(now, inputPath)
 
    # Load pretrained model and predict
    # first
    input_c_1 = ['xMONTH', 'yMONTH', 'xDAY', 'yDAY', 'xWD','yWD', 'JF','AftJF','BefJF','HolidaysFR','DAY_LENGTH', 'MEAN_LAGGED','STD_LAGGED']
    model_path_1 = "./ccf/model_retraining_result/pkl/RF-midcall-123-300-17-0.5-2-9.39032.pkl"
    rg_1 = pickle.load(open(model_path_1, 'rb'))
    prediction_1 = rg_1.predict(df_pred[input_c_1])
    # second
    input_c_2 =  ['HALF_HOURL', 'HALF_HOURC', 'xMONTH', 'yMONTH', 'xDAY', 'yDAY', 'xWD', 'yWD', 'TSE', 'JF','HolidaysFR', 'AftJF', 'BefJF',
                   'DAY_LENGTH', 'MG_1', 'MG_2', 'MG_3', 'WDG_1', 'WDG_2', 'WDG_3', 'WDG_4', 'MEAN_LAGGED','STD_LAGGED', 'LAGGED_LY_DW_avg']
    model_path_2 = "./ccf/model_retraining_result/pkl/RF-midcall-123-200-9-0.33-2-10.58734.pkl"
    rg_2 = pickle.load(open(model_path_2, 'rb'))
    prediction_2 = rg_2.predict(df_pred[input_c_2])
    # take average
    prediction = 1/2*prediction_1 + 1/2*prediction_2
    print("***3")

    # Save Result
    assert prediction.shape[0]== df_test['time'].shape[0]
    df_save = pd.DataFrame({'TIME': df_test['time'].apply(lambda x : x.date()), 'Pred': prediction})
    # # 1. Read old result file 
    # df_save_old = pd.read_csv("result.csv",sep=";") 
    # df_save_old.TIME = pd.to_datetime(df_save_old.TIME)
    # # 2. Keep the old prediction if there are two predictions made for the same date
    # df_save.TIME = pd.to_datetime(df_save.TIME)
    # df_save = pd.concat([df_save_old[df_save.columns], df_save], axis=0).drop_duplicates(subset=['TIME'],keep='first').reset_index(drop=True)
    # # 3. Add true values : y
    # df.TIME = pd.to_datetime(df.TIME)
    # df_save = df_save.merge(df, on='TIME', how='left')
    # # 4. Save to disk
    # df_save[['TIME','y','Pred']].sort_values(by=['TIME']).to_csv("result.csv", sep=';', index=False)
    print(prediction)
    print("***4")

    df_save[['TIME','Pred']].sort_values(by=['TIME']).to_csv("./ccf/result/result-mid-call.csv", sep=';', index=False)

def main_mid_case():
    """
    Inference of midterm case 
    """
    # now = datetime.datetime.now().date()
    now = datetime.datetime.strptime('2020-01-01', '%Y-%m-%d')   
    inputPath = "./data/NbcaseDay.csv"
    df, df_test, df_pred = inference(now, inputPath)
 
    # Load pretrained model and predict
    # first
    input_c_1 =   ['xMONTH', 'yMONTH', 'xDAY', 'yDAY', 'xWD','yWD', 'JF','AftJF','BefJF','HolidaysFR','DAY_LENGTH', 'MEAN_LAGGED','STD_LAGGED']
    model_path_1 = "./ccf/model_retraining_result/pkl/RF-midcase-123-100-17-0.5-2-7.47505.pkl"
    rg_1 = pickle.load(open(model_path_1, 'rb'))
    prediction_1= rg_1.predict(df_pred[input_c_1])
    # second
    input_c_2 =  ['HALF_HOURL', 'HALF_HOURC', 'xMONTH', 'yMONTH', 'xDAY', 'yDAY', 'xWD', 'yWD', 'TSE', 'JF','HolidaysFR', 'AftJF', 'BefJF',
                   'DAY_LENGTH', 'MG_1', 'MG_2', 'MG_3', 'WDG_1', 'WDG_2', 'WDG_3', 'WDG_4', 'MEAN_LAGGED','STD_LAGGED', 'LAGGED_LY_DW_avg']
    model_path_2 = "./ccf/model_retraining_result/pkl/RF-midcase-123-300-15-0.33-4-8.65204.pkl"
    rg_2 = pickle.load(open(model_path_2, 'rb'))
    prediction_2 = rg_2.predict(df_pred[input_c_2])
    # take average
    prediction = 1/2*prediction_1 + 1/2*prediction_2
    print("***3")


    # Save Result
    assert prediction.shape[0]== df_test['time'].shape[0]
    df_save = pd.DataFrame({'TIME': df_test['time'].apply(lambda x : x.date()), 'Pred': prediction})
    # # 1. Read old result file 
    # df_save_old = pd.read_csv("result.csv",sep=";") 
    # df_save_old.TIME = pd.to_datetime(df_save_old.TIME)
    # # 2. Keep the old prediction if there are two predictions made for the same date
    # df_save.TIME = pd.to_datetime(df_save.TIME)
    # df_save = pd.concat([df_save_old[df_save.columns], df_save], axis=0).drop_duplicates(subset=['TIME'],keep='first').reset_index(drop=True)
    # # 3. Add true values : y
    # df.TIME = pd.to_datetime(df.TIME)
    # df_save = df_save.merge(df, on='TIME', how='left')
    # # 4. Save to disk
    # df_save[['TIME','y','Pred']].sort_values(by=['TIME']).to_csv("result.csv", sep=';', index=False)
    print(prediction)
    print("***4")

    df_save[['TIME','Pred']].sort_values(by=['TIME']).to_csv("./ccf/result/result-mid-case.csv", sep=';', index=False)


if __name__ == "__main__":
    # main_mid_call()
    main_mid_case()