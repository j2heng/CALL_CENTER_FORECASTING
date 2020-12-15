#!/usr/bin/env python

import sys
import pandas as pd
import numpy as np
import datetime, time
from features.TimeFeatureBuilder import HourFeatureBuilding
import pickle

def inference(now, inputPath):
    # Get Data
    Ephemeride = pd.read_csv("./data/Ephemeride_Paris.csv", sep=';')
    df =  pd.read_csv(inputPath,sep=";") 
    # metter en commentaire ces deux lignes si prediction j a j
    df['time'] = pd.to_datetime(df['time']) 
    df = df.loc[df['time'] <now]
    print("***0")

    index = pd.date_range(now, now+datetime.timedelta(days=31), freq='H')
    df_test = pd.DataFrame({ df.columns[0]: index, df.columns[1]: np.nan })[df.columns].fillna(0)
    df_new = pd.concat([df, df_test], axis=0)[df.columns]
    df_new.reset_index(inplace=True, drop=True)
    df_new.columns = ['TIME', 'y']
    df = df_new.copy()
    print("***1")

    # Feature Engineering      
    df = HourFeatureBuilding(df,Ephemeride, '2012-01-01',(now+datetime.timedelta(days=31)).strftime("%Y-%m-%d"))
    df = df.transform()
    df_pred= df.loc[df['DATE'] >= now.strftime("%Y-%m-%d")]
    print("***2")

    return df, df_test, df_pred


####################################################       
##################  Main function  ####################
####################################################        

def main_short_call():
    """
    Inference of shortterm call 
    """
    # now = datetime.datetime.now().date()
    now = datetime.datetime.strptime('2020-01-01', '%Y-%m-%d')   
    inputPath = "./data/NbcallHour.csv"
    df, df_test, df_pred = inference(now, inputPath)
 
    # Load pretrained model and predict
    # first
    input_c_1 = "TODO"
    model_path_1 = "TODO"
    rg_1 = pickle.load(open(model_path_1, 'rb'))
    prediction_1 = rg_1.predict(df_pred[input_c_1])
    # second
    input_c_2 =  "TODO"
    model_path_2 = "TODO"
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

    df_save[['TIME','Pred']].sort_values(by=['TIME']).to_csv("./ccf/result/result-short-call.csv", sep=';', index=False)

def main_short_case():
    """
    Inference of shortterm case 
    """
    # now = datetime.datetime.now().date()
    now = datetime.datetime.strptime('2020-01-01', '%Y-%m-%d')   
    inputPath = "./data/NbcaseHour.csv"
    df, df_test, df_pred = inference(now, inputPath)

    df_pred.to_csv("./ccf/result/df_pred-short-case.csv", sep=';', index=False)
 
    # # Load pretrained model and predict
    # # first
    # input_c_1 = ['xMONTH', 'yMONTH', 'xDAY', 'yDAY', 'xWD', 'yWD', 'xHOUR', 'yHOUR',
    #         'TSE', 'JF', 'HolidaysFR', 'AftJF', 'BefJF', 'MEAN_LAGGED', 'STD_LAGGED', 'LAGGED_LY_DW_avg']
    # model_path_1 = "./ccf/model_retraining_result/pkl/RF-shortcase-123-100-13-0.33-3-22.33192.pkl"
    # rg_1 = pickle.load(open(model_path_1, 'rb'))
    # prediction_1= rg_1.predict(df_pred[input_c_1])
    # # second
    # input_c_2 =  ['xMONTH', 'yMONTH', 'xDAY', 'yDAY', 'xWD', 'yWD', 'xHOUR', 'yHOUR',
    #         'JF', 'HolidaysFR', 'AftJF', 'BefJF', 'MEAN_LAGGED', 'STD_LAGGED']
    # model_path_2 = "./ccf/model_retraining_result/pkl/RF-shortcase-123-200-13-0.5-2-22.24850.pkl"
    # rg_2 = pickle.load(open(model_path_2, 'rb'))
    # prediction_2 = rg_2.predict(df_pred[input_c_2])
    # # third
    # input_c_3 =  ['xMONTH', 'yMONTH', 'xDAY', 'yDAY', 'xWD', 'yWD', 'xHOUR', 'yHOUR',
    #         'JF', 'HolidaysFR', 'AftJF', 'BefJF', 'LAGGED_LY_DW_avg']
    # model_path_3 = "./ccf/model_retraining_result/pkl/RF-shortcase-123-100-9-0.5-3-22.90402.pkl"
    # rg_3 = pickle.load(open(model_path_3, 'rb'))
    # prediction_3 = rg_3.predict(df_pred[input_c_3])
    # # take average
    # prediction = 1/2*prediction_1 + 1/4*prediction_2 + 1/4*prediction_3
    # print("***3")


    # # Save Result
    # assert prediction.shape[0]== df_test['time'].shape[0]
    # df_save = pd.DataFrame({'TIME': df_test['time'].apply(lambda x : x.date()), 'Pred': prediction})
    # # # 1. Read old result file 
    # # df_save_old = pd.read_csv("result.csv",sep=";") 
    # # df_save_old.TIME = pd.to_datetime(df_save_old.TIME)
    # # # 2. Keep the old prediction if there are two predictions made for the same date
    # # df_save.TIME = pd.to_datetime(df_save.TIME)
    # # df_save = pd.concat([df_save_old[df_save.columns], df_save], axis=0).drop_duplicates(subset=['TIME'],keep='first').reset_index(drop=True)
    # # # 3. Add true values : y
    # # df.TIME = pd.to_datetime(df.TIME)
    # # df_save = df_save.merge(df, on='TIME', how='left')
    # # # 4. Save to disk
    # # df_save[['TIME','y','Pred']].sort_values(by=['TIME']).to_csv("result.csv", sep=';', index=False)
    # print(prediction)
    # print("***4")

    # df_save[['TIME','Pred']].sort_values(by=['TIME']).to_csv("./ccf/result/result-short-case.csv", sep=';', index=False)


if __name__ == "__main__":
    # main_short_call()
    main_short_case()