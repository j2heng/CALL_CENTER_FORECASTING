import sys
import pandas as pd
import numpy as np
import datetime, time
from features.TimeFeatureBuilder import DayFeatureBuilding
from models.train_model import MidTermModel
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import pickle



def model_retraining():

    t_max = datetime.datetime.strptime('2020-01-01', '%Y-%m-%d') 
    t_split = datetime.datetime.strptime('2019-01-01', '%Y-%m-%d') 

    def symmetric_mean_absolute_percentage_error(A, F):
        return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F))) 
    
    # Get Data
    Ephemeride = pd.read_csv("./data/Ephemeride_Paris.csv", sep=';')
    df =  pd.read_csv("./data/NbcallDay.csv",sep=";") 
    df['time'] = pd.to_datetime(df['time']) 
    df = df.loc[df['time'] < t_max]
    df.columns = ['TIME', 'Nb_appels_recus']

    # Feature 
    df = DayFeatureBuilding(df, Ephemeride, '2012-01-01', t_max.strftime("%Y-%m-%d"))
    df = df.transform()



    # Train
    rg = MidTermModel(df,t_split)
    best_model_ = rg.fit()
    y_pred = rg.predict()
    df_pred = pd.DataFrame({
        'DATE':     df[df['DATE'] >=t_split]['DATE'],
        'YEAR':     df[df['DATE'] >=t_split]['YEAR'],
        'MONTH':    df[df['DATE'] >=t_split]['MONTH'],
        'DAY':      df[df['DATE'] >=t_split]['DAY'],
        'WEEK_DAY': df[df['DATE'] >=t_split]['WEEK_DAY'],
        'y_pred':   y_pred,
        'y_true':   df[df['DATE'] >=t_split]['Nb_appels_recus'],
        })
    df_pred = df_pred.loc[df_pred.YEAR==2019]

    g_mae = mean_absolute_error(df_pred.y_true, df_pred.y_pred)
    g_rmse =  sqrt(mean_squared_error(df_pred.y_true, df_pred.y_pred))
    g_smape = symmetric_mean_absolute_percentage_error(df_pred.y_true, df_pred.y_pred)
    width = [15, 10]
    print("***** General Performance *****")
    print(
        f"\n{'SMAPE:':<{width[0]}}{g_smape:>{width[1]}}",
        f"\n{'MAE:':<{width[0]}}{g_mae:>{width[1]}}",    
        f"\n{'RMSE:':<{width[0]}}{g_rmse:>{width[1]}}",
    )

    for y in set(df_pred.YEAR):
        for m in set(df_pred.MONTH):
            print("*** Year: %d, Month: %d ***" %(y,m))
            tmp = df_pred.loc[(df_pred.YEAR==y) & (df_pred.MONTH==m)]
            mae = mean_absolute_error(tmp.y_true, tmp.y_pred)
            rmse = sqrt(mean_squared_error(tmp.y_true, tmp.y_pred))
            smape = symmetric_mean_absolute_percentage_error(tmp.y_true, tmp.y_pred)
            print(
                f"\n{'SMAPE:':<{width[0]}}{smape:>{width[1]}}",
                f"\n{'MAE:':<{width[0]}}{mae:>{width[1]}}",    
                f"\n{'RMSE:':<{width[0]}}{rmse:>{width[1]}}",
            )

    # Save to disk
    n_estimator = best_model_.get_params()['n_estimators']
    n_depth = best_model_.get_params()['max_depth']
    n_feature =  best_model_.get_params()['max_features']
    n_samples_leaf =  best_model_.get_params()['min_samples_leaf'] 
    rand_state = best_model_.get_params()['random_state'] 

    model_path = "./midterm_appel/model_retraining_result/pkl/RF-{}-{}-{}-{}-{}-{:.5f}.pkl".format(rand_state,n_estimator,n_depth,n_feature,n_samples_leaf,g_smape)
    file_path = "./midterm_appel/model_retraining_result/csv/result-RF-{}-{}-{}-{}-{}-{:.5f}.csv".format(rand_state,n_estimator,n_depth,n_feature,n_samples_leaf,g_smape)
    
    pickle.dump(best_model_, open(model_path, 'wb'))
    df_pred.sort_values(by=['DATE']).to_csv(file_path, sep=';', index=False)




# if __name__ == "__main__":
#     model_retraining()