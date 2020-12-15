import pandas as pd
import numpy as np
import datetime, time
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt


inputPath = "./ccf/result/result-mid-case-final2.csv"
t = datetime.datetime.strptime('2020-04-01', '%Y-%m-%d') 

def symmetric_mean_absolute_percentage_error(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F))) 

def getErrors(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse =  sqrt(mean_squared_error(y_true, y_pred))
    smape = symmetric_mean_absolute_percentage_error(y_true, y_pred)
    width = [15, 10]
    print("***")
    print(
        f"\n{'SMAPE:':<{width[0]}}{smape:>{width[1]}}",
        f"\n{'MAE:':<{width[0]}}{mae:>{width[1]}}",    
        f"\n{'RMSE:':<{width[0]}}{rmse:>{width[1]}}",
    )
    return 1

def main():
    df =  pd.read_csv(inputPath,sep=";") 
    df.TIME = pd.to_datetime(df.TIME)
    df = df.loc[df['TIME'] < t]
    df.YEAR = df['TIME'].apply(lambda x : x.year)
    df.MONTH = df['TIME'].apply(lambda x : x.month)

    for y in set(df.YEAR):
        for m in set(df.MONTH):
            print("\n *** Year: %d, Month: %d ***" %(y,m))
            tmp = df.loc[(df.YEAR==y) & (df.MONTH==m)]
            print(" => y_pred")
            getErrors(tmp.y_true, tmp.y_pred)
            print("=> y_pred_lake")
            getErrors(tmp.y_true, tmp.y_pred_lake)

if __name__ == "__main__":
    main()