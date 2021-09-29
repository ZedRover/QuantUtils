import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale,MinMaxScaler
from pyforecaster.papa.client import get_model_forecaster_by_name,login
from qrutils.future_utils import get_X_and_snapshot_baptized
pd.set_option('display.max_columns',None)
from qrutils import log_return ,corr,calc_rolling_corr 
from numba import jit



def replace_outliers(df):
    outlier_indices = []

    # 1st quartile (25%)
    Q1 = np.percentile(df, 25)
    # 3rd quartile (75%)
    Q3 = np.percentile(df, 75)
    # Interquartile range (IQR)
    IQR = Q3 - Q1
    # outlier step
    outlier_step = 1.5 * IQR
    for nu in range(len(df)):
        if (df[nu] < Q1 - outlier_step):
            df[nu] = Q1-outlier_step
        elif df[nu] > Q3 +outlier_step:
            df[nu] = Q3+outlier_step
    return df


def xy_finder(x,y,thres,reverse = False):
    if reverse == False:
        if type(thres)!= list:
            if thres>0:
                idx = np.where(x>thres)[0]
                return y[idx]
            else:
                idx = np.where(x<thres)[0]
                return y[idx]

        elif len(thres)==2:
            idx = list(set(np.where(thres[0]<x)) & set(np.where(x<thres[1])))
            return y[idx]   
    else:
        if type(thres)!= list:
            if thres>0:
                idx = np.where(y>thres)[0]
                return x[idx]
            else:
                idx = np.where(y<thres)[0]
                return x[idx]

        elif len(thres)==2:
            idx = list(set(np.where(thres[0]<y)) & set(np.where(y<thres[1])))
            return x[idx]   

        