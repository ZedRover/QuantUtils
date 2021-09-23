import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale,MinMaxScaler
from pyforecaster.papa.client import get_model_forecaster_by_name,login
from qrutils.future_utils import get_X_and_snapshot_baptized
pd.set_option('display.max_columns',None)
from qrutils import log_return ,corr,calc_rolling_corr 




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
