from qrutils import log_return, corr, calc_rolling_corr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, MinMaxScaler
from pyforecaster.papa.client import get_model_forecaster_by_name, login
from qrutils.future_utils import get_X_and_snapshot_baptized, get_all_predictors
from qrutils.future_utils import get_product_information,get_rolling_product,get_X_and_snapshot
pd.set_option('display.max_columns', None)
import datetime


def get_model_forecaster(prod='i0001',
                         bgn_date=20210901,
                         end_date=20210921):

    login('wangyu', 'password')
    # prod = '_XBTUSD'#or i0001  or i1
    # model = '_XBTUSD_making_20210818_0_2'#like   'i1_20210821_1_mid_cindy'
    model = 'i1_20210821_1_mid_cindy'
    forecaster = get_model_forecaster_by_name(model)
    # df_all = get_X_and_snapshot_baptized(prod = prod,start_date = bgn_date,end_date = end_date,
    #                                     extra_predictors=[forecaster],venue = 'btc_hk.m_md')
    df_all = get_X_and_snapshot_baptized(prod=prod, start_date=bgn_date, end_date=end_date,
                                         extra_predictors=[forecaster], venue='dce_golden.m')
    return df_all





def get_future_info(prod = 'i1', date = None):
    if date == None:
        date = int((datetime.date.today()-datetime.timedelta(days=1)).strftime("%Y%m%d"))


    return get_product_information(get_rolling_product(prod,date = date))



def get_oc_threshold(prod = 'i1',date = None):
    info = get_future_info(prod,date)
    if info['close_ratio_by_volume']==0:
        return (info['open_ratio_by_money']+info['close_ratio_by_money'])*(1-info['rebate'])/2

    else:
        print('volume threshold to be defined')


def get_X(prod = 'i1',start_date = 20210901,end_date = 20210902):
    login('yu.wang','123')
    return get_X_and_snapshot(prod,start_date,end_date,extra_predictors=get_all_predictors(prod),must_calculate = True,disable_old_batch = True)



def get_ic_value(signal,price,time,horizon = 2):
    logr = -log_return(time,price, -horizon*10**9)


    return corr(logr,signal)

def get_rolling_ic_value(signal,price,time,horizon = 2):
    logr = -log_return(time,price, -horizon*10**9)
    return calc_rolling_corr(time,logr,signal,rolling_window =2 )

    


def get_sig_info(df_all,horizon = 2,rolling = 1):
    
    fig,axs = plt.subplots(2,2,dpi = 100,figsize = (12,10))
    
    x = df_all.iloc[:,-1]
    y = pd.Series(-log_return(df_all.machine_timestamp,df_all.mid_price,-horizon*10**9))

    print('IC ({:}s):{:}'.format(horizon,corr(x,y)))
    print('STATS OF SIG\n',x.describe())
    print('***************************')
    print('STATS OF LOG-RETURN\n',y.describe())
    
    rolling_ic = calc_rolling_corr(df_all.machine_timestamp,y,x,rolling)
    plt.xticks(rotation = 90)
    rolling_ic.plot(ax = axs[0,0],title = 'rolling IC')
    
    x.plot(kind = 'hist',ax = axs[0,1],color = 'b',alpha = 0.5,bins = 50,label = 'sig')
    y.plot(kind = 'hist',ax = axs[1,0],color = 'r',alpha = 0.5,bins = 50,label  = 'log_return')
    
#     xz = MinMaxScaler().fit_transform(x.values.reshape(-1,1))
#     yz = MinMaxScaler().fit_transform(y.values.reshape(-1,1))
    xz = scale(x)
    yz = scale(y)
    # pd.Series(xz.reshape(1,-1)[0]).plot(kind = 'hist',ax = axs[1,1],color = 'b',alpha = 0.5,bins = 30,label = 'sig',xlim = [-0.01,0.01])
    # pd.Series(yz.reshape(1,-1)[0]).plot(kind = 'hist',ax = axs[1,1],color = 'r',alpha = 0.5,bins = 30,label = 'log_return',xlim = [-0.01,0.01])
    x.plot(kind = 'hist',ax = axs[1,1],color = 'b',alpha = 0.5,bins = 100,label = 'sig',xlim = [-0.002,0.002])
    y.plot(kind = 'hist',ax = axs[1,1],color = 'r',alpha = 0.5,bins = 100,label = 'log_return',xlim = [-0.002,0.002],rot = 45)         
    plt.legend()
    plt.show()
    
    return x,y,rolling_ic
    
    
def plot_y_for_x(x,y,thres,direction = 'x-y'):
    if direction == 'y-x':
        x_up_99 = x.quantile(0.99)
        x_down_01 = x.quantile(0.01)

        print(x_up_99)
        print(x_down_01)

        up_idx = np.where(x>x_up_99)[0]
        down_idx  = np.where(x<x_down_01)[0]

        m_up_idx = list(set(np.where(x> x.quantile(0.5))[0]) & set(np.where(x<x.quantile(0.51))[0]))

        m_down_idx = list(set(np.where(x< x.quantile(0.5))[0]) & set(np.where(x> x.quantile(0.49))[0]))

        fig,axs = plt.subplots(2,2,dpi = 100,figsize = (20,10))
        y[list(set(up_idx))].plot(ax = axs[0,0],kind = 'hist',title = 'x>0.99',bins = 30,grid = 1)
        y[list(set(down_idx))].plot(ax = axs[0,1],kind = 'hist',title = 'x<0.01',bins = 30,grid = 1)
        y[list(set(m_up_idx))].plot(ax = axs[1,0],kind = 'hist',title = '0.50<x<0.51',bins = 40,grid = 1)
        y[list(set(m_down_idx))].plot(ax = axs[1,1],kind = 'hist',title = '0.49<x<0.50',bins = 40,grid = 1)
        plt.show()
    else:
        y_up_99 = y.quantile(0.99)
        y_down_01 = y.quantile(0.01)

        print(y_up_99)
        print(y_down_01)

        up_idx = np.where(y > y_up_99)[0]
        down_idx = np.where(y < y_down_01)[0]
        
        
        
        
        m_up_idx = list(set(np.where(y > thres)[0]))
        m_down_idx = list(set(np.where(y < -thres)[0]))


        fig, axs = plt.subplots(2, 2, dpi=100, figsize=(20, 10))
        x[list(set(up_idx))].plot(ax=axs[0, 0],
                                kind='hist', title='y>0.99', bins=30, grid=1,)
        x[list(set(down_idx))].plot(ax=axs[0, 1],
                                    kind='hist', title='y<0.01', bins=30, grid=1)
        
        x[list(set(m_up_idx))].plot(ax=axs[1, 0], kind='hist',
                                    title='y> taking_thres', bins=40, grid=1)
        x[list(set(m_down_idx))].plot(ax=axs[1, 1], kind='hist',
                                    title='y< taking_thres', bins=40, grid=1)
        plt.show()





