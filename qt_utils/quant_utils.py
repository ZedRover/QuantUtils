from pandas.core.window import rolling
from rs_utils.qrutils import log_return, corr, calc_rolling_corr
import pandas as pd
import numpy as np
from scipy import stats 
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, MinMaxScaler
from pyforecaster.papa.client import get_model_forecaster_by_name, login
from qrutils.future_utils import get_X_and_snapshot_baptized, get_all_predictors
from qrutils.future_utils import get_product_information,get_rolling_product,get_X_and_snapshot
pd.set_option('display.max_columns', None)
import datetime
from sfutils import signal_utils as sgu 
from sklearn import linear_model
from tqdm import trange
from sklearn.metrics import confusion_matrix,f1_score
import functools 
import seaborn as sns 
from scipy import stats 
import sys 
from sklearn.metrics import r2_score


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
        return np.log(1/(1-(info['open_ratio_by_money']+info['close_ratio_by_money'])*(1-info['rebate'])/2))
        # taking thres * tick size  
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

def exchange_generator(prod):
    prod = ''.join([i for i in prod if not i.isdigit()])
    if prod.upper() in ['I', "J", "JM", 'A', 'B', 'M', 'C', 'CS', 'V', 'EB', 'EG',
                        'PG', 'RR', 'L', 'BB', 'FB', 'PP', 'P', 'Y', 'M', 'LH', 'JD']:
        exchange = "dce"
    elif prod.upper() in ['CU', 'PB', 'AL', 'ZN', 'SN', 'NI', 'SS', "RB",
                          'HC', 'AG', 'AU', 'BU', 'FU', 'SP', 'SC', 'RU', 'WR','BC', 'NR', 
                          'LU']:
        exchange = "shfe"
    elif prod.upper() in ['IF', 'IC', 'IH', 'T', 'TF', 'TS']:                               
        exchange = "cffex"
    elif prod.upper() in ['TA', 'AP', 'CJ', 'MA', 'CJ', 'FG', 'SA', 'OI', 'ZC','CY', 'PF',
                          'CF', 'SM', 'PK', 'SR', 'RM', 'SF', 'UR']:
        exchange = "czce"                                                       
    else:
        raise ValueError('No exchange for product {}'.format(prod))
    return exchange

def plot_y_for_x(x,y,thres,direction = 'x-y'):
    """
    Parameters
    ----------
    x : pd.Series
        The signal 
    y : pd.Series 
        Log return
    thres: float 
        The taking threshold
    direction: str,optional,"x-y" or "y-x" 
        Plot x according to y or reverse 
        
    Returns
    -------
    None
    
        
    """
    if direction == 'y-x':
        
        upperb = 0.80
        lowerb = 0.20
        
        x_upper = x.quantile(upperb)
        x_lower = x.quantile(lowerb)
        

        up_idx = np.where(x>x_upper)[0]
        down_idx  = np.where(x<x_lower)[0]

        over_thres_id = np.where(x > thres)[0]
        down_thres_id = np.where(x < -thres)[0]
        fig,axs = plt.subplots(2,2,dpi = 100,figsize = (20,10))
        plt.suptitle('log return hist according to signal')
        y[list(set(up_idx))].plot(ax = axs[0,0],kind = 'hist',title = 'x> {} (pct)'.format(upperb),bins = 30,grid = 1)
        y[list(set(down_idx))].plot(ax = axs[0,1],kind = 'hist',title = 'x< {} (pct)'.format(lowerb),bins = 30,grid = 1)
        y[list(set(over_thres_id))].plot(ax = axs[1,0],kind = 'hist',title = 'x > thres',bins = 40,grid = 1)
        y[list(set(down_thres_id))].plot(ax = axs[1,1],kind = 'hist',title = 'x < -thres',bins = 40,grid = 1)
        plt.show()
        
    else:
        upperb = 0.90
        lowerb = 0.10
        y_up_99 = y.quantile(upperb)
        y_down_01 = y.quantile(lowerb)

        up_idx = np.where(y > y_up_99)[0]
        down_idx = np.where(y < y_down_01)[0]
    
        m_up_idx = np.where(y > thres)[0]
        m_down_idx = np.where(y < -thres)[0]
        print((x[list(set(up_idx))]))
        fig, axs = plt.subplots(2, 2, dpi=100, figsize=(20, 10))
        plt.suptitle('signal hist according to log return')
        x[list(set(up_idx))].plot(ax=axs[0, 0],
                                kind='hist', title='y>{} pct,skew: {:.4f}'.format(upperb,stats.skew(x[list(set(up_idx))])), bins=30, grid=1,)
        x[list(set(down_idx))].plot(ax=axs[0, 1],
                                    kind='hist', title='y<{} pct,skew: {:.4f}'.format(lowerb,stats.skew(x[list(set(down_idx))])), bins=30, grid=1)
        x[list(set(m_up_idx))].plot(ax=axs[1, 0], kind='hist',
                                    title='y > taking_thres ; skew: {:.4f}'.format(stats.skew(x[list(set(m_up_idx))])), bins=40, grid=1)
        x[list(set(m_down_idx))].plot(ax=axs[1, 1], kind='hist',
                                    title='y < -taking_thres ; skew: {:.4f}'.format(stats.skew(x[list(set(m_down_idx))])), bins=40, grid=1)

        plt.show()






def calc_rolling_score(time,log_r,signals,rolling_window,nda = None):
    """
    Parameters
    ----------
    time: pd.DatetimeIndex,dtype = 'datetime64[ns]'
        The index of snapshot dataframe, instead of the 'machine_timestamp'.
    log_r: pd.Series, np.array
        The log return.
    signals: pd.Series, np.array
        The signal or forecaster.
    rolling_window: int
        Calculate n days once for ic and others.
    nda: str,None or 'nda' ,optional
        Split one day into 3 time zones.

    Returns
    -------
    [DataFrame , int]
        The dataframe contains ic, skew and mean of the log return when signal is in extreme condition.
        The int variable is the weighted sum of all evaluators.
    
    """
    if len(signals.shape)==2:
        columns = list(range(signals.shape[1]))
    else:
        columns = [0]
    if hasattr(log_r, 'values'):
        log_r = log_r.values
    if hasattr(signals, 'columns'):
        columns = signals.columns
    if hasattr(signals, 'values'):
        signals = signals.values
        
    log_r = pd.Series(log_r, index=pd.to_datetime(time))
    signals = pd.DataFrame(signals, index=pd.to_datetime(time))
    trading_days = pd.unique(log_r.index.date)
    result_dic = {}
    
    for i in list(range(len(trading_days) - rolling_window +1)):
        k = _calc_score(i, trading_days, rolling_window, log_r, signals,nda)
        result_dic[k[0]] = k[1]
        
    rolling_score = pd.DataFrame(result_dic).T
    if nda==None:
        rolling_score.columns = ['mean +','mean -','skew +','skew -','ic +','ic -','r2 +','r2 -']
        rolling_score['skew']=rolling_score['skew +']-rolling_score['skew -']
        rolling_score['ic'] = (rolling_score['ic +']+rolling_score['ic -'])/2
        rolling_score['r2'] = (rolling_score['r2 +']+rolling_score['r2 -'])/2
    else:
        col = []
        for x in nda:
            col.extend([x+'_mean +',x+'_mean -',x+'_skew +',x+'_skew -',x+'_ic +',x+'_ic -',
                        x+'_r2 +',x+'_r2 -'])
        rolling_score.columns = col
        for x in nda:
            rolling_score['skew']=rolling_score[x+'_skew +']-rolling_score[x+'_skew -']
            rolling_score['ic'] = (rolling_score[x+'_ic +']+rolling_score[x+ '_ic -'])/2
            rolling_score['r2'] = (rolling_score[x+'_r2 +']+rolling_score[x+ '_r2 -'])/2
        
    return rolling_score ,rolling_score.apply(_calc_result_score,axis=1)

def _calc_result_score(df):
    cols = df.index 
    res = 0
    for col in cols:
        if col[-1] == '+':
            if "mean" in col:
                res+=df[col]*1e3
            else:
                res+=df[col]
        else:
            if 'ic' in col:
                res+=df[col]
            if "mean" in col:
                res-=df[col]*1e3
            else:
                res-=df[col]    

    return res 

def _calc_score(i, trading_days, rolling_window, log_r, signal,nda = None):
    result = []
    logr_i = log_r.loc[str(trading_days[i]):str(trading_days[i + rolling_window-1])]
    signals_i = signal.loc[str(trading_days[i]):str(trading_days[i + rolling_window-1])]
    thres = get_oc_threshold('i1')
    if nda == None:
        signals_i = signals_i.reset_index(drop=True).values.flatten()
        logr_i    = logr_i.reset_index(drop=True).values.flatten()
        m_logr_u = sgu.xy_finder(signals_i,logr_i,thres,reverse = False)
        m_logr_d = sgu.xy_finder(signals_i,logr_i,-thres,reverse = False)
        ic =[corr(signals_i[np.where(signals_i>thres)[0]],m_logr_u),\
             corr(signals_i[np.where(signals_i<-thres)[0]],m_logr_d)]
        
        result.extend([np.mean(m_logr_u),np.mean(m_logr_d),stats.skew(m_logr_u),stats.skew(m_logr_d)])
        result.extend(ic)
        result.extend([r2_score(m_logr_u,signals_i[np.where(signals_i>thres)[0]]),r2_score(m_logr_d,signals_i[np.where(signals_i<-thres)[0]])])
        
    else:
        # plt.plot(logr_i.index)
        # sys.exit()
        n_id = signals_i[(signals_i.index < pd.to_datetime(str(trading_days[i])+" 23:00")) | \
            (signals_i.index > pd.to_datetime(str(trading_days[i])+" 21:00"))].index

        d_id = signals_i[(signals_i.index < pd.to_datetime(str(trading_days[i])+" 12:00")) & \
            (signals_i.index > pd.to_datetime(str(trading_days[i])+" 09:00"))].index
        
        a_id = signals_i[(signals_i.index < pd.to_datetime(str(trading_days[i])+" 15:00")) & \
            (signals_i.index > pd.to_datetime(str(trading_days[i])+" 13:30"))].index

        for x in nda:
            id = eval(x+'_id')
            m_logr_u = sgu.xy_finder(signals_i.loc[id,:].values.flatten(),\
                logr_i[id].values.flatten(),thres)
            m_logr_d = sgu.xy_finder(signals_i.loc[id,:].values.flatten(),logr_i[id].values.flatten(),-thres)
            thres_per = stats.percentileofscore(signals_i,thres)
            thres_per2 = stats.percentileofscore(signals_i,thres)
            
            # ic = [corr(x = signals_i.loc[id,:].to_numpy(), y = logr_i[id].to_numpy() ,x_percent = [thres_per,thres_per2])]
            x = signals_i.loc[id,:].values.flatten()
            x1 = x[np.where(x>thres)[0]]
            x2 = x[np.where(x<-thres)[0]]
            y1 = m_logr_u.flatten()
            y2 = m_logr_d.flatten()
            if (len(x)==0)|(len(x1)==0):
                ic = [np.nan,np.nan]
            else:
                ic = [corr(x1,y1),corr(x2,y2)]
            result.extend([np.mean(m_logr_u),np.mean(m_logr_d),stats.skew(m_logr_u),stats.skew(m_logr_d)])
            result.extend(ic)
            result.extend([r2_score(y1,x1),r2_score(y2,x2)])
            
    return str(trading_days[i + rolling_window-1]), result 



   


def signal_confusion_matrix(time,x,y,thres,horizon = 1):
    """
    Parameters
    ----------
        time: df['machine_timestamp']
        x: signal
        y: log return 
        thres: [making_thres,taking_thres]
    """
    t = [-np.inf, -thres[1],thres[0],0,thres[0],thres[1],np.inf]
    t = [-np.inf, -thres[1],0,thres[1],np.inf]
    data = pd.concat([pd.Series(x),pd.Series(y)],axis=1).dropna(how = 'any',axis=0)
    reg = linear_model.LinearRegression()
    reg.fit(data.iloc[:,0].values.reshape(-1,1),data.iloc[:,1].values.reshape(-1,1))
    # assert np.abs(reg.intercept_) < 1e-4,'wrong model'
    a = reg.coef_[0]
    a =1 
    y = pd.Series(y, index=pd.to_datetime(time))
    x = pd.DataFrame(x, index=pd.to_datetime(time))
    rolling_window=1
    trading_days = pd.unique(pd.to_datetime(time.values).date)
    results = []
    scores  = []
    for i in trange(len(trading_days)):    
        yi = y.loc[str(trading_days[i]):str(trading_days[i + rolling_window-1])].reset_index(drop=1).values.flatten()
        xi = x.loc[str(trading_days[i]):str(trading_days[i + rolling_window-1])].reset_index(drop=1).values.flatten()
        # days
        assert len(xi)==len(yi)
        resulti,scorei = cts_confusion_matrix(xi,yi,a,thres,_clf_func)         
        results.append(resulti)
        scores.append(scorei)
    return results ,scores 


def cts_confusion_matrix(x,y,a,thres,_clf_func):
    if len(x.shape)==2:
        columns = list(range(x.shape[1]))
    else:
        columns = [0]
    if hasattr(y, 'values'):
        y = y.values.flatten()
    if hasattr(x, 'columns'):
        columns = x.columns
    if hasattr(x, 'values'):
        x = x.values.flatten()
    # xc = list(map(_clf_func,x*a,thres))
    # yc = list(map(_clf_func,y,thres))
    _clf_func_thres = functools.partial(_clf_func,thres = thres)
    a = float(a)
    xc = np.frompyfunc(_clf_func_thres,1,1)(x*a)
    yc = np.frompyfunc(_clf_func_thres,1,1)(y)
    xc = pd.Series(xc,dtype = 'int64')
    yc = pd.Series(yc,dtype = 'int64')
    result = confusion_matrix(yc,xc)
    score = f1_score(yc,xc,average = 'micro')
    return result,score 


def _clf_func(x,thres):
    thres *=1
    assert len(thres)>1,'wrong thres'
    if x<-thres[1]:
        return 0 
    elif x<thres[1]:
        return 1 
    else:
        return 2