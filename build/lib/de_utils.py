'''
Author: Samuel Wang
Date: 2021-11-10 22:00:51
Description: 
'''
import pandas as pd 
import numpy as np 
import qrutils.stock_utils as su 
import glob 
import dill as pickle
import warnings

from pyforecaster.papa.client import login
import forecaster_utils as fu
from forecaster_meta import BaseForecaster
from pyforecaster.papa.client import get_baptized_id
import luffy2 as lf
from qrutils.stock_utils import log_return
warnings.filterwarnings('ignore')

def get_valid_dataset(stockId,version,horizon=60,sampler=3,latency=4,start_date=20211025,end_date=20211109):
    df = su.get_X_and_snapshot(stockId,start_date,end_date,version = version,sampler = 3,latency = 4) 
    Y = su.log_return(df.machine_timestamp,df.mid_price,horizon*10**9,1)
    Y = pd.Series(Y,name = 'log_r',index = df.index)
    assert len(Y)==len(df)
    dataset = df.filter(regex = 'raw')
    dataset['log_r'] = Y
    return dataset

def save_de_model(model,address):
    output_hal = open(address, 'wb')
    str = pickle.dumps(model)
    output_hal.write(str)
    output_hal.close()

def load_de_model(empty_model,address):
    model = empty_model
    with open(address,'rb') as file:
        model  = pickle.loads(file.read())
    return model 



def get_old_data_loc(stockId='300128',start_date=2021031109,end_date=2021102509,horizon=60,vol = False):  
    if len(str(start_date))!=10:
        print('need set hour')
        
    login('wangyu', '20010923')
    forecaster_file = '/mnt/hdd_storage_2/stock_models_lgb_tmp/20211031/v{}@20211031@{}.forecaster'.format(stockId,horizon)
    fu.load_exports()
    forecaster = fu.load_from_file(forecaster_file)
    predictor_set = [BaseForecaster.parse_source_config(i) for i in forecaster.to_dict()['params']['slots']]
    bp_ids = get_baptized_id(predictor_set)

    df = lf.get_data_local(f'stock/{stockId}.lcsv',start_date,end_date,'/mnt/t9_data/lcsv')
    
    Y = log_return(df.machine_timestamp,df.mid_price,horizon*10**9,error_pct=1)
    sig = ['raw_'+i for i in bp_ids] 
    sig = [i for i in sig if i in df.columns]
    sig = df[sig]
    sig['Y']=Y
    sig.index = pd.to_datetime(df['machine_timestamp'])+pd.Timedelta(hours=8)
    if vol==True:
        vol = df.raw_target_with_first_10.values
        return sig,vol
    return sig

def get_with_vol(stockId='300128',start_date=2021031109,end_date=2021102509,horizon=60):
    login('wangyu', '20010923')
    forecaster_file = '/mnt/hdd_storage_2/stock_models_lgb_tmp/20211031/v{}@20211031@{}.forecaster'.format(stockId,horizon)
    fu.load_exports()
    forecaster = fu.load_from_file(forecaster_file)
    predictor_set = [BaseForecaster.parse_source_config(i) for i in forecaster.to_dict()['params']['slots']]
    bp_ids = get_baptized_id(predictor_set)
    df = lf.get_data_local(f'stock/{stockId}.lcsv',start_date,end_date,'/mnt/data/lcsv')
    Y = log_return(df.machine_timestamp,df.mid_price,horizon*10**9,error_pct=1)
    df['Y']= Y
    
    return df
    
    
    
def get_market_status(df,col,thres = [0.001810,0.005506]):
    res = [[]]*3
    for date,data in df.groupby(df.index.date):
        
        if df[col].mean() > thres[1]:
            print(date,df[col].mean(),'high')
            res[0].append(data)
        elif df[col].mean() < thres[0]:
            res[2].append(data)
            print(date,df[col].mean(),'low')
        else:
            # print(date,df[col].mean(),'mid')
            res[1].append(data)
    resdf = [ pd.concat(res[i]) for i in range(3)]
    return resdf


def get_stats(stockIds,version = '60'):
    files = glob.glob('/mnt/hdd_storage_2/stock_models_lgb_tmp/20211031/*.csv')  
    statistic = []
    for file in files:
        temp = pd.read_csv(file,dtype = {'stockId':str})
        if 'model_file' in temp.columns:
            statistic.append(temp)

    statistic = pd.concat(statistic,sort = False)
    statistic = statistic.reset_index(drop = True)
    stats = statistic[statistic.stockId.isin(stockIds)]
    mask = [True if i[-13:-11] ==version else False for i in stats.forecaster_file]
    
    return stats[mask]

def get_stock_pool():

    files = glob.glob('/mnt/hdd_storage_2/stock_models_lgb_tmp/20211031/*.csv')  
    statistic = []
    for file in files:
        temp = pd.read_csv(file,dtype = {'stockId':str})
        if 'model_file' in temp.columns:
            statistic.append(temp)

    statistic = pd.concat(statistic,sort = False)
    statistic = statistic.reset_index(drop = True)

    universe = pd.read_csv('/mnt/hdd_storage_2/yanhui/stock_universe/universe_20211102.csv',dtype = {0:str}, header = None)
    stock_list = universe[0].to_list()

    statistic['in_universe'] = statistic.apply(lambda x: True if x.stockId in stock_list else False,axis = 1)                                                                                  
    # universe -> sz       
    statistic['version'] = statistic.apply(lambda x: x.model_file.split('/')[-1].split('.')[0],axis = 1)
    statistic['horizon'] = statistic.apply(lambda x: int(x.version.split('@')[-1]),axis = 1)
                        
    stock_pool = statistic[(statistic.in_universe == True) & (statistic.horizon == 60)].copy().reset_index()
    
    return stock_pool



