
from tqdm import trange
import numpy as np 
import pandas as pd 
from numba import jit
from pandas.core.frame import DataFrame 
from tqdm import tqdm ,trange
import talib   
import numpy as np
from utils import replace_outliers
import warnings

def zero_divide(x, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = np.divide(x,y)
    if hasattr(y, "__len__"):
        res[y == 0] = 0
    elif y == 0:
        res = 0
        
    return res


def get_all_signals(df,t,clean = False):
    m = df.shape[1]
    if clean:
        sigs1 = pd.DataFrame(signal_gen1(df)).reset_index(drop = True).apply(replace_outliers,axis=1)
        df['bs'] = (df['bv1']-df['sv1'])/(df['bv1']+df['sv1']).apply(replace_outliers,axis=1)
        sigs2 = signal_talib(df,t).reset_index(drop = True).apply(replace_outliers,axis=1)
    else:
        sigs1 = pd.DataFrame(signal_gen1(df)).reset_index(drop = True)
        df['bs'] = (df['bv1']-df['sv1'])/(df['bv1']+df['sv1'])
        sigs2 = signal_talib(df,t).reset_index(drop = True)
    df_sig = pd.concat([df.reset_index(drop = True),sigs1,sigs2],axis=1)
    return df_sig


def get_my_signals(df,t = 10):
    sig1 = signal_VOI(df,t)
    sig2 = signal_VOI2(df,t)
    sig3 = signal_OIR(df,t)
    sig4 = signal_MPB(df)
    sigs = pd.DataFrame({'VOI':sig1,'VOI2':sig2,'OIR':sig3,'MPB':'sig4'})
    return sigs



# #@jit()
def signal_gen1(dfSet):
    #Features representation
    ##Basic Set
    ###V1: price and volume (10 levels)
    featV1 = dfSet[['sp1', 'sp2', 'sp3', 'sp4', 'sp5','sv1', 'sv2', 'sv3', 'sv4', 'sv5','bp1',
       'bp2', 'bp3', 'bp4', 'bp5',  'bv1',
       'bv2', 'bv3', 'bv4', 'bv5', ]]
    featV1 = np.array(featV1)

    ##Time-insensitive Set
    ###V2: bid-ask spread and mid-prices
    temp1 = featV1[:,0:5] - featV1[:,10:15]
    temp2 = (featV1[:,0:5] + featV1[:,10:15])*0.5
    featV2 = np.zeros([temp1.shape[0],temp1.shape[1]+temp2.shape[1]])
    featV2[:,0:temp1.shape[1]] = temp1
    featV2[:,temp1.shape[1]:] = temp2

    ###V3: price differences
    temp1 = featV1[:,4] - featV1[:,0]
    temp2 = featV1[:,10] - featV1[:,14]
    temp3 = abs(featV1[:,1:5] - featV1[:,0:4])
    temp4 = abs(featV1[:,11:15] - featV1[:,10:14])
    featV3 = np.zeros([temp1.shape[0],1+1+temp3.shape[1]+temp4.shape[1]])
    featV3[:,0] = temp1
    featV3[:,1] = temp2
    featV3[:,2:2+temp3.shape[1]] = temp3
    featV3[:,2+temp3.shape[1]:] = temp4

    ###V4: mean prices and volumns
    temp1 = np.mean(featV1[:,0:5],1)
    temp2 = np.mean(featV1[:,10:15],1)
    temp3 = np.mean(featV1[:,5:10],1)
    temp4 = np.mean(featV1[:,15:],1)
    featV4 = np.zeros([temp1.shape[0],1+1+1+1])
    featV4[:,0] = temp1
    featV4[:,1] = temp2
    featV4[:,2] = temp3
    featV4[:,3] = temp4

    ###V5: accumulated differences
    temp1 = np.sum(featV2[:,0:5],1)
    temp2 = np.sum(featV1[:,5:10] - featV1[:,15:],1)
    featV5 = np.zeros([temp1.shape[0],1+1])
    featV5[:,0] = temp1
    featV5[:,1] = temp2

    ##Time-insensitive Set
    ###V6: price and volume derivatives
    temp1 = featV1[1:,0:5] - featV1[:-1,0:5]
    temp2 = featV1[1:,10:15] - featV1[:-1,10:15]
    temp3 = featV1[1:,5:10] - featV1[:-1,5:10]
    temp4 = featV1[1:,15:] - featV1[:-1,15:]
    featV6 = np.zeros([temp1.shape[0]+1,temp1.shape[1]+temp2.shape[1]+temp3.shape[1]+temp4.shape[1]]) #由于差分，少掉一个数据，此处补回
    featV6[1:,0:temp1.shape[1]] = temp1
    featV6[1:,temp1.shape[1]:temp1.shape[1]+temp2.shape[1]] = temp2
    featV6[1:,temp1.shape[1]+temp2.shape[1]:temp1.shape[1]+temp2.shape[1]+temp3.shape[1]] = temp3
    featV6[1:,temp1.shape[1]+temp2.shape[1]+temp3.shape[1]:] = temp4

    ##combining the features
    feat = np.zeros([featV1.shape[0],sum([featV1.shape[1],featV2.shape[1],featV3.shape[1],featV4.shape[1],featV5.shape[1],featV6.shape[1]])])
    feat[:,:featV1.shape[1]] = featV1
    feat[:,featV1.shape[1]:featV1.shape[1]+featV2.shape[1]] = featV2
    feat[:,featV1.shape[1]+featV2.shape[1]:featV1.shape[1]+featV2.shape[1]+featV3.shape[1]] = featV3
    feat[:,featV1.shape[1]+featV2.shape[1]+featV3.shape[1]:featV1.shape[1]+featV2.shape[1]+featV3.shape[1]+featV4.shape[1]] = featV4
    feat[:,featV1.shape[1]+featV2.shape[1]+featV3.shape[1]+featV4.shape[1]:featV1.shape[1]+featV2.shape[1]+featV3.shape[1]+featV4.shape[1]+featV5.shape[1]] = featV5
    feat[:,featV1.shape[1]+featV2.shape[1]+featV3.shape[1]+featV4.shape[1]+featV5.shape[1]:] = featV6


    return feat

#@jit()
def signal_VOI(df,t = 10):
    """
    author: vvy
    basic voi signal
    """

    id_bp = 5
    id_bv = 15
    id_sp = 10
    id_sv = 20
    voi = [0]*len(df)
    dfv = df.values
    for i in range(t,dfv.shape[0]):
        if dfv[i,5]>dfv[i-t,5]:
            dvb = dfv[i,15]
        elif dfv[i,5]==dfv[i-t,5]:
            dvb = dfv[i,15]-dfv[i-t,15]
        else:
            dvb = 0 

        if dfv[i,10]>dfv[i-t,10]:
            dva = 0
        elif dfv[i,5]==dfv[i-t,5]:
            dva = dfv[i,10]-dfv[i-t,10]
        else:
            dva = dfv[i,20]
        voi[i] = (dvb-dva)
    return np.array(voi)
#@jit()
def signal_VOI2(df,t):
    voi2=[0]*len(df)
    dfv = df.values
    wvtb=[0]*len(df)
    wvts=[0]*len(df)
    for i in range(t,dfv.shape[0]):
        wvtb[i]=np.sum(np.array([(1-(j/5))*dfv[i,j+15] for j in range(5)]))/np.sum(np.array([(1-((j)/5)) for j in range(5)]))
        wvts[i]=np.sum(np.array([(1-(j/5))*dfv[i,j+20] for j in range(5)]))/np.sum(np.array([(1-((j)/5)) for j in range(5)]))
        if dfv[i,5]>dfv[i-t,5]:
            dvb = wvtb[i]
        elif dfv[i,5]==dfv[i-t,5]:
            dvb = wvtb[i]-wvtb[i-t]
        else:
            dvb = 0 

        if dfv[i,10]>dfv[i-t,10]:
            dva = 0
        elif dfv[i,5]==dfv[i-t,5]:
            dva = wvts[i]-wvts[i-t]
        else:
            dva = wvts[i]


        voi2[i] = (dvb-dva)	
    return np.array(voi2)


    


def signal_OIR(df,t):
    """
    OIR Oder Imbalance Ratio
    """
    id_bp = 5
    id_bv = 15
    id_sp = 10
    id_sv = 20
    oir=[0]*len(df)
    dfv=df.values
    for i in range(t,dfv.shape[0]):
        wvtb=np.sum(np.array([(1-(j/5))*dfv[i,j+15] for j in range(5)]))/np.sum(np.array([(1-(j/5)) for j in range(5)]))
        wvts=np.sum(np.array([(1-(j/5))*dfv[i,j+20] for j in range(5)]))/np.sum(np.array([(1-(j/5)) for j in range(5)]))
        oir[i]=(wvtb-wvts)/(wvtb+wvts)
    return np.array(oir)

#@jit()
def signal_MPB(df):
    oir=[0]*len(df)
    dfv=df.values
    TP=[0]*len(df)
    MP=[0]*len(df)
    MPB=[0]*len(df)
    for i in range(len(df)):
        if dfv[i,4]==0:
            TP[i]=TP[i-1]
        else:
            TP[i]=dfv[i,3]/dfv[i,4]
        MP[i]=(dfv[i,5]+dfv[i,10])/2
        MPB=TP[i]-MP[i]
    return np.array(MPB)





def signal_rev1(df,t):
    """
    high frequaency reverse signal
    vectorize
    """
    # dfv = df.values
    id_bp = 5
    id_bv = 15
    id_sp = 10
    id_sv = 20
    id_vol = 4
    id_mp = np.where(df.columns == 'mid_price')[0][0]

    rollvolsum = df.volume.rolling(window = t).sum().values
    rollp = [0]*(t+1)
    rollvol=[0]*(t+1)
    rev1 = np.zeros_like(df.mid_price,np.float64)
    rev2 = np.zeros_like(df.mid_price,np.float64)
    for i in range(0,t+1):
        rollp[i] = np.roll(df.mid_price.values,i)
        rollvol[i] = np.roll(df.volume.values,i)
    for i in range(0,t):
        rev1+=np.log((rollp[0+i]/rollp[1+i]).astype(np.float64))*(rollvol[i].astype(np.float64))
        rev2+=np.log((rollp[0+i]/rollp[1+i]).astype(np.float64))*((zero_divide(1,rollvol[i])).astype(np.float64))
    rev2*=rollvolsum
    revstruct=rev1-rev2
    rev1 = rev1/rollvolsum
    return pd.DataFrame({'rev1':rev1,'rev2':revstruct})




def signal_talib(df,t):
    # vvy
    """
    overlap signals
    """
    
    SMA1 = talib.SMA(df.mid_price.values,t)
    SMA2 = talib.SMA(df.bs.values,t)
    upper, middle, lower = talib.BBANDS(df.mid_price.values,t,matype = talib.MA_Type.EMA)
    DEMA = talib.DEMA(df.mid_price.values, timeperiod = t)
    MA = talib.MA(df.mid_price.values, timeperiod = t, matype = 0)
    EMA = talib.EMA(df.mid_price.values, timeperiod = t)
    KAMA = talib.KAMA(df.mid_price.values, timeperiod = t)
    SAR = talib.SAR(df['sp1'].values, df['bp1'].values, acceleration=0, maximum=0)
    T3 = talib.T3(df.mid_price.values, timeperiod = t, vfactor = 0)
    TEMA = talib.TEMA(df.mid_price.values, timeperiod=t)
    SAREXT = talib.SAREXT(df['sp1'].values, df['bp1'].values, startvalue=0, offsetonreverse=0, accelerationinitlong=0, accelerationlong=0, accelerationmaxlong=0, accelerationinitshort=0, accelerationshort=0, accelerationmaxshort=0)


    """
    volume signals 
    """
    ATR = talib.ATR(df['sp1'].values, df['bp1'].values, df.mid_price.values, timeperiod=t)
    NATR = talib.NATR(df['sp1'].values, df['bp1'].values, df.mid_price.values, timeperiod=t)
    TRANGE = talib.TRANGE(df['sp1'].values, df['bp1'].values, df.mid_price.values)

    OBV = talib.OBV(df.mid_price.values,df.volume)

    """
    period signals
    """
    HT_DCPERIOD = talib.HT_DCPERIOD(df.mid_price.values)
    HT_DCPHASE = talib.HT_DCPHASE(df.mid_price.values)
    HT_PHASOR_inphase,HT_PHASOR_quadrature = talib.HT_PHASOR(df.mid_price.values)
    HT_SINE_sine,HT_SINE_leadsine = talib.HT_SINE(df.mid_price.values)
    HT_TRENDMODE = talib.HT_TRENDMODE(df.mid_price.values)

    """
    price signals 
    """
    AVGPRICE = talib.AVGPRICE(df.mid_price.shift(t).values, df['sp1'].values, df['bp1'].values, df.mid_price.values)


    AD = talib.AD(df['sp1'], df['bp1'], df.mid_price.values,df['volume'])
    ADOSC = talib.ADOSC(df['sp1'], df['bp1'], df.mid_price.values,df['volume'], fastperiod=t/2, slowperiod=t)


    """
    movement signals 
    """
    ADX = talib.ADX(df['sp1'].values, df['bp1'].values, df.mid_price.values, timeperiod=t)
    ADXR = talib.ADXR(df['sp1'].values, df['bp1'].values, df.mid_price.values, timeperiod=t)
    APO = talib.APO(df.mid_price.values, fastperiod=t, slowperiod=2*t, matype=0)
    AROON_aroondown,AROON_aroonup = talib.AROON(df['sp1'].values, df['bp1'].values, timeperiod=t)
    AROONOSC = talib.AROONOSC(df['sp1'].values, df['bp1'].values, timeperiod=t)
    BOP= talib.BOP(df['mid_price'].shift(t).values, df['sp1'].values, df['bp1'].values, df.mid_price.values)
    CCI = talib.CCI(df['sp1'].values, df['bp1'].values,df.mid_price.values, timeperiod=t)
    CMO = talib.CMO(df.mid_price.values, timeperiod=t)
    DX = talib.DX(df['sp1'].values, df['bp1'].values, df.mid_price.values, timeperiod=3*t)
    MACD_macd,MACD_macdsignal,MACD_macdhist = talib.MACD(df.mid_price.values, fastperiod=t, slowperiod=2*t, signalperiod=int(0.9*t))
    MACDEXT_macd,MACDEXT_macdsignal,MACDEXT_macdhist = talib.MACDEXT(df.mid_price.values, fastperiod=t, fastmatype=0, slowperiod=2*t, slowmatype=0, signalperiod=int(0.9*t), signalmatype=0)
    MFI = talib.MFI(df['sp1'].values, df['bp1'].values, df.mid_price.values, df['volume'], timeperiod=t)
    MINUS_DI = talib.MINUS_DI(df['sp1'].values, df['bp1'].values, df.mid_price.values, timeperiod=3*t)

    # 14. MINUS_DM：上升动向值
    # real = MINUS_DM(high, low, timeperiod=3*t)
    # 参数说明：high:最高价；low:最低价；df.mid_price.values：收盘价；timeperiod：时间周期
    MINUS_DM = talib.MINUS_DM(df['sp1'].values, df['bp1'].values, timeperiod=3*t)


    # 六、波动率指标
    # 1.MOM： 上升动向值
    # real = MOM(df.mid_price.values, timeperiod=2*t)
    # 参数说明：df.mid_price.values：收盘价；timeperiod：时间周期
    MOM = talib.MOM(df.mid_price.values, timeperiod=2*t)

    # 2.PLUS_DI
    # real = PLUS_DI(high, low, df.mid_price.values, timeperiod=3*t)
    # 参数说明：high:最高价；low:最低价；df.mid_price.values：收盘价；timeperiod：时间周期
    PLUS_DI = talib.PLUS_DI(df['sp1'].values, df['bp1'].values, df.mid_price.values, timeperiod=3*t)

    # 3.PLUS_DM
    # real = PLUS_DM(high, low, timeperiod=3*t)
    # 参数说明：high:最高价；low:最低价；df.mid_price.values：收盘价；timeperiod：时间周期
    PLUS_DM = talib.PLUS_DM(df['sp1'].values, df['bp1'].values, timeperiod=3*t)

    # 4. PPO： 价格震荡百分比指数
    # real = PPO(df.mid_price.values, fastperiod=12, slowperiod=26, matype=0)
    # 参数说明：df.mid_price.values：收盘价；timeperiod：时间周期，fastperiod:快周期； slowperiod：慢周期
    PPO = talib.PPO(df.mid_price.values, fastperiod=12, slowperiod=26, matype=0)

    # 5.ROC：变动率指标
    # real = ROC(df.mid_price.values, timeperiod=2*t)
    # 参数说明：df.mid_price.values：收盘价；timeperiod：时间周期
    ROC = talib.ROC(df.mid_price.values, timeperiod=2*t)

    # 6. ROCP：变动百分比
    # real = ROCP(df.mid_price.values, timeperiod=2*t)
    # 参数说明：df.mid_price.values：收盘价；timeperiod：时间周期
    ROCP = talib.ROCP(df.mid_price.values, timeperiod=2*t)

    # 7.ROCR ：变动百分率
    # real = ROCR(df.mid_price.values, timeperiod=2*t)
    # 参数说明：df.mid_price.values：收盘价；timeperiod：时间周期
    ROCR = talib.ROCR(df.mid_price.values, timeperiod=2*t)

    # 8. ROCR100 ：变动百分率（*100）
    # real = ROCR100(df.mid_price.values, timeperiod=2*t)
    # 参数说明：df.mid_price.values：收盘价；timeperiod：时间周期
    ROCR100 = talib.ROCR100(df.mid_price.values, timeperiod=2*t)

    # 9. RSI：相对强弱指数
    # real = RSI(df.mid_price.values, timeperiod=3*t)
    # 参数说明：df.mid_price.values：收盘价；timeperiod：时间周期
    RSI = talib.RSI(df.mid_price.values, timeperiod=3*t)

    # 10.STOCH ：随机指标,俗称KD
    # slowk, slowd = STOCH(high, low, df.mid_price.values, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    # 参数说明：high:最高价；low:最低价；df.mid_price.values：收盘价；fastk_period：N参数, slowk_period：M1参数, slowk_matype：M1类型, slowd_period:M2参数, slowd_matype：M2类型
    STOCH_slowk,STOCH_slowd = talib.STOCH(df['sp1'].values, df['bp1'].values, df.mid_price.values, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)

    # 11. STOCHF ：快速随机指标
    # fastk, fastd = STOCHF(high, low, df.mid_price.values, fastk_period=5, fastd_period=3, fastd_matype=0)
    STOCHF_fastk,STOCHF_fastd = talib.STOCHF(df['sp1'].values, df['bp1'].values, df.mid_price.values, fastk_period=5, fastd_period=3, fastd_matype=0)

    # 12.STOCHRSI：随机相对强弱指数
    # fastk, fastd = STOCHRSI(high, low, df.mid_price.values, timeperiod=3*t, fastk_period=5, fastd_period=3, fastd_matype=0)
    STOCHRSI_fastk,STOCHRSI_fastd = talib.STOCHF(df['sp1'].values, df['bp1'].values, df.mid_price.values, fastk_period = 5, fastd_period = 3, fastd_matype = 0)

    # 13.TRIX：1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
    # real = TRIX(df.mid_price.values, timeperiod=6*t)
    TRIX = talib.TRIX(df.mid_price.values, timeperiod=6*t)

    # 14.ULTOSC：终极波动指标
    # real = ULTOSC(high, low, df.mid_price.values, timeperiod1=7, timeperiod2=14, timeperiod3=28)
    ULTOSC = talib.ULTOSC(df['sp1'].values, df['bp1'].values, df.mid_price.values, timeperiod1=t, timeperiod2=2*t, timeperiod3=4*t)

    # 15.WILLR ：威廉指标
    # real = WILLR(high, low, df.mid_price.values, timeperiod=3*t)
    WILLR = talib.WILLR(df['sp1'].values, df['bp1'].values, df.mid_price.values, timeperiod = 3*t)


    # 七、Statistic Functions 统计学指标
    # 1. BETA：β系数也称为贝塔系数
    # real = BETA(high, low, timeperiod=5)
    BETA = talib.BETA(df['sp1'].values, df['bp1'].values, timeperiod = t)

    # 2. CORREL ：皮尔逊相关系数
    # real = CORREL(high, low, timeperiod=6*t)
    CORREL = talib.CORREL(df['sp1'].values, df['bp1'].values, timeperiod = 6*t)

    # 3.LINEARREG ：线性回归
    # real = LINEARREG(df.mid_price.values, timeperiod=3*t)
    LINEARREG = talib.LINEARREG(df.mid_price.values, timeperiod=3*t)

    # 4.LINEARREG_ANGLE ：线性回归的角度
    # real = LINEARREG_ANGLE(df.mid_price.values, timeperiod=3*t)
    LINEARREG_ANGLE = talib.LINEARREG_ANGLE(df.mid_price.values, timeperiod=3*t)

    # 5. LINEARREG_INTERCEPT ：线性回归截距
    # real = LINEARREG_INTERCEPT(df.mid_price.values, timeperiod=3*t)
    LINEARREG_INTERCEPT = talib.LINEARREG_INTERCEPT(df.mid_price.values, timeperiod=3*t)

    # 6.LINEARREG_SLOPE：线性回归斜率指标
    # real = LINEARREG_SLOPE(df.mid_price.values, timeperiod=3*t)
    LINEARREG_SLOPE = talib.LINEARREG_SLOPE(df.mid_price.values, timeperiod=3*t)

    # 7.STDDEV ：标准偏差
    # real = STDDEV(df.mid_price.values, timeperiod=5, nbdev=1)
    STDDEV = talib.STDDEV(df.mid_price.values, timeperiod=t, nbdev=1)

    # 8.TSF：时间序列预测
    # real = TSF(df.mid_price.values, timeperiod=3*t)
    TSF = talib.TSF(df.mid_price.values, timeperiod=3*t)

    # 9. VAR：方差
    # real = VAR(df.mid_price.values, timeperiod=5, nbdev=1)
    VAR = talib.VAR(df.mid_price.values, timeperiod=t, nbdev=1)
    
    sigdf = DataFrame({})

    __function_groups__ = {
        'Cycle Indicators': [
            'HT_DCPERIOD',
            'HT_DCPHASE',
            'HT_PHASOR',
            'HT_SINE',
            'HT_TRENDMODE',
            ],
        'Math Operators': [
            'ADD',
            'DIV',
            'MAX',
            'MAXINDEX',
            'MIN',
            'MININDEX',
            'MINMAX',
            'MINMAXINDEX',
            'MULT',
            'SUB',
            'SUM',
            ],

        'Momentum Indicators': [
            'ADX',
            'ADXR',
            'APO',
            'AROON',
            'AROONOSC',
            'BOP',
            'CCI',
            'CMO',
            'DX',
            'MACD',
            'MACDEXT',
            'MACDFIX',
            'MFI',
            'MINUS_DI',
            'MINUS_DM',
            'MOM',
            'PLUS_DI',
            'PLUS_DM',
            'PPO',
            'ROC',
            'ROCP',
            'ROCR',
            'ROCR100',
            'RSI',
            'STOCH',
            'STOCHF',
            'STOCHRSI',
            'TRIX',
            'ULTOSC',
            'WILLR',
            ],
        'Overlap Studies': [
            'BBANDS',
            'DEMA',
            'EMA',
            'HT_TRENDLINE',
            'KAMA',
            'MA',
            'MAMA',
            'MAVP',
            'MIDPOINT',
            'MIDPRICE',
            'SAR',
            'SAREXT',
            'SMA',
            'T3',
            'TEMA',
            'TRIMA',
            'WMA',
            ],
    
        'Price Transform': [
            'AVGPRICE',
            'MEDPRICE',
            'TYPPRICE',
            'WCLPRICE',
            ],
        'Statistic Functions': [
            'BETA',
            'CORREL',
            'LINEARREG',
            'LINEARREG_ANGLE',
            'LINEARREG_INTERCEPT',
            'LINEARREG_SLOPE',
            'STDDEV',
            'TSF',
            'VAR',
            ],
        'Volatility Indicators': [
            'ATR',
            'NATR',
            'TRANGE',
            ],
        'Volume Indicators': [
            'AD',
            'ADOSC',
            'OBV'
            ],
        }
    sigs = []
    for x in __function_groups__.values():
        sigs.extend(x)
    
    for x in sigs:
        try:
            sigdf[x] = eval(x)
        except:
            pass
            # print('no this signal')

    return sigdf 


