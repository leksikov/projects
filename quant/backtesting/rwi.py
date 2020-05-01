import numpy as np
from hurst import compute_Hc, random_walk
from pyfinance import TSeries
import pandas as pd
import math

def ShannonReallyFast(shannon_df):
    #print(shannon_df, 'suka')
    patterns = {'000', '001', '010', '100', '101', '110', '011', '111'}
    shannon_df = pd.DataFrame(shannon_df.copy(), columns=['returns'])
    shannon_df['pattern1'] = np.sign(shannon_df[['returns']])
    shannon_df['pattern2'] = np.sign(shannon_df[['returns']].shift(1))
    shannon_df['pattern3'] = np.sign(shannon_df[['returns']].shift(2))
    shannon_df.dropna(inplace=True)
    shannon_df = shannon_df[(shannon_df.pattern1 != 0.0) & (shannon_df.pattern2 != 0.0) & (shannon_df.pattern3 != 0.0)]
    shannon_df['merged'] = shannon_df.pattern1.astype('str') + shannon_df.pattern2.astype('str') + shannon_df.pattern3.astype('str')
    shannon_df['merged'] = shannon_df['merged'].str.replace('.0', '')
    shannon_df['merged'] = shannon_df['merged'].str.replace('-1', '0')
    prob_df = shannon_df.groupby('merged').count()['returns']
    pattern_df = pd.DataFrame(patterns).set_index(0).sort_values(by=0)
    #prob_df = prob_df.drop(index='111')
    pattern_df = pattern_df.join(prob_df).fillna(0)
    
    ProbSum = 0.0
    for pattern in patterns:
        
        p = pattern_df.loc[pattern] / pattern_df.returns.sum()      
        value = p * np.log2(p)        
        if math.isnan(value):
            continue
        ProbSum = ProbSum + value
    Shannon_val = -ProbSum
    
    return Shannon_val
    

def ShannonFast(df):
    # wrong pattern match
    df=df[df!=0.0]
    shift_returns = df.shift(1)
    shift_returns2 = df.shift(2)
    shift_returns3 = df.shift(3)
    
    #df['test']  = np.sign(df['returns']).astype('str').dropna() + np.sign(df['shift_returns2']).astype('str').dropna() +  np.sign(df['shift_returns3']).dropna().astype('str')
    
    df =df.dropna(axis=0)
    
    Pattern  = np.sign(df.shift(1).dropna()).astype('str') + np.sign(shift_returns2.dropna()).astype('str') +  np.sign(shift_returns3.dropna()).astype('str')
    
    Pattern=Pattern.dropna().str.replace('.0','')
    patternList = Pattern.dropna().str.replace('.0','').unique().tolist()
    total = 0.0
    ShannonPatterns = {}
    
    for e in patternList:
        ShannonPatterns[e] = 0

    for pattern in ShannonPatterns.keys():        
        value = (len(np.where(Pattern==pattern)[0]))
        ShannonPatterns[pattern] +=  value
        total += value
        
    
    ProbSum = 0.0
    for pattern in ShannonPatterns.keys():
        p = ShannonPatterns[pattern] / total        
        value = p * np.log2(p)        
        if math.isnan(value):
            continue
        ProbSum = ProbSum + value
    Shannon_val = -ProbSum
    return Shannon_val

def Shannon(df, patternSize):
    
    chunks = []
    for i in range(0, len(df)):
        chunks.append(df[i:i+patternSize])
    
    chunks = chunks[:-patternSize-1]
    
    
    chunks = [np.array2string(x) for x in chunks.copy()]
    
    
    
    
    chunks_set = list(set(chunks))
    
    
    visited = {}
    total = 0
    for el in chunks_set:
        if (el not in visited):
            f = chunks.count(el)
            visited[el] = f
            total = total + f
    ProbSum = 0.0
    for el in visited:
        p = visited[el]/total
        value = p * np.log2(p)
        #visited[el] = value
        ProbSum = ProbSum + value
    Shannon_val = -ProbSum
    del visited, chunks
    
    
    
    return Shannon_val

def marketMeannes(df_):
    
    m = np.median(df_) 
    nh = 0
    nl = 0
    
    for i in range(1, len(df_)-1):
        Pt = df_[i]
        Py = df_[i-1]
        
        if (Py > m) & (Py > Pt):
            nl += 1
        elif (Py < m) & (Py < Pt):
            nh += 1
        else:
            None
    return (nl+nh)/(len(df_)-1)
        
    
    
def Momersion(df_):
    #print(np.where(df == 1)[0])
    #df = df['returns'].copy() * df['returns'].shift(1)
    df_ = df_.copy() * df_.shift(1)
    
    df_ = df_.fillna(0) #df.dropna()
    df_ = np.sign(df_)
    pos = len(np.where(df_ == 1)[0])
    neg = len(np.where(df_ == -1)[0])
    #zero = len(np.where(df == 0.0)[0])
    if (pos + neg) == 0.0:
        return -1.0
    #print(pos, neg)
    mom = (pos / (pos+neg )) 
    return mom

def momersionPeriod(ts_, p_):
    
    ts_1 = np.sign(ts_.pct_change(p_).fillna(0))
    ts_2 = np.sign(ts_.pct_change(p_).shift(p_).fillna(0))
    
    val = ts_1 * ts_2
    pos = len(val[val == 1])
    neg = len(val[val == -1])
        
    if pos + neg == 0:
        return 0
    
    mom = (pos/(pos+neg)) 
    return mom
    
# https://pypi.org/project/hurst/
def hurst(ts):
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    # plot on log-log scale
    #plt.plot(np.log(lags), np.log(tau)); plt.show()
    # calculate Hurst as slope of log-log plot
    #print(lags, tau)
    m = np.polyfit(np.log(lags), np.log(tau), 1)

 
    hurst = m[0]*2.0
    #print ('hurst = ',hurst)
    #plt.clf(), plt.close()
    return hurst

def MomersionDouble(df_):

    #df = df[df!=0.0]
    shift_returns = df_.shift(1).fillna(0)
    shift_returns2 = df_.shift(2).fillna(0)
    shift_returns3 = df_.shift(3).fillna(0)
    Pattern = np.sign(shift_returns * shift_returns2)
    Pattern2= np.sign(shift_returns2 * shift_returns3)
    df_ = df_.fillna(0) #df.dropna()

    pp = len(np.where( (Pattern == 1 ) & (Pattern2 == 1 ) )[0])
    pm = len(np.where( (Pattern == 1 ) & (Pattern2 == -1 ) )[0])
    mp = len(np.where( (Pattern == -1 ) & (Pattern2 == 1 ) )[0])
    mm = len(np.where( (Pattern == -1 ) & (Pattern2 == -1 ) )[0])

    total = 0.50+(pp+pm-mp-mm)/(pp + pm + mp+ mm)
    #threshUp = total>=np.sqrt(len(df))

    return total #(total, len(df), np.sqrt(len(df)))

def proportion(df):
    pp = len(np.where( (df >0.0  ) )[0])
    mm = len(np.where( (df < 0.0 ) )[0])
    
    if mm == 0.0 or mm is None:
        mm = 1
    return pp/mm
    
def proportionPos(df):
    pp = len(np.where( (df >0.0  ) )[0])
    mm = len(np.where( (df < 0.0 ) )[0])
    
    if mm == 0.0 or mm is None:
        mm = 1
    return  pp/(pp+mm) 

def autoCorrel(df, lag):
    return pd.Series.autocorr(df, lag) 


def hurstF(ts):
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    # plot on log-log scale
    #plt.plot(np.log(lags), np.log(tau)); plt.show()
    # calculate Hurst as slope of log-log plot
    #print(lags, tau)
    m = np.polyfit(np.log(lags), np.log(tau), 1)

 
    hurst = m[0]*2.0
    #print ('hurst = ',hurst)
    #plt.clf(), plt.close()
    return hurst


def hurstF2(p):
    lags = range(2,100)


    variancetau = []; tau = []

    for lag in lags: 

        #  Write the different lags into a vector to compute a set of tau or lags
        tau.append(lag)

        # Compute the log returns on all days, then compute the variance on the difference in log returns
        # call this pp or the price difference
        pp = np.subtract(p[lag:], p[:-lag])
        variancetau.append(np.var(pp))

    # we now have a set of tau or lags and a corresponding set of variances.
    #print tau
    #print variancetau

    # plot the log of those variance against the log of tau and get the slope
    m = np.polyfit(np.log10(tau),np.log10(variancetau),1)

    hurst = m[0] / 2

    return hurst

def hurstF3(series):
   

    #H, c, data = compute_Hc(series.replace([np.inf, -np.inf], np.na).dropna(), kind='price', simplified=True)
    H, c, data = compute_Hc(series.replace([np.inf, -np.inf], np.nan).dropna(), kind='random_walk', simplified=False)
    return H

def hurstF4(series):
   
    
    H, c, data = compute_Hc(series, kind='random_walk', simplified=True)
    return H



def generateRWI2(df):
    MomVal = Momersion(df['returns']) 
    MomDouble = MomersionDouble(df['returns'])
    try:
    
        h1 =  hurstF(df[['price']])
    except:
        h1=[0]
    
    try:
        h2 =  hurstF2(df[['price']])
    except:
        h2=[0]
    
    try:
        h3 =  hurstF3(df['price'])
    except:
        h3=[0]
    
    try:
    
        h4 = hurstF4(df['price'])
    except:
        h4=[0]
  
    #df = df.copy().join(autoCorr_features(df[['returns']].copy()), rsuffix='_suka_')
    
    MMIR = marketMeannes(df['returns'].values)
    
    MMIP = marketMeannes(df['price'].values)
    ShannonVal = ShannonFast(df['returns'])
    prop = proportionPos(df['returns'])
    correl_1 = autoCorrel(df.returns, 1)
    correl_2 = autoCorrel(df.returns, 2)
    correl_3 = autoCorrel(df.returns, 3)
    correl_4 = autoCorrel(df.returns, 4)
    correl_5 = autoCorrel(df.returns, 5)
    correl_10 = autoCorrel(df.returns, 10)
    correl_20 = autoCorrel(df.returns, 20)
    correl_100 = autoCorrel(df.returns, 100)
    correl_list = [correl_1, correl_2, correl_3, correl_4, correl_5, correl_10, correl_20, correl_100]
    
    var_std = df['returns'].std()
    var_mean = df['returns'].mean()
    var_median = df['returns'].mean()
    
    
    return [MomVal, MomDouble, h1[0], h2[0], h3, h4, MMIR, MMIP, ShannonVal, prop, var_std, var_mean, var_median] + correl_list

def generate_features(df):
    
    
    df['returns'] = np.log(df['price']).pct_change(1)
    
    df['ROC_2'] =  np.log(df['price'].copy()).pct_change(2)
    
    df['ROC_3'] = np.log(df['price'].copy()).pct_change(3)
    df['ROC_5'] = np.log(df['price'].copy()).pct_change(5)
    df['ROC_20'] = np.log(df['price'].copy()).pct_change(20)
    df['ROC_50'] = np.log(df['price'].copy()).pct_change(50)
    df['ROC_100'] = np.log(df['price'].copy()).pct_change(100)
    df['ROC_200'] = np.log(df['price'].copy()).pct_change(200)
    #df['ROC_300'] = np.log(df['price'].copy()).pct_change(300)
    df['ROC_500'] = np.log(df['price'].copy()).pct_change(500)
    
    
    df['abs_returns'] = np.abs( np.log(np.abs(df['price'].copy())).pct_change())
    

    df = df.replace([np.inf, -np.inf], np.nan)
    
    
    
    return df

def transform_series(tmp):
    #tmp = [e[0] if (type(e)==np.ndarray) else e for e in tmp.copy()]
    df = pd.DataFrame(np.asarray(tmp)+100)
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    df_ = scaler.fit_transform(df)
    df_ = pd.DataFrame(df_, columns=df.columns,index=df.index)
    df = (100+df_*100)
    return df
    
def momentum_strategy(ts, period, type_ret):
    
    ts['returns'] = ts[['Close']].diff(period).fillna(0)
    ts['returns2'] = ts[['Close']].diff(period).fillna(0).shift(period).fillna(0)
    
    ts['entry'] = np.sign(ts['returns'] * ts.returns2) # signal
    #ts['entry'] = ts['entry'].shift(1).fillna(0)
    ts['direction'] = np.sign(ts.returns) # signal up, down
    ts['shift_returns'] = ts.returns.shift(-1).fillna(0)
    ts['shift_Change'] = ts['Change'].shift(-1).fillna(0)
    if type_ret == 1:
        ts['val'] = ts.shift_returns * ts.direction
    else:
        ts['val'] = ts.shift_Change * ts.direction
    return ts #ts[(ts.entry==1) & (ts.val > 0) ].val.sum() / (ts[(ts.entry==1) & (ts.val < 0) ].val.abs().sum())  #.dropna()

def mean_reversal(ts, period, type_ret):
    
    ts['returns'] = ts[['Close']].diff(period).fillna(0)
    ts['returns2'] = ts[['Close']].diff(period).fillna(0).shift(period).fillna(0)
    
    ts['entry'] = np.sign(ts['returns'] * ts.returns2) * (-1) # signal
    #ts['entry'] = ts['entry'].shift(1).fillna(0)
    ts['direction'] = np.sign(ts.returns * (-1)) # signal up, down
    ts['shift_returns'] = ts.returns.shift(-1).fillna(0)
    ts['shift_Change'] = ts['Change'].shift(-1).fillna(0)
    ts['val'] = ts.shift_returns * ts.direction
    if type_ret == 1:
        ts['val'] = ts.shift_returns * ts.direction
    else:
        ts['val'] = ts.shift_Change * ts.direction
    return ts #ts[(ts.entry==1) & (ts.val > 0) ].val.sum() / (ts[(ts.entry==1) & (ts.val < 0) ].val.abs().sum())  #.dropna()


def mean_reversal2(ts, period, type_ret):
    
    ts['returns'] = ts[['Close']].diff(period).fillna(0)
    ts['returns2'] = ts[['Close']].diff(period).fillna(0).shift(period).fillna(0)
    
    ts['entry'] = np.sign(ts['returns'] * ts.returns2)  # signal
    #ts['entry'] = ts['entry'].shift(1).fillna(0)
    ts['direction'] = np.sign(ts.returns * (-1)) # signal up, down
    ts['shift_returns'] = ts.returns.shift(-1).fillna(0)
    ts['shift_Change'] = ts['Change'].shift(-1).fillna(0)
    ts['val'] = ts.shift_returns * ts.direction
    if type_ret == 1:
        ts['val'] = ts.shift_returns * ts.direction
    else:
        ts['val'] = ts.shift_Change * ts.direction
    return ts #ts[(ts.entry==1) & (ts.val > 0) ].val.sum() / (ts[(ts.entry==1) & (ts.val < 0) ].val.abs().sum())  #.dropna()


