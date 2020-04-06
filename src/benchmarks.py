import numpy as np
from numpy.random import seed
seed(1)

import pandas as pd
from math import sqrt

from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin

########################
# UTILS
########################

def detrend(insample_data):
  """
  Calculates a & b parameters of LRL
  :param insample_data:
  :return:
  """
  x = np.arange(len(insample_data))
  a, b = np.polyfit(x, insample_data, 1)
  return a, b

def deseasonalize(original_ts, ppy):
  """
  Calculates and returns seasonal indices
  :param original_ts: original data
  :param ppy: periods per year
  :return:
  """
  """
  # === get in-sample data
  original_ts = original_ts[:-out_of_sample]
  """
  if seasonality_test(original_ts, ppy):
    # ==== get moving averages
    ma_ts = moving_averages(original_ts, ppy)

    # ==== get seasonality indices
    le_ts = original_ts * 100 / ma_ts
    le_ts = np.hstack((le_ts, np.full((ppy - (len(le_ts) % ppy)), np.nan)))
    le_ts = np.reshape(le_ts, (-1, ppy))
    si = np.nanmean(le_ts, 0)
    norm = np.sum(si) / (ppy * 100)
    si = si / norm
  else:
    si = np.ones(ppy)

  return si

def ses(a, x, h, job):
    y = np.empty(x.size + 1)
    y[0] = x[0]

    for i, val in enumerate(x):
        y[i+1] = a * val + (1-a) * y[i]

    fitted = y[:-1]
    forecast = np.repeat(y[-1], h)
    if job == 'train':
        return np.mean((fitted - x)**2)
    if job == 'fit':
        return fitted
    return {'fitted': fitted, 'mean': forecast}

def demand(x):
    return x[x > 0]

def intervals(x):
    y = []

    ctr = 1
    for i, val in enumerate(x):
        if val == 0:
            ctr += 1
        else:
            y.append(ctr)
            ctr = 1

    y = np.array(y)
    return y

def moving_averages(ts_init, window):
  """
  Calculates the moving averages for a given TS
  :param ts_init: the original time series
  :param window: window length
  :return: moving averages ts
  """
  """
  As noted by Professor Isidro Lloret Galiana:
  line 82:
  if len(ts_init) % 2 == 0:
  
  should be changed to
  if window % 2 == 0:
  
  This change has a minor (less then 0.05%) impact on the calculations of the seasonal indices
  In order for the results to be fully replicable this change is not incorporated into the code below
  """
  ts_init = pd.Series(ts_init)
  
  if len(ts_init) % 2 == 0:
    ts_ma = ts_init.rolling(window, center=True).mean()
    ts_ma = ts_ma.rolling(2, center=True).mean()
    ts_ma = np.roll(ts_ma, -1)
  else:
    ts_ma = ts_init.rolling(window, center=True).mean()

  return ts_ma

def seasonality_test(original_ts, ppy):
  """
  Seasonality test
  :param original_ts: time series
  :param ppy: periods per year
  :return: boolean value: whether the TS is seasonal
  """
  s = acf(original_ts, 1)
  for i in range(2, ppy):
    s = s + (acf(original_ts, i) ** 2)

  limit = 1.645 * (sqrt((1 + 2 * s) / len(original_ts)))

  return (abs(acf(original_ts, ppy))) > limit

def acf(data, k):
  """
  Autocorrelation function
  :param data: time series
  :param k: lag
  :return:
  """
  m = np.mean(data)
  s1 = 0
  for i in range(k, len(data)):
    s1 = s1 + ((data[i] - m) * (data[i - k] - m))

  s2 = 0
  for i in range(0, len(data)):
    s2 = s2 + ((data[i] - m) ** 2)

  return float(s1 / s2)

###################################
# CONTINUOUS BENCHMARK MODELS
###################################

class Naive:
  """
  Naive model.
  """
  def __init__(self):
    pass
  
  def fit(self, ts_init):
    """
    ts_init: the original time series
    ts_naive: last observations of time series
    """
    self.ts_naive = [ts_init[-1]]
    return self

  def predict(self, h):
    return np.array(self.ts_naive * h)


class SeasonalNaive:
  """
  Seasonal Naive model.
  """
  def __init__(self):
    pass
  
  def fit(self, ts_init, seasonality):
    """
    ts_init: the original time series
    frcy: frequency of the time series
    ts_naive: last observations of time series
    """
    self.ts_seasonal_naive = ts_init[-seasonality:]
    return self

  def predict(self, h):
    repetitions = int(np.ceil(h/len(self.ts_seasonal_naive)))
    y_hat = np.tile(self.ts_seasonal_naive, reps=repetitions)[:h]
    return y_hat


class Naive2(BaseEstimator, RegressorMixin):
  """
  Naive2: Naive after deseasonalization.
  """
  def __init__(self, seasonality=7):
    self.seasonality = seasonality
    
  def fit(self, ts_init):
    seasonality_in = deseasonalize(ts_init, ppy=self.seasonality)
    windows = int(np.ceil(len(ts_init) / self.seasonality))
    
    self.ts_init = ts_init
    self.s_hat = np.tile(seasonality_in, reps=windows)[:len(ts_init)]
    self.ts_des = ts_init / self.s_hat
            
    return self
    
  def predict(self, h):
    s_hat = SeasonalNaive().fit(self.s_hat,
                                seasonality=self.seasonality).predict(h)
    r_hat = Naive().fit(self.ts_des).predict(h)        
    y_hat = s_hat * r_hat
    return y_hat

class RandomWalkDrift:
  """
  RandomWalkDrift: Random Walk with drift.
  """
  def __init__(self):
    pass
  
  def fit(self, ts_init):
    self.drift = (ts_init[-1] - ts_init[0])/(len(ts_init)-1)
    self.naive = [ts_init[-1]]
    return self

  def predict(self, h):
    naive = np.array(self.ts_naive * h)
    drift = self.drift*np.array(range(1,h+1))
    y_hat = naive + drift
    return y_hat

###################################
# SPARSE BENCHMARK MODELS
###################################

class CrostonMethod(BaseEstimator, RegressorMixin):
  """
  CrostonMethod:
  """
  def __init__(self, kind='classic'):
      allowed_kinds = ('classic', 'optimized', 'sba')
      if kind not in allowed_kinds:
          raise ValueError(f'kind must be one of {allowed_kinds}')
      self.kind = kind
    
  def fit(self, ts_init):
      if self.kind in ('classic', 'optimized'):
          self.mult = 1
      else:
          self.mult = 0.95
      if self.kind in ('classic', 'sba'):
          self.a1 = self.a2 = 0.1
      yd = demand(ts_init)
      yi = intervals(ts_init)
      if self.kind == 'optimized':
          self.a1 = minimize(ses, 0, args=(yd, 1, 'train'), bounds=[(0.1, 0.3)], method='L-BFGS-B').x[0]
          self.a2 = minimize(ses, 0, args=(yi, 1, 'train'), bounds=[(0.1, 0.3)], method='L-BFGS-B').x[0]
      ydp = ses(self.a1, yd, h=1, job='forecast')['mean']
      yip = ses(self.a2, yi, h=1, job='forecast')['mean']
      self.pred = ydp / yip * self.mult
      return self

  def predict(self, h):
      preds = np.repeat(self.pred, h)
      return preds

class TSBMethod:
  """
  Teunter-Syntetos-Babai Method:
  """
  def __init__(self):
    pass
    
  def fit(self, ts_init):
    pass
    
  def predict(self, h):
    pass

