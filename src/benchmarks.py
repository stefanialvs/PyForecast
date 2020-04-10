import numpy as np
from numpy.random import seed
seed(1)

import pandas as pd
from math import sqrt

from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin, clone


######################################################################
# UTILS
######################################################################

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

######################################################################
# PANEL MODEL CLASS
######################################################################

class PanelModel:
    """
    Panel model class.
    This class inherits an instantiated univariate time series model with 
    fit and predict methods and declares common fit and predict methods
    for full panel data. The panel dataframe is defined by the each series 
    unique_id and their datestamps.
    """
    def __init__(self, model, fill_na=True):
        """
        model: sklearn BaseEstimator class
        """
        self.model = model
        self.fill_na = fill_na

    def fit(self, X, y):
        """
        X: pandas dataframe
            dataframe with panel data covariates defined by 'unique_id' and 'ds'
        y: pandas dataframe
            dataframe with panel data target variable defined by 'unique_id' 
            and 'ds'
        """
        assert X.index.names == ['unique_id', 'ds']
        assert y.index.names == ['unique_id', 'ds']
        self.model_ = {}
        self.mean_ = {}
        for uid, X_uid in X.groupby('unique_id'): 
            y_uid = y.loc[uid]
            self.model_[uid] = clone(self.model)
            self.model_[uid].fit(X_uid.values, y_uid.values)
            if self.fill_na:
                self.mean_[uid] = np.nanmean(y_uid.values)
        return self

    def predict(self, X):
        """
        X: pandas dataframe
            dataframe with panel data covariates defined by 'unique_id' and 'ds'
        """
        idxs, preds = [], []
        for uid, X_uid in X.groupby('unique_id'):
            y_hat_uid = self.model_[uid].predict(X_uid.values)
            if self.fill_na:
                y_hat_uid[np.isnan(y_hat_uid)] = self.mean_[uid]
            assert len(y_hat_uid)==len(X_uid), "Predictions length mismatch"
            idxs.extend(X_uid.index)
            preds.extend(y_hat_uid)
        idx = pd.MultiIndex.from_tuples(idxs, names=('unique_id', 'ds'))
        preds = pd.Series(preds, index=idx)
        return preds


######################################################################
# CONTINUOUS BENCHMARK MODELS
######################################################################


class Naive(BaseEstimator, RegressorMixin):
    """
    Naive model.
    This benchmark model produces a forecast that is equal to
    the last observed value for a given time series.
    """
    def __init__(self, h):
        """
        h: int
            forecast horizon, the number of times the last value
            will be repeated
        """
        self.h = h
  
    def fit(self, X, y):
        """
        X: numpy array
            time series covariates (for pipeline compatibility)
        y: numpy array
            train values of the time series
        """
        self.y_hat = [float(y[-1])]
        return self
    
    def predict(self, X):
        """
        X: numpy array
            time series covariates (for pipeline compatibility)
        return
        y_hat: numpy array
            forecast for time horizon 'h' repeating the last
            value of y.
        """
        y_hat = np.array(self.y_hat * self.h)
        return y_hat


class SeasonalNaive(BaseEstimator, RegressorMixin):
    """
    Seasonal Naive model.
    This benchmark model produces a forecast that is equal to
    the last observed value of the same season for a given time 
    series.
    """
    def __init__(self, h, seasonality):
        """
        h: int
            forecast horizon, the number of times the last value
            will be repeated.
        seasonality: int
            seasonality of the time series.
        """
        self.seasonality = seasonality
        self.h = h
  
    def fit(self, X, y):
        """
        X: numpy array
            time series covariates (for pipeline compatibility)
        y: numpy array
            train values of the time series
        """
        self.y_hat = y[-self.seasonality:].flatten()
        return self

    def predict(self, X):
        """
        X: numpy array
            time series covariates (for pipeline compatibility)
        return
        y_hat: numpy array
            forecast for time horizon 'h' repeating the last
            values for each season.
        """
        repetitions = int(np.ceil(self.h/self.seasonality))
        y_hat = np.tile(self.y_hat, reps=repetitions)
        y_hat = y_hat[:self.h]        
        assert len(y_hat)==self.h
        return y_hat


class Naive2(BaseEstimator, RegressorMixin):
    """
    Naive2 model.
    This benchmark model produces a forecast that is equal to
    the last observed value for a given time series.
    """
    def __init__(self, h, seasonality):
        """
        h: int
            forecast horizon
        seasonality: int
            seasonality of the time series.
        """
        self.h = h
        self.seasonality = seasonality
        self.sn_model = SeasonalNaive(h=self.h, seasonality=self.seasonality)
        self.n_model = Naive(h=self.h)
    
    def fit(self, X, y):
        """
        X: numpy array
            time series covariates (for pipeline compatibility)
        y: numpy array
            train values of the time series
        """
        y = y.flatten()
        seasonality_in = deseasonalize(y, ppy=self.seasonality)
        windows = int(np.ceil(len(y) / self.seasonality))
    
        self.y = y
        self.s_hat = np.tile(seasonality_in, reps=windows)[:len(y)]
        self.ts_des = y / self.s_hat
            
        return self
    
    def predict(self, X):
        """
        X: numpy array
            time series covariates (for pipeline compatibility)
        return
        y_hat: numpy array
            forecast for time horizon 'h' repeating the last
            values for each season.
        """
        s_hat = self.sn_model.fit(X, self.s_hat).predict(X)
        r_hat = self.n_model.fit(X, self.ts_des).predict(X)
        y_hat = s_hat * r_hat
        return y_hat


class RandomWalkDrift(BaseEstimator, RegressorMixin):
    """
    RandomWalkDrift: Random Walk with drift.
    """
    def __init__(self, h):
        """
        h: int
            forecast horizon
        """
        self.h = h
  
    def fit(self, X, y):
        """
        X: numpy array
            time series covariates (for pipeline compatibility)
        y: numpy array
            train values of the time series
        """
        self.drift = (float(y[-1]) - float(y[0]))/(len(y)-1)
        self.naive = [float(y[-1])]
        return self

    def predict(self, h):
        """
        X: numpy array
            time series covariates (for pipeline compatibility)
        return
        y_hat: numpy array
            forecast for time horizon 'h'.
        """
        naive = np.array(self.naive * self.h)
        drift = self.drift * np.array(range(1,self.h+1))
        y_hat = naive + drift
        return y_hat

######################################################################
# SPARSE BENCHMARK MODELS
######################################################################

class Croston(BaseEstimator, RegressorMixin):

    def __init__(self, kind='classic'):
        allowed_kinds = ('classic', 'optimized', 'sba')
        if kind not in allowed_kinds:
            raise ValueError(f'kind must be one of {allowed_kinds}')
        self.kind = kind
        if kind in ('classic', 'optimized'):
            self.mult = 1
        else:
            self.mult = 0.95
        if kind in ('classic', 'sba'):
            self.a1 = self.a2 = 0.1

    def _optimize(self, y):
        res = minimize(fun=ses, x0=0, args=(y, 1, 'train'), 
                       bounds=[(0.1, 0.3)],
                       method='L-BFGS-B').x[0]
        return res

    def fit(self, X, y):
        yd = demand(y)
        yi = intervals(y)

        if self.kind == 'optimized':
            self.a1 = self._optimize(yd) 
            self.a2 = self._optimize(yi) 

        ydp = ses(self.a1, yd, h=1, job='forecast')['mean']
        yip = ses(self.a2, yi, h=1, job='forecast')['mean']
        self.pred_ = ydp / yip * self.mult
        return self

    def predict(self, X):
        h = X.shape[0]
        preds = np.repeat(self.pred_, h)
        return preds 
