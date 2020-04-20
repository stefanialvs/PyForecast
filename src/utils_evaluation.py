import numpy as np
from numpy.random import seed
seed(1)

import pandas as pd
from math import sqrt

from sklearn.decomposition import PCA

######################################################################
# METRICS
######################################################################

def mse(y, y_hat):
    """
    Calculates Mean Squared Error.
    y: numpy array
      actual test values
    y_hat: numpy array
      predicted values
    return: MSE
    """
    mse = np.mean(np.square(y - y_hat))
    return mse

def mape(y, y_hat):
    """
    Calculates Mean Absolute Percentage Error.
    y: numpy array
      actual test values
    y_hat: numpy array
      predicted values
    return: MAPE
    """
    mape = np.mean(np.abs(y - y_hat) / np.abs(y))
    mape = 100 * mape
    return mape

def smape(y, y_hat):
    """
    Calculates Symmetric Mean Absolute Percentage Error.
    y: numpy array
      actual test values
    y_hat: numpy array
      predicted values
    return: sMAPE
    """
    smape = np.mean(np.abs(y - y_hat) / (np.abs(y) + np.abs(y_hat)))
    smape = 200 * smape
    return smape

def mase(y, y_hat, y_train, seasonality=1):
    """
    Calculates the M4 Mean Absolute Scaled Error.
    The scale is the mean absolute error of the seasonal naive model.
    y: numpy array
      actual test values
    y_hat: numpy array
      predicted values
    y_train: numpy array
      actual train values for Naive1 predictions
    seasonality: int
      main frequency of the time series
      Hourly 24,  Daily 7, Weekly 52,
      Monthly 12, Quarterly 4, Yearly 1
    return: MASE
    """
    scale = np.mean(abs(y_train[seasonality:] - y_train[:-seasonality]))
    mase = np.mean(abs(y - y_hat)) / scale
    mase = 100 * mase
    return mase


def rmsse(y, y_hat, y_train, seasonality=1):
    """
    Calculates the M5 Root Mean Squared Scaled Error.
    The scale is the mean absolute error of the seasonal naive model.
    y: numpy array
      actual test values
    y_hat: numpy array of len h (forecasting horizon)
      predicted values
    seasonality: int
      main frequency of the time series
      Hourly 24,  Daily 7, Weekly 52,
      Monthly 12, Quarterly 4, Yearly 1
    return: RMSSE
    """
    scale = np.mean(np.square(y_train[seasonality:] - y_train[:-seasonality]))
    rmsse = sqrt(mse(y, y_hat) / scale)
    rmsse = 100 * rmsse
    return rmsse

def evaluate_panel(y_test, y_hat, y_train, 
                   metric, seasonality):
    """
    Calculates metric for y and y_hat
    y_test: pandas df
      df with columns unique_id, ds, y
    y_hat: pandas df
      df with columns unique_id, ds, y_hat
    y_train: pandas df
      df with columns unique_id, ds, y (train)
      this is used in the scaled metrics
    seasonality: int
      main frequency of the time series
      Hourly 24,  Daily 7, Weekly 52,
      Monthly 12, Quarterly 4, Yearly 1
    return: list of metric evaluations
    """
    metric_name = metric.__code__.co_name
    uids = y_test.index.get_level_values('unique_id').unique()
    y_hat_uids = y_hat.index.get_level_values('unique_id').unique()
    assert len(y_test)==len(y_hat), "not same length"
    assert all(uids == y_hat_uids), "not same u_ids"

    idxs, evaluations = [], []
    for uid in uids:
        y_test_uid = y_test.loc[uid].values
        y_hat_uid = y_hat.loc[uid].values
        y_train_uid = y_train.loc[uid].y.values

        if metric_name in ['mase', 'rmsse']:
            evaluation_uid = metric(y=y_test_uid, y_hat=y_hat_uid, 
                                    y_train=y_train_uid, seasonality=seasonality)
        else:
            evaluation_uid = metric(y=y_test_uid, y_hat=y_hat_uid)

        idxs.append(uid)
        evaluations.append(evaluation_uid)
    
    idxs = pd.Index(idxs, name='unique_id')
    evaluations = pd.Series(evaluations, index=idxs)
    return evaluations

def compute_evaluations(y_test, y_hat, y_train, metrics, seasonality):
    """
    Calculates all metrics for y and y_hat panel data,
    And creates rank based on PCA dimensionality reduction.
    y_test: pandas df
      df with columns unique_id, ds, y
    y_hat: pandas df
      df with columns unique_id, ds, y_hat
    y_train: pandas df
      df with columns unique_id, ds, y (train)
      this is used in the scaled metrics
    metrics: list
      list of strings containing all metrics to compute
    seasonality: int
      main frequency of the time series
      Hourly 24,  Daily 7, Weekly 52,
      Monthly 12, Quarterly 4, Yearly 1
    return: list of metric evaluations
    """
    print("\n Evaluating models")
    evaluations = {}
    for metric_name, metric in metrics.items():
        print(metric_name)
        for col in y_hat.columns:
            mod_evaluation = evaluate_panel(y_test=y_test, y_hat=y_hat[col],
                                            y_train=y_train, metric=metric,
                                            seasonality=seasonality)
            mod_evaluation.name = y_hat[col].name

            if not (metric_name in evaluations.keys()):
                evaluations[metric_name] = [mod_evaluation]
            else:
                evaluations[metric_name].append(mod_evaluation)
    
    # Collapse Metrics
    for metric_name, metric in metrics.items():
        evaluations[metric_name] = pd.concat(evaluations[metric_name], axis=1)
        evaluations[metric_name] = evaluations[metric_name].mean(axis=0)
    
    evaluations = pd.DataFrame.from_dict(evaluations)
    
    # PCA rank
    X = evaluations.values
    pca = PCA(n_components=1)
    pca.fit(X)
    evaluations['pca_rank'] = pca.fit_transform(X)
    evaluations['pca_rank'] = evaluations['pca_rank'].rank(ascending=False)
    evaluations['pca_rank'] = evaluations['pca_rank'].astype(int)
    
    evaluations.sort_values(by='pca_rank', inplace=True)
    return evaluations
