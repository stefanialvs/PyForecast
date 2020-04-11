import numpy as np
from numpy.random import seed
seed(1)

import pandas as pd
from math import sqrt

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
    y = np.reshape(y, (-1,))
    y_hat = np.reshape(y_hat, (-1,))
    mse = np.mean(np.square(y - y_hat)).item()
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
    y = np.reshape(y, (-1,))
    y_hat = np.reshape(y_hat, (-1,))
    mape = np.mean(np.abs(y - y_hat) / np.abs(y))
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
    y = np.reshape(y, (-1,))
    y_hat = np.reshape(y_hat, (-1,))
    smape = np.mean(2.0 * np.abs(y - y_hat) / (np.abs(y) + np.abs(y_hat)))
    return smape

def mase(y, y_hat, y_train, seasonality):
    """
    Calculates Mean Absolute Scaled Error.
    y: numpy array
      actual test values
    y_hat: numpy array
      predicted values
    y_train: numpy array
      actual train values for Naive1 predictions
    seasonality: int
      main frequency of the time series
      Quarterly 4, Daily 7, Monthly 12
    return: MASE
    """
    y_hat_naive = []
    for i in range(seasonality, len(y_train)):
        y_hat_naive.append(y_train[(i - seasonality)])

    masep = np.mean(abs(y_train[seasonality:] - y_hat_naive))
    mase = np.mean(abs(y - y_hat)) / masep
    return mase


def rmsse(y, y_hat, denom):
    """
    Calculates the Root Mean Squared Scaled Error.
    y: numpy array
      actual test values
    y_hat: numpy array of len h (forecasting horizon)
      predicted values
    h: forecasting horizon
    return: RMSSE
    """
    num = mse(y, y_hat)
    res = sqrt(num / denom)
    return res

def evaluate_panel(y_panel, y_hat_panel, metric,
                   y_insample=None, seasonality=None):
    """
    Calculates metric for y_panel and y_hat_panel
    y_panel: pandas df
    panel with columns unique_id, ds, y
    y_naive2_panel: pandas df
    panel with columns unique_id, ds, y_hat
    y_insample: pandas df
    panel with columns unique_id, ds, y (train)
    this is used in the MASE
    seasonality: int
    main frequency of the time series
    Quarterly 4, Daily 7, Monthly 12
    return: list of metric evaluations
    """
    metric_name = metric.__code__.co_name
    assert len(y_panel)==len(y_hat_panel)
    uids = y_panel.index.get_level_values('unique_id').unique()
    y_hat_uids = y_hat_panel.index.get_level_values('unique_id').unique()
    assert all(uids == y_hat_uids), "not same u_ids"

    evaluation = []
    for uid in uids: 
        y_uid = y_panel.loc[uid].values
        y_hat_uid = y_hat_panel.loc[uid].values

        if metric_name == 'mase':
            assert (y_insample is not None) and (seasonality is not None)
            y_insample_uid = y_insample_panel.loc[uid].values
            evaluation_uid = metric(y_uid, y_hat_uid, y_insample_uid, seasonality)
        else:
            print("len(y_uid)", len(y_uid))
            print("len(y_hat_uid)", len(y_hat_uid))
            print("y_uid", y_uid)
            print("y_hat_uid", y_hat_uid)
            evaluation_uid = metric(y_uid, y_hat_uid)
    evaluation.append(evaluation_uid)
    return evaluation
