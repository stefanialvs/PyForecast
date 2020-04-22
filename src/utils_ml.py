import sys
sys.path.append('./src/')

import numpy as np
import pandas as pd


from src.benchmarks import *
from src.utils_data import m4_parser, seas_dict
from src.utils_evaluation import *
from src.utils_visualization import plot_grid_series

######################################################################
# PARSE USER INTERACTION
######################################################################

def button_models_metrics_filter(dataset_name, models_filter, metrics_filter):
    # Parse time series arguments
    h = seas_dict[dataset_name]['output_size']
    seasonality = seas_dict[dataset_name]['seasonality']
    
    models = {'Naive':  Naive(h=h),
              'SeasonalNaive': SeasonalNaive(h=h, seasonality=seasonality),
              'Naive2': Naive2(h=h, seasonality=seasonality),
              'RandomWalkDrift': RandomWalkDrift(h=h),
              'Croston': Croston(kind='classic'),
              'MovingAverage': MovingAverage(h=h, n_obs=20),
              'SeasonalMovingAverage': SeasonalMovingAverage(h=h, n_seasons=2,
                                                             seasonality=seasonality)}
    
    metrics = {'RMSE': rmse, 'MAPE': mape, 'SMAPE': smape, 
               'MASE': mase, 'RMSSE': rmsse}
    
    # Filtered models and metrics
    models = {model_name: models[model_name] for model_name in models_filter}
    metrics = {metric_name: metrics[metric_name] for metric_name in metrics_filter}
    
    return models, metrics

def button_uids_filter(y_df, size=8):
    uids = y_df.index.get_level_values('unique_id').unique()
    uids_sample = np.random.choice(uids, size=8)
    return uids_sample

######################################################################
# MACHINE LEARNING PIPELINE
######################################################################

def ml_pipeline(dataset_name, num_obs, models_filter, metrics_filter, h):
    # Set random seeds
    np.random.seed(1)

    # Filter models and metrics
    models, metrics = button_models_metrics_filter(dataset_name, models_filter, metrics_filter)


    # Parse arguments
    # h = seas_dict[dataset_name]['output_size']
    seasonality = seas_dict[dataset_name]['seasonality']
    
    # Read data
    directory = './data/'
    #X_train_df, y_train_df, X_test_df, y_test_df = m4_parser(dataset_name, directory, num_obs)
    X_train_df, y_train_df, X_test_df, y_test_df = prepare_m4_data(dataset_name, directory, num_obs, h)

    # Pre sort dataframes for efficiency
    X_train_df = X_train_df.set_index(['unique_id', 'ds']).sort_index()
    y_train_df = y_train_df.set_index(['unique_id', 'ds']).sort_index()
    X_test_df = X_test_df.set_index(['unique_id', 'ds']).sort_index()
    y_test_df = y_test_df.set_index(['unique_id', 'ds']).sort_index()
    
    # Fit and predict benchmark models
    preds = [y_test_df.y]
    print("\n Fitting models")
    for model_name, model in models.items():
        print(model_name)
        
        panel_model = PanelModel(model)
        panel_model.fit(X_train_df, y_train_df)
        mod_preds = panel_model.predict(X_test_df)
        
        mod_preds.name = model_name
        preds.append(mod_preds)
    
    # Merge y_df for visualization purpose
    y_hat_df = pd.concat(preds, axis=1)
    y_train_df['split'] = 'train'
    y_hat_df['split'] = 'test'
    y_df = y_train_df.append(y_hat_df, sort=False)
    
    # Evaluation Metrics
    y_hat_df = y_hat_df.drop(['split', 'y'], axis=1)
    evaluations = compute_evaluations(y_test=y_test_df, y_hat=y_hat_df, 
                                      y_train=y_train_df, metrics=metrics, 
                                      seasonality=seasonality)

    # Pipeline results
    uids_sample = button_uids_filter(y_df, size=8)
    plot_grid_series(y_df, uids_sample, models)
    evaluations.to_csv('./results/metrics.csv', index=False)
