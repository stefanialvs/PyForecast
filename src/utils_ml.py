import sys
sys.path.append('./src/')

import numpy as np
import pandas as pd


from src.benchmarks import *
from src.utils_data import prepare_data, seas_dict
from src.utils_evaluation import *
from src.utils_visualization import plot_grid_series, plot_distributions

from ESRNN import ESRNN
from ESRNN.utils_configs import get_config

######################################################################
# PARSE USER INTERACTION
######################################################################

def models_metrics_filter(h, freq, models_filter, metrics_filter):
    # Parse time series arguments
    seasonality = seas_dict[freq]['seasonality']
    
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
    
    if 'ESRNN' in models_filter:
        # Instantiate ESRNN
        config = get_config(freq)
        print(config)
        models['ESRNN'] = ESRNN(max_epochs=7,
                                batch_size=config['train_parameters']['batch_size'],
                                freq_of_test=2,
                                learning_rate=float(config['train_parameters']['learning_rate']),
                                lr_scheduler_step_size=config['train_parameters']['lr_scheduler_step_size'],
                                lr_decay=config['train_parameters']['lr_decay'],
                                per_series_lr_multip=config['train_parameters']['per_series_lr_multip'],
                                gradient_clipping_threshold=config['train_parameters']['gradient_clipping_threshold'],
                                rnn_weight_decay=0.0,
                                noise_std=0.000001,
                                level_variability_penalty=config['train_parameters']['level_variability_penalty'],
                                testing_percentile=config['train_parameters']['testing_percentile'],
                                training_percentile=config['train_parameters']['training_percentile'],
                                ensemble=False,
                                max_periods=config['data_parameters']['max_periods'],
                                seasonality=config['data_parameters']['seasonality'],
                                input_size=config['data_parameters']['input_size'],
                                output_size=config['data_parameters']['output_size'],
                                frequency=config['data_parameters']['frequency'],
                                cell_type=config['model_parameters']['cell_type'],
                                state_hsize=config['model_parameters']['state_hsize'],
                                dilations=config['model_parameters']['dilations'],
                                add_nl_layer=config['model_parameters']['add_nl_layer'],
                                random_seed=config['model_parameters']['random_seed'],
                                device='cpu')
        
    # Filtered models and metrics
    models = {model_name: models[model_name] for model_name in models_filter}
    metrics = {metric_name: metrics[metric_name] for metric_name in metrics_filter}
    
    return models, metrics

def uids_filter(y_df, size=8):
    uids = y_df.index.get_level_values('unique_id').unique()
    uids_sample = np.random.choice(uids, size=8)
    return uids_sample


######################################################################
# MACHINE LEARNING PIPELINE
######################################################################

def ml_pipeline(directory, h, freq, models_filter, metrics_filter, progress_bar):
    # Set random seeds
    np.random.seed(1)

    # Filter models and metrics
    models, metrics = models_metrics_filter(h, freq,
                                            models_filter, metrics_filter)


    # Parse arguments
    seasonality = seas_dict[freq]['seasonality']
    
    # Read data
    X_train_df, y_train_df, X_test_df, y_test_df = prepare_data(directory, h)

    # Pre sort dataframes for efficiency
    X_train_df = X_train_df.set_index(['unique_id', 'ds']).sort_index()
    y_train_df = y_train_df.set_index(['unique_id', 'ds']).sort_index()
    X_test_df = X_test_df.set_index(['unique_id', 'ds']).sort_index()
    y_test_df = y_test_df.set_index(['unique_id', 'ds']).sort_index()
    
    # Fit and predict benchmark models
    preds = [y_test_df.y]
    print("\n Fitting models")
    progress_bar['maximum']=len(models.items()) + len(metrics.items())
    progress_bar['value']=0
    for model_name, model in models.items():
        print(model_name)
        
        if not model_name=='ESRNN':
            panel_model = PanelModel(model)
            panel_model.fit(X_train_df, y_train_df)
            mod_preds = panel_model.predict(X_test_df)
        else:
            Xtrain_df = X_train_df.reset_index()
            ytrain_df = y_train_df.reset_index()
            Xtest_df = X_test_df.reset_index()
            
            model.fit(Xtrain_df, ytrain_df)
            mod_preds = model.predict(Xtest_df)
        
        mod_preds.name = model_name
        preds.append(mod_preds)
        progress_bar['value']+=1
        progress_bar.update()
    
    # Merge y_df for visualization purpose
    y_hat_df = pd.concat(preds, axis=1)
    y_train_df['split'] = 'train'
    y_hat_df['split'] = 'test'
    y_df = y_train_df.append(y_hat_df, sort=False)
    
    # Evaluation Metrics
    y_hat_df = y_hat_df.drop(['split', 'y'], axis=1)
    evaluations = compute_evaluations(y_test=y_test_df, y_hat=y_hat_df, 
                                      y_train=y_train_df, metrics=metrics, 
                                      seasonality=seasonality, progress_bar=progress_bar)

    # Compute SMAPE residuals
    residuals_dict = {}
    for col in y_hat_df.columns:
        residuals_dict[col] = evaluate_panel(y_test=y_test_df, y_hat=y_hat_df[col],
                                             y_train=y_train_df, metric=smape,
                                             seasonality=seasonality)
    plot_distributions(residuals_dict)

    # Pipeline results
    plot_grid(y_df, models, h)
    evaluations.to_csv('./results/metrics.csv', index=False)

    return y_df, models

def plot_grid(y_df, models, h):
    uids_sample = uids_filter(y_df, size=8)
    plot_grid_series(y_df, uids_sample, models, plt_h=(h*5))
