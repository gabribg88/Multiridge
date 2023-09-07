import os
import math
import time
import torch
import random
import numpy as np
import pandas as pd
from typing import Tuple
from itertools import chain
from tqdm.notebook import tqdm

from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle as util_shuffle
from sklearn.model_selection import learning_curve
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.models import TrueModel

DEFAULT_RANDOM_SEED = 42

def seed_basic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def seed_everything(seed=DEFAULT_RANDOM_SEED):
    seed_basic(seed)
    
def make_regression(n_samples=100,
                    n_features=100,
                    *,
                    n_informative=10,
                    n_targets=1,
                    bias=0.0,
                    snr_db=None,
                    shuffle=True,
                    coef=False,
                    random_state=None):
    
    """Adapted from sklearn.datasets.make_regression"""

    n_informative = min(n_features, n_informative)
    generator = check_random_state(random_state)

    # Randomly generate a well conditioned input set
    X = generator.standard_normal(size=(n_samples, n_features))

    # Generate a ground truth model with only n_informative features being non
    # zeros (the other features are not correlated to y and should be ignored
    # by a sparsifying regularizers such as L1 or elastic net)
    ground_truth = np.zeros((n_features, n_targets))
    ground_truth[:n_informative, :] = 100 * generator.uniform(size=(n_informative, n_targets)) - 50

    y = np.dot(X, ground_truth) + bias

    # Add noise
    if snr_db is not None:
        snr = 10**(snr_db/10)
        noise = (np.dot(X, ground_truth)).std() / math.sqrt(snr) # compute noise standard deviation from snr from definition: SNR = var(X@theta) / var(eps),
                                                                 # see https://rdrr.io/cran/L0Learn/man/GenSynthetic.html
        y += generator.normal(scale=noise, size=y.shape)

    # Randomly permute samples and features
    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

        indices = np.arange(n_features)
        generator.shuffle(indices)
        X[:, :] = X[:, indices]
        ground_truth = ground_truth[indices]

    y = np.squeeze(y)

    if coef:
        return X, y, np.squeeze(ground_truth)

    else:
        return X, y
    
def generate_synthetic_dataset(n_samples_train,
                               n_samples_test,
                               n_features,
                               n_informative,
                               snr_db,
                               random_state):
    
    X, Y, theta_true = make_regression(n_samples=n_samples_train+n_samples_test, n_features=n_features, n_informative=n_informative,
                       n_targets=1, bias=0.0, snr_db=snr_db, random_state=random_state, coef=True)

    features = [f'x{i}' for i in range(1, n_features+1)]
    X = pd.DataFrame(data=X, columns=features)
    target = 'y'
    Y = pd.DataFrame(Y, columns=[target])

    ## Train-test splitting
    comb = pd.concat([X, Y], axis=1)
    train = comb.iloc[:n_samples_train].reset_index(drop=True).copy()
    test = comb.iloc[n_samples_train:n_samples_train+n_samples_test].reset_index(drop=True).copy()
    
    return train, test, features, target, theta_true

def check_snr(train,
              test,
              features,
              target,
              theta_true,
              snr_db):
    
    train_tmp = train.copy()
    test_tmp  = test.copy()
    comb = pd.concat([train_tmp, test_tmp], ignore_index=True)
    estimated_snr_db = 10*np.log10((comb[features]@theta_true).var() / (comb[target] - comb[features]@theta_true).var())
    assert(np.abs(snr_db - estimated_snr_db) < 1)

def create_folds(train,
                 features,
                 target,
                 num_folds,
                 shuffle=True,
                 seed=42):
    
    kf = KFold(n_splits=num_folds, shuffle=shuffle, random_state=seed)
    folds = []
    for n_fold, (train_fold_idx, valid_fold_idx) in enumerate(kf.split(train[features], train[target])):
        folds.append((train_fold_idx, valid_fold_idx))
    return folds
    
    
def optimize_baselines(train,
                       features,
                       target,
                       folds,
                       estimators=['TrueModel', 'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet'],
                       normalize=True,
                       true_coef=None,
                       true_intercept=None,
                       grid_size=100,
                       n_jobs=-1,
                       seed=42):
    
    results = dict()
    models  = dict()
    
    if 'TrueModel' in estimators:
        params = dict()
        scoring = {'r2': 'r2', 'mse': 'neg_mean_squared_error'}
        
        model = TrueModel(coef=true_coef, intercept=true_intercept)
        gs = GridSearchCV(model, params, scoring=scoring, cv=folds, verbose=0, refit='r2', return_train_score=True, n_jobs=1)
        gs.fit(train[features], train[target])
        df_results = pd.DataFrame(gs.cv_results_)
        df_results.columns = df_results.columns.str.replace('test', 'valid')
        results['TrueModel'] = df_results
        models['TrueModel']  = gs.best_estimator_
        
    if 'LinearRegression' in estimators:
        params = dict()
        scoring = {'r2': 'r2', 'mse': 'neg_mean_squared_error'}
        
        if normalize is True:
            model = LinearRegression(fit_intercept=False)
            pipe  = Pipeline([('scaler', StandardScaler()), ('model', model)])
            treg  = TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())
            gs = GridSearchCV(treg, params, scoring=scoring, cv=folds, verbose=0, refit='r2', return_train_score=True, n_jobs=1)
        else:
            model = LinearRegression(fit_intercept=False)
            gs = GridSearchCV(model, params, scoring=scoring, cv=folds, verbose=0, refit='r2', return_train_score=True, n_jobs=1)
            
        gs.fit(train[features], train[target])
        df_results = pd.DataFrame(gs.cv_results_)
        df_results.columns = df_results.columns.str.replace('test', 'valid')
        results['LinearRegression'] = df_results
        models['LinearRegression']  = gs.best_estimator_
    
    if 'Ridge' in estimators:
        params = dict()
        scoring = {'r2': 'r2', 'mse': 'neg_mean_squared_error'}
        
        if normalize is True:
            params['regressor__model__alpha'] = np.logspace(-3, 6, grid_size, endpoint=True)
            model = Ridge(fit_intercept=False)
            pipe  = Pipeline([('scaler', StandardScaler()), ('model', model)])
            treg  = TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())
            gs = GridSearchCV(treg, params, scoring=scoring, cv=folds, verbose=0, refit='r2', return_train_score=True, n_jobs=n_jobs)
        else:
            params['alpha'] = np.logspace(-3, 6, grid_size, endpoint=True)
            model = Ridge(fit_intercept=False)
            gs = GridSearchCV(model, params, scoring=scoring, cv=folds, verbose=0, refit='r2', return_train_score=True, n_jobs=n_jobs)
        
        gs.fit(train[features], train[target])
        df_results = pd.DataFrame(gs.cv_results_)
        df_results.columns = df_results.columns.str.replace('test', 'valid')
        results['Ridge'] = df_results
        models['Ridge']  = gs.best_estimator_
    
    if 'Lasso' in estimators:
        params = dict()
        scoring = {'r2': 'r2', 'mse': 'neg_mean_squared_error'}
        
        if normalize is True:
            params['regressor__model__alpha'] = np.logspace(-5, 2, grid_size, endpoint=True)
            model = Lasso(fit_intercept=False)
            pipe  = Pipeline([('scaler', StandardScaler()), ('model', model)])
            treg  = TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())
            gs = GridSearchCV(treg, params, scoring=scoring, cv=folds, verbose=0, refit='r2', return_train_score=True, n_jobs=n_jobs)
        else:
            params['alpha'] = np.logspace(-5, 2, grid_size, endpoint=True)
            model = Lasso(fit_intercept=False)
            gs = GridSearchCV(model, params, scoring=scoring, cv=folds, verbose=0, refit='r2', return_train_score=True, n_jobs=n_jobs)
        
        gs.fit(train[features], train[target])
        df_results = pd.DataFrame(gs.cv_results_)
        df_results.columns = df_results.columns.str.replace('test', 'valid')
        results['Lasso'] = df_results
        models['Lasso']  = gs.best_estimator_
    
    if 'ElasticNet' in estimators:
        params = dict()
        scoring = {'r2': 'r2', 'mse': 'neg_mean_squared_error'}
        
        if normalize is True:
            params['regressor__model__alpha'] =  np.logspace(-5, 3, grid_size, endpoint=True)
            params['regressor__model__l1_ratio'] = np.linspace(0, 1, grid_size, endpoint=True)
            model = ElasticNet(fit_intercept=False)
            pipe  = Pipeline([('scaler', StandardScaler()), ('model', model)])
            treg  = TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())
            gs = RandomizedSearchCV(treg, params, n_iter=grid_size, scoring=scoring, cv=folds, verbose=0, refit='r2', return_train_score=True, n_jobs=n_jobs, random_state=seed)
        else:
            params['alpha'] =  np.logspace(-5, 3, grid_size, endpoint=True)
            params['l1_ratio'] = np.linspace(0, 1, grid_size, endpoint=True)
            model = ElasticNet(fit_intercept=False)
            gs = RandomizedSearchCV(model, params, n_iter=grid_size, scoring=scoring, cv=folds, verbose=0, refit='r2', return_train_score=True, n_jobs=n_jobs, random_state=seed)

        gs.fit(train[features], train[target])
        df_results = pd.DataFrame(gs.cv_results_)
        df_results.columns = df_results.columns.str.replace('test', 'valid')
        results['ElasticNet'] = df_results
        models['ElasticNet']  = gs.best_estimator_
    
    return results, models

def theta_rescale(model,
                  model_name):
    if model_name != 'TrueModel':
        try:
            σy = np.sqrt(model.transformer_.var_)
            σx = np.sqrt(model.regressor_.named_steps.scaler.var_)
            scale_factor = σy / σx
            theta_rescaled = scale_factor*model.regressor_.named_steps.model.coef_
            return theta_rescaled
        except Exception as e: # quando il dataset è già normalizzato e non ho usato ne Pipelines ne TransformedTargetRegressor
            print(e)
            return model.coef_
    else:
        return model.coef_
    
def compute_baselines_results(train,
                              test,
                              features,
                              target,
                              theta_true,
                              folds,
                              models,
                              metrics):

    models_list = list(models.keys())
    columns = [f'fold{fold+1}_{split}_{metric}' for fold in range(len(folds)) for split in ['train', 'valid'] for metric in metrics.keys()] + \
              [f'{stat}_{split}_{metric}' for stat in ['mean', 'std'] for split in ['train', 'valid'] for metric in metrics.keys()] + \
              list(chain.from_iterable((f'test_{metric}_ensemble1', f'test_{metric}_ensemble2', f'test_{metric}_refit', f'train_{metric}_refit') for metric in metrics)) + \
              [f'fold{fold+1}_theta_true_distance' for fold in range(len(folds))] + \
              ['ensemble_theta_true_distance', 'refit_theta_true_distance']

    df = pd.DataFrame(data=0.0, columns=columns, index=models_list)
    df_coef = pd.DataFrame(data=0.0, columns=[f'theta{i+1}' for i in range(len(features))],
                           index=[f'{model_name}_theta_fold{n_fold+1}' for model_name in models_list for n_fold,_ in enumerate(folds)] + \
                        list(chain.from_iterable((f'{model_name}_theta_ensemble',f'{model_name}_theta_refit') for model_name in models_list)))
    df_coef.loc['theta_true'] = theta_true

    for model_name in models_list:
        train_preds_refit = models[model_name].predict(train[features])
        test_preds_refit = models[model_name].predict(test[features])
        for metric, metric_func in metrics.items():
            df.loc[model_name, f'train_{metric}_refit'] = metric_func(train[target], train_preds_refit)
            df.loc[model_name, f'test_{metric}_refit'] = metric_func(test[target], test_preds_refit)

        theta_rescaled = theta_rescale(models[model_name], model_name)
        df.loc[model_name, 'refit_theta_true_distance'] = np.linalg.norm(theta_true - theta_rescaled)
        df_coef.loc[f'{model_name}_theta_refit'] = theta_rescaled

        test_preds_ensemble = np.zeros(len(test))
        for n_fold, (train_fold_idx, valid_fold_idx) in enumerate(folds):
            train_fold = train.loc[train_fold_idx].copy()
            valid_fold = train.loc[valid_fold_idx].copy()

            model = clone(models[model_name])
            model.fit(train_fold[features], train_fold[target])
            train_fold_preds = model.predict(train_fold[features])
            valid_fold_preds = model.predict(valid_fold[features])
            test_preds_ensemble += model.predict(test[features])/len(folds)

            for metric, metric_func in metrics.items():
                df.loc[model_name, f'fold{n_fold+1}_train_{metric}'] = metric_func(train_fold[target], train_fold_preds)
                df.loc[model_name, f'fold{n_fold+1}_valid_{metric}'] = metric_func(valid_fold[target], valid_fold_preds)

            theta_fold_rescaled = theta_rescale(model, model_name)
            df.loc[model_name, f'fold{n_fold+1}_theta_true_distance'] = np.linalg.norm(theta_true - theta_fold_rescaled)
            df_coef.loc[f'{model_name}_theta_fold{n_fold+1}'] = theta_fold_rescaled
            df_coef.loc[f'{model_name}_theta_ensemble'] += theta_fold_rescaled/len(folds)

        for metric, metric_func in metrics.items():
            df.loc[model_name, f'mean_train_{metric}'] = df.loc[model_name, [f'fold{fold+1}_train_{metric}' for fold in range(len(folds))]].mean()
            df.loc[model_name, f'std_train_{metric}']  = df.loc[model_name, [f'fold{fold+1}_train_{metric}' for fold in range(len(folds))]].std(ddof=0)
            df.loc[model_name, f'mean_valid_{metric}'] = df.loc[model_name, [f'fold{fold+1}_valid_{metric}' for fold in range(len(folds))]].mean()
            df.loc[model_name, f'std_valid_{metric}']  = df.loc[model_name, [f'fold{fold+1}_valid_{metric}' for fold in range(len(folds))]].std(ddof=0)
            df.loc[model_name, f'test_{metric}_ensemble1'] = metric_func(test[target], test_preds_ensemble)
            df.loc[model_name, f'test_{metric}_ensemble2'] = metric_func(test[target], test[features].values@df_coef.loc[f'{model_name}_theta_ensemble'].values)
        df.loc[model_name, 'ensemble_theta_true_distance'] = df.loc[model_name, [f'fold{fold+1}_theta_true_distance' for fold in range(len(folds))]].mean()
    
    return df, df_coef


def plot_diagnostics(train,
                     test,
                     features,
                     target,
                     model,
                     model_name,
                     gs,
                     axes_idx,
                     fig_params,
                     metric_params,
                     share_axes=None):
    
    metric = metric_params['metric']
    metric_func = metric_params['metric_func']
    metric_label = metric_params['metric_label']
    train_color = fig_params['train_color']
    test_color = fig_params['test_color']
    if share_axes is not None:
        ax0 = plt.subplot(gs[axes_idx[0]], sharex=share_axes[0], sharey=share_axes[0])
    else:
        ax0 = plt.subplot(gs[axes_idx[0]])
    test_actual = test[target].values
    test_preds = model.predict(test[features])
    train_actual = train[target].values
    train_preds = model.predict(train[features])
    ax0.scatter(test_preds, test_actual, alpha=0.2, c=test_color, label='Test {}$= {:0.3f}$'.format(metric_label, metric_func(test_actual, test_preds)))
    ax0.scatter(train_preds, train_actual, alpha=0.6, c=train_color, label='Train {}$= {:0.3f}$'.format(metric_label, metric_func(train_actual, train_preds)))
    # ## Make square
    # ylim = ax0.get_ylim()
    # xlim = ax0.get_xlim()
    # bounds = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
    # ax0.set_xlim(bounds)
    # ax0.set_ylim(bounds)
    # ax0.set_aspect('equal', adjustable='box')
    ## Draw identity line
    # ax0.plot(bounds, bounds, lw=2, ls='--', color='k', alpha=0.5)
    ax0.set_xlabel(r"$\hat{y}$")
    ax0.set_ylabel(r"$y$")
    # ax0.set_xticks(ax0.get_yticks())
    # for label in ax0.get_xticklabels():
    #     label.set_rotation(45)
    #     label.set_ha('right')
    leg = ax0.legend(loc='best', frameon=True)
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
    ax0.set_title(f'{model_name} prediction plot')
    
    if share_axes is not None:
        ax1 = plt.subplot(gs[axes_idx[1]], sharex=share_axes[1], sharey=share_axes[1])
    else:
        ax1 = plt.subplot(gs[axes_idx[1]])
    test_residual = test_actual - test_preds
    train_residual = train_actual - train_preds
    ax1.scatter(test_preds, test_residual, alpha=0.2, c=test_color, label='Test {}$= {:0.3f}$'.format(metric_label, metric_func(test_actual, test_preds)))
    ax1.scatter(train_preds, train_residual, alpha=0.6, c=train_color, label='Train {}$= {:0.3f}$'.format(metric_label, metric_func(train_actual, train_preds)))
    #ax1.axhline(y=0, lw=2, ls='--', color='k', alpha=0.5)
    #ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlabel(r"$\hat{y}$")
    ax1.set_ylabel(r"$y - \hat{y}$")
    leg = ax1.legend(loc='best', frameon=True)
    for lh in leg.legendHandles: 
        lh.set_alpha(1)

    divider = make_axes_locatable(ax1)
    hax = divider.append_axes("right", size=1, pad=0.1, sharey=ax1)
    hax.yaxis.tick_right()
    hax.grid(False, axis="x")
    hax.hist(test_residual, bins=50, orientation="horizontal", density=True, color=test_color)
    hax.hist(train_residual, bins=50, orientation="horizontal", density=True, color=train_color)
    ax1.set_title(f'{model_name} residual plot')
    
    return (ax0, ax1)

def plot_optimization_results(base_opt_results,
                              model_name,
                              metric,
                              gs,
                              axes_idx,
                              fig_params,
                              metric_params):
    
    metric = metric_params['metric']
    metric_func = metric_params['metric_func']
    metric_label = metric_params['metric_label']
    train_color = fig_params['train_color']
    valid_color = fig_params['valid_color']
    test_color = fig_params['test_color']
    ax = plt.subplot(gs[axes_idx])
    if model_name in ['Ridge', 'Lasso']:
        param_alpha_columns = [c for c in base_opt_results[model_name].columns if 'alpha' in c][0]
        tmp = base_opt_results[model_name].set_index(param_alpha_columns).copy()
        ax.plot(tmp[f'mean_train_{metric}'].index, tmp[f'mean_train_{metric}'].values, lw=2, color=train_color, label=fr'Train {metric_label}')
        ax.plot(tmp[f'mean_valid_{metric}'].index, tmp[f'mean_valid_{metric}'].values, lw=2, color=valid_color, label=fr'Valid {metric_label}')
        max_value_valid, idxmax_value = tmp[f'mean_valid_{metric}'].max(), tmp[f'mean_valid_{metric}'].idxmax()
        max_value_train = tmp.loc[idxmax_value, f'mean_train_{metric}']
        #ax.vlines(x=idxmax_value, ymax=max_value, ymin=min_value, lw=2, ls='--', color='k', alpha=0.8, label=r'$\alpha^* = {:0.3f}$'.format(idxmax_value))
        ax.axvline(x=idxmax_value, lw=2, ls='--', color='k', alpha=0.8, label=r'$\alpha^* = {:0.3f}$'.format(idxmax_value))
        ax.scatter(idxmax_value, max_value_valid, marker='*', color=valid_color, s=200, label=r'Valid {}$(\alpha^*) = {:0.3f}$'.format(metric_label, max_value_valid))
        ax.scatter(idxmax_value, max_value_train, marker='*', color=train_color, s=200, label=r'Train {}$(\alpha^*) = {:0.3f}$'.format(metric_label, max_value_train))
        ax.set_xscale('log')
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel(fr'{metric_label}')
        ax.legend(loc='best')
    elif model_name in ['ElasticNet']:
        param_alpha_columns = [c for c in base_opt_results[model_name].columns if 'alpha' in c][0]
        param_l1_ratio_columns = [c for c in base_opt_results[model_name].columns if 'l1_ratio' in c][0]
        tmp = base_opt_results['ElasticNet'].copy()
        ax.scatter(x=tmp[param_alpha_columns].values, y=tmp[param_l1_ratio_columns].values, c=tmp[f'mean_valid_{metric}'].values, cmap='viridis', s=30)
        best_alpha = base_opt_results['ElasticNet'].sort_values(f'rank_valid_{metric}').iloc[0][param_alpha_columns]
        best_l1_ratio = base_opt_results['ElasticNet'].sort_values(f'rank_valid_{metric}').iloc[0][param_l1_ratio_columns]
        best_valid = base_opt_results['ElasticNet'].sort_values(f'rank_valid_{metric}').iloc[0][f'mean_valid_{metric}']
        best_train = base_opt_results['ElasticNet'].sort_values(f'rank_valid_{metric}').iloc[0][f'mean_train_{metric}']
        ax.scatter(best_alpha, best_l1_ratio, marker='*', color='magenta', s=100, label=r'$\alpha^* = {:0.3f}$'.format(best_alpha) +\
                                                                                         '\n' + r'$l1 ratio^* = {:0.3f}$'.format(best_l1_ratio) +\
                                                                                         '\n' + r'Valid {}$(\alpha^*, l1 ratio^*) = {:0.3f}$'.format(metric_label, best_valid) +\
                                                                                         '\n' + r'Train {}$(\alpha^*, l1 ratio^*) = {:0.3f}$'.format(metric_label, best_train))
        ax.set_xscale('log')
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel(r'$l1 ratio$')
        ax.legend(loc='best')
    ax.set_title(f'{model_name} optimization')
    return ax

def plot_theta(X_pca,
               base_theta,
               model_name,
               gs,
               axes_idx,
               share_axes=None):
    
    if share_axes is not None:
        ax = plt.subplot(gs[axes_idx], sharex=share_axes, sharey=share_axes)
    else:
        ax = plt.subplot(gs[axes_idx])
    
    for i, (name,_) in enumerate(base_theta.iterrows()):
        apex = name.split('_')[-1]
        subscript = name.split('_')[0]
        if subscript == 'LinearRegression':
            subscript = 'LR'
        elif subscript == 'ElasticNet':
            subscript = 'Enet'
        if name.find(model_name)==0:
            x = X_pca[i,0]
            y = X_pca[i,1]
            ax.scatter(x, y, c='b')
            ax.annotate(r"$\theta_{{{0}}}^{{{1}}}$".format(subscript, apex), (x+0.0005, y+0.0001), size=14)
        elif (name.find(model_name)!=0) & (name != 'theta_true'):
            pass
        else:
            x = X_pca[i,0]
            y = X_pca[i,1]
            ax.scatter(x, y, c='r')
            ax.annotate(r'$\theta^o$', (x+0.0005, y+0.0001), size=14)
    ax.margins(0.1, 0.1)
    ax.set_xlabel(r'$\theta_{pca1}$')
    ax.set_ylabel(r'$\theta_{pca2}$')
    ax.set_title(f'{model_name} theta plot')
    return ax

def plot_cv_results(base_opt_results,
                    folds,
                    model_name,
                    gs,
                    axes_idx,
                    fig_params,
                    metric_params):
    
    metric = metric_params['metric']
    metric_label = metric_params['metric_label']
    train_color = fig_params['train_color']
    valid_color = fig_params['valid_color']
    try:
        x_ticks_labels = [f'fold{i+1}' for i in range(len(folds))]
    except:
        x_ticks_labels = [f'fold{i+1}' for i in range(folds)]
    x_ticks = np.arange(len(x_ticks_labels))
    try:
        train_results = base_opt_results.loc[model_name, [f'{fold}_train_{metric}' for fold in x_ticks_labels]].values
        valid_results = base_opt_results.loc[model_name, [f'{fold}_valid_{metric}' for fold in x_ticks_labels]].values
    except:
        train_results = base_opt_results.iloc[-1][[f'{fold}_train_{metric}' for fold in x_ticks_labels]].values
        valid_results = base_opt_results.iloc[-1][[f'{fold}_valid_{metric}' for fold in x_ticks_labels]].values
    ax = plt.subplot(gs[axes_idx])
    ax.bar(x_ticks-0.2, train_results, 0.4, color=train_color, label='Train')
    ax.bar(x_ticks+0.2, valid_results, 0.4, color=valid_color, label='Valid')
    ax.set_ylabel(fr'{metric_label} score')
    ax.set_xticks(x_ticks)
    ax.set_xticklabels('')
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f'{model_name} cross-validation scores')
    ax.legend()
    for i in range(len(x_ticks_labels)):
        ax.containers[0][i].set_edgecolor('black')
        ax.containers[0][i].set_linewidth(.6)
        ax.containers[1][i].set_edgecolor('black')
        ax.containers[1][i].set_linewidth(.6)
    ax.set_axisbelow(True)
    tab=ax.table(cellText=[train_results.round(3), valid_results.round(3)],
                      rowLabels=[fr'Train {metric_label}', fr'Valid {metric_label}'],
                      colLabels=x_ticks_labels,
                      loc='bottom',
                      cellLoc='center')
    #tab.scale(1, 1.3)
    return ax

def plot_learning_curve(model,
                        model_name,
                        train,
                        features,
                        target,
                        gs,
                        axes_idx,
                        fig_params,
                        metric_params,
                        ylim=None,
                        cv=None,
                        n_jobs=1,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    
        metric = metric_params['metric']
        metric_label = metric_params['metric_label']
        train_color = fig_params['train_color']
        valid_color = fig_params['valid_color']
        ax = plt.subplot(gs[axes_idx])
        
        # compute the learning curve and store the scores on the estimator
        curve = learning_curve(model, train[features], train[target], cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_sizes_, train_scores_, test_scores_ = curve

        # compute the mean and standard deviation of the training data
        train_scores_mean_ = np.mean(train_scores_, axis=1)
        train_scores_std_ = np.std(train_scores_, axis=1)

        # compute the mean and standard deviation of the test data
        test_scores_mean_ = np.mean(test_scores_, axis=1)
        test_scores_std_ = np.std(test_scores_, axis=1)
        
        labels = (fr"Train {metric_label}", fr"Valid {metric_label}")
        curves = ((train_scores_mean_, train_scores_std_),(test_scores_mean_, test_scores_std_),)

        # Get the colors for the train and test curves
        colors = [train_color, valid_color]

        # Plot the fill betweens first so they are behind the curves.
        for idx, (mean, std) in enumerate(curves):
            # Plot one standard deviation above and below the mean
            ax.fill_between(train_sizes_, mean - std, mean + std, alpha=0.25, color=colors[idx])

        # Plot the mean curves so they are in front of the variance fill
        for idx, (mean, _) in enumerate(curves):
            ax.plot(train_sizes_, mean, "o-", color=colors[idx], label=labels[idx])
        
        ax.set_title("Learning Curve for {}".format(model_name))

        # Add the legend
        ax.legend(frameon=True, loc="best")

        # Set the axis labels
        ax.set_xlabel("Training Instances")
        ax.set_ylabel(fr'{metric_label}')
        return ax
    
    
def plot_monitoring_dashboard(train,
                              test, 
                              features, 
                              target,
                              theta_true,
                              folds,
                              base_opt_results,
                              models,
                              base_results,
                              base_theta,
                              fig_params,
                              metric_params,
                              seed,
                              savefig=None,
                              show=True):
    
    fig = plt.figure(constrained_layout=True, figsize=(25, 30))
    gs = gridspec.GridSpec(6, 5, figure=fig, width_ratios=[1,1,1,1,1.3], height_ratios=[1,1,1,1,1,1])

    ax001 = plt.subplot(gs[0, 0:2])
    comb = pd.concat([train, test], ignore_index=True)
    signal = comb[features]@theta_true
    noise = comb[target] - signal
    estimated_snr_db = 10*np.log10(signal.var() / noise.var())
    signal.plot(ax=ax001, label='Signal')
    noise.plot(ax=ax001, label='Noise')
    ax001.legend(loc='best', frameon=True)
    ax001.set_title(f'Dataset signal vs noise, SNRdb={round(estimated_snr_db)}')

    pca = PCA(n_components=2, random_state=seed)
    X_pca = pca.fit(base_theta).transform(base_theta)

    ax1_list = []
    ax3_list = []
    ax4_list = []

    ## True model
    ax02 = plot_cv_results(base_results, folds=folds, model_name='TrueModel', gs=gs, axes_idx=(0,2), fig_params=fig_params, metric_params=metric_params)
    ax03, ax04 = plot_diagnostics(train, test, features, target, model=models['TrueModel'], model_name='TrueModel', gs=gs, axes_idx=[(0,3), (0,4)], fig_params=fig_params, metric_params=metric_params)
    ax3_list.append(ax03)
    ax4_list.append(ax04)

    ## Linear Regression
    ax10 = plot_learning_curve(model=models['LinearRegression'], model_name='LinearRegression', train=train, features=features, target=target, gs=gs,
                               axes_idx=(1,0), fig_params=fig_params, metric_params=metric_params, ylim=None, cv=folds, n_jobs=10, train_sizes=np.linspace(.1, 1.0, 5))
    ax11 = plot_theta(X_pca, base_theta, model_name='LinearRegression', gs=gs, axes_idx=(1,1))
    ax12 = plot_cv_results(base_results, folds=folds, model_name='LinearRegression', gs=gs, axes_idx=(1,2), fig_params=fig_params, metric_params=metric_params)
    ax13, ax14 = plot_diagnostics(train, test, features, target, model=models['LinearRegression'], model_name='LinearRegression', gs=gs, axes_idx=[(1,3), (1,4)], fig_params=fig_params, metric_params=metric_params)
    ax1_list.append(ax11)
    ax3_list.append(ax13)
    ax4_list.append(ax14)

    # ## Ridge
    ax20 = plot_optimization_results(base_opt_results, model_name='Ridge', metric='r2', gs=gs, axes_idx=(2,0), fig_params=fig_params, metric_params=metric_params)
    ax21 = plot_theta(X_pca, base_theta, model_name='Ridge', gs=gs, axes_idx=(2,1))
    ax22 = plot_cv_results(base_results, folds=folds, model_name='Ridge', gs=gs, axes_idx=(2,2), fig_params=fig_params, metric_params=metric_params)
    ax23, ax24 = plot_diagnostics(train, test, features, target, model=models['Ridge'], model_name='Ridge', gs=gs, axes_idx=[(2,3), (2,4)], fig_params=fig_params, metric_params=metric_params)
    ax1_list.append(ax21)
    ax3_list.append(ax23)
    ax4_list.append(ax24)

    ## Lasso
    ax30 = plot_optimization_results(base_opt_results, model_name='Lasso', metric='r2', gs=gs, axes_idx=(3,0), fig_params=fig_params, metric_params=metric_params)
    ax31 = plot_theta(X_pca, base_theta, model_name='Lasso', gs=gs, axes_idx=(3,1))
    ax32 = plot_cv_results(base_results, folds=folds, model_name='Lasso', gs=gs, axes_idx=(3,2), fig_params=fig_params, metric_params=metric_params)
    ax33, ax34 = plot_diagnostics(train, test, features, target, model=models['Lasso'], model_name='Lasso', gs=gs, axes_idx=[(3,3), (3,4)], fig_params=fig_params, metric_params=metric_params)
    ax1_list.append(ax31)
    ax3_list.append(ax33)
    ax4_list.append(ax34)

    ## ElasticNet
    ax40 = plot_optimization_results(base_opt_results, model_name='ElasticNet', metric='r2', gs=gs, axes_idx=(4,0), fig_params=fig_params, metric_params=metric_params)
    ax41 = plot_theta(X_pca, base_theta, model_name='ElasticNet', gs=gs, axes_idx=(4,1))
    ax42 = plot_cv_results(base_results, folds=folds, model_name='ElasticNet', gs=gs, axes_idx=(4,2), fig_params=fig_params, metric_params=metric_params)
    ax43, ax44 = plot_diagnostics(train, test, features, target, model=models['ElasticNet'], model_name='ElasticNet', gs=gs, axes_idx=[(4,3), (4,4)], fig_params=fig_params, metric_params=metric_params)
    ax1_list.append(ax41)
    ax3_list.append(ax43)
    ax4_list.append(ax44)

    for ax in ax1_list:
        bounds = min([ax.get_xlim()[0] for ax in ax1_list]+[ax.get_ylim()[0] for ax in ax1_list]), max([ax.get_xlim()[1] for ax in ax1_list]+[ax.get_ylim()[1] for ax in ax1_list])
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        ax.set_aspect('equal', adjustable='box')

    for ax in ax3_list:
        bounds = min([ax.get_xlim()[0] for ax in ax3_list]+[ax.get_ylim()[0] for ax in ax3_list]), max([ax.get_xlim()[1] for ax in ax3_list]+[ax.get_ylim()[1] for ax in ax3_list])
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        ax.set_aspect('equal', adjustable='box')
        ax.plot(bounds, bounds, lw=2, ls='--', color='k', alpha=0.5)
        # n_x, ny = 3, 8
        # ax.set_xticks( np.linspace(*ax.get_xlim(), num=n_x+1) ) #Need 4 points to make 3 intervals
        # ax.set_yticks( np.linspace(*ax.get_ylim(), num=n_y+1) ) #Need 9 points to make 8 intervals
        # ax.tick_params(axis='x', labelrotation=45)

    for ax in ax4_list:
        bounds = min([ax.get_xlim()[0] for ax in ax4_list]+[ax.get_ylim()[0] for ax in ax4_list]), max([ax.get_xlim()[1] for ax in ax4_list]+[ax.get_ylim()[1] for ax in ax4_list])
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        ax.set_aspect('equal', adjustable='box')
        ax.axhline(y=0, lw=2, ls='--', color='k', alpha=0.5)
    
    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig, dpi = 200)
    if show is True:
        plt.show()
    else:
        plt.close()


def plot_learning_curve_multiridge(history_df,
                                   model_name,
                                   gs,
                                   axes_idx,
                                   fig_params,
                                   metric_params):
    
        metric = metric_params['metric']
        metric_label = metric_params['metric_label']
        train_color = fig_params['train_color']
        valid_color = fig_params['valid_color']
        test_color = fig_params['test_color']
        ax = plt.subplot(gs[axes_idx])

        ax.plot(history_df.index.values, history_df[f'mean_train_{metric}'].values, color=train_color, label=fr"Train {metric_label}")
        ax.plot(history_df.index.values, history_df[f'mean_valid_{metric}'].values, color=valid_color, label=fr"Valid {metric_label}")
        ax.plot(history_df.index.values, history_df[f'test_{metric}_refit'].values, color=test_color, label=fr"Test {metric_label}")
        
        ax.set_title("Learning Curves for {}".format(model_name))

        # Add the legend
        ax.legend(frameon=True, loc="best")

        # Set the axis labels
        ax.set_xlabel("Epoch")
        ax.set_ylabel(fr'{metric_label}')
        ax.set_ylim(0.0, 1.0)
        return ax

def plot_monitoring_dashboard_multiridge(train,
                                         test, 
                                         features, 
                                         target,
                                         folds,
                                         model,
                                         history,                   
                                         fig_params,
                                         metric_params,
                                         savefig=None,
                                         show=True):

    fig = plt.figure(constrained_layout=True, figsize=(20, 5))
    gs = gridspec.GridSpec(1, 4, figure=fig)
    
    ax1 = plot_learning_curve_multiridge(history, model_name='Multiridge', gs=gs, axes_idx=(0), fig_params=fig_params, metric_params=metric_params)
    ax2 = plot_cv_results(history, folds=folds, model_name='Multiridge', gs=gs, axes_idx=(1), fig_params=fig_params, metric_params=metric_params)
    ax3, ax4 = plot_diagnostics(train, test, features, target, model=model, model_name='Multiridge',
                                gs=gs, axes_idx=[(2), (3)], fig_params=fig_params, metric_params=metric_params)

    plt.tight_layout()
    if savefig is not None:
        plt.savefig(savefig, dpi = 200)
    if show is True:
        plt.show()
    else:
        plt.close()



### Figure saving function taken from:
### https://zhauniarovich.com/post/2022/2022-09-matplotlib-graphs-in-research-papers/#saving-figures
def save_fig(
        fig: matplotlib.figure.Figure, 
        fig_name: str, 
        fig_dir: str, 
        fig_fmt: str,
        fig_size: Tuple[float, float] = [6.4, 4], 
        save: bool = True, 
        dpi: int = 300,
        transparent_png = True,
    ):
    """This procedure stores the generated matplotlib figure to the specified 
    directory with the specified name and format.

    Parameters
    ----------
    fig : [type]
        Matplotlib figure instance
    fig_name : str
        File name where the figure is saved
    fig_dir : str
        Path to the directory where the figure is saved
    fig_fmt : str
        Format of the figure, the format should be supported by matplotlib 
        (additional logic only for pdf and png formats)
    fig_size : Tuple[float, float]
        Size of the figure in inches, by default [6.4, 4] 
    save : bool, optional
        If the figure should be saved, by default True. Set it to False if you 
        do not want to override already produced figures.
    dpi : int, optional
        Dots per inch - the density for rasterized format (png), by default 300
    transparent_png : bool, optional
        If the background should be transparent for png, by default True
    """
    if not save:
        return
    
    fig.set_size_inches(fig_size, forward=False)
    fig_fmt = fig_fmt.lower()
    fig_dir = os.path.join(fig_dir, fig_fmt)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    pth = os.path.join(
        fig_dir,
        '{}.{}'.format(fig_name, fig_fmt.lower())
    )
    if fig_fmt == 'pdf':
        metadata={
            'Creator' : '',
            'Producer': '',
            'CreationDate': None
        }
        fig.savefig(pth, bbox_inches='tight', metadata=metadata)
    elif fig_fmt == 'png':
        alpha = 0 if transparent_png else 1
        axes = fig.get_axes()
        fig.patch.set_alpha(alpha)
        for ax in axes:
            ax.patch.set_alpha(alpha)
        fig.savefig(
            pth, 
            bbox_inches='tight',
            dpi=dpi,
        )
    else:
        try:
            fig.savefig(pth, bbox_inches='tight')
        except Exception as e:
            print("Cannot save figure: {}".format(e))
    

def _get_gpu_mem(synchronize=True, empty_cache=True):
    return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()


def _generate_mem_hook(handle_ref, mem, idx, hook_type, exp):
    def hook(self, *args):
        if len(mem) == 0 or mem[-1]["exp"] != exp:
            call_idx = 0
        else:
            call_idx = mem[-1]["call_idx"] + 1

        mem_all, mem_cached = _get_gpu_mem()
        torch.cuda.synchronize()
        mem.append({
            'layer_idx': idx,
            'call_idx': call_idx,
            'layer_type': type(self).__name__,
            'exp': exp,
            'hook_type': hook_type,
            'mem_all': mem_all,
            'mem_cached': mem_cached
        })

    return hook


def _add_memory_hooks(idx, mod, mem_log, exp, hr):
    h = mod.register_forward_pre_hook(_generate_mem_hook(hr, mem_log, idx, 'pre', exp))
    hr.append(h)

    h = mod.register_forward_hook(_generate_mem_hook(hr, mem_log, idx, 'fwd', exp))
    hr.append(h)

    h = mod.register_full_backward_hook(_generate_mem_hook(hr, mem_log, idx, 'bwd', exp))
    hr.append(h)

def log_mem(model, inp, mem_log=None, exp=None):
    mem_log = mem_log or []
    exp = exp or f'exp_{len(mem_log)}'
    hr = []
    for idx, module in enumerate(model.modules()):
        _add_memory_hooks(idx, module, mem_log, exp, hr)

    try:
        #out = model(inp)
        #loss = out.sum()
        #loss.backward()
        loss = model(inp)
        start = time.time()
        loss.backward()
        end = time.time()
        
    finally:
        [h.remove() for h in hr]
        
        #return mem_log
        return inp.grad, mem_log[2]['mem_all'], end-start
    
# def optimize_MultiRidge(train, test, features, target, theta_true, folds, metrics, exp_params, device, normalize=True, initialization=None, decay_rate=0.999, lr_scheduler=None, grad_check=False, noise_rate=0.0):
    
#     def vec(A):
#         return A.T.flatten()
    
#     def Loss_function(λ):
#         Λ = torch.diag(λ)
#         theta_fold_hat = torch.linalg.lstsq(X_train_fold.T @ X_train_fold + N_train_fold * Λ@Λ, X_train_fold.T @ Y_train_fold)[0]
#         Y_valid_fold_hat = X_valid_fold @ theta_fold_hat
#         return (1/(2*N_valid_fold)) * torch.linalg.norm(Y_valid_fold_hat - Y_valid_fold)**2# + (γ/2) *  torch.var(λ) - ν * torch.mean(λ)

#     EXP_ID = exp_params.exp_id
#     SEED = exp_params.seed
#     if exp_params.torch_dtype == 64:
#         DTYPE = torch.float64
#     elif exp_params.torch_dtype == 32:
#         DTYPE = torch.float32
#     N = exp_params.samples_number
#     D = exp_params.features_number
#     INITIALIZATION = exp_params.initialization
#     EPOCHS = exp_params.epochs_number
#     LEARNING_RATE = exp_params.learning_rate


#     columns = [f'fold{fold+1}_{split}_{metric}' for fold in range(len(folds)) for split in ['train', 'valid'] for metric in metrics.keys()] + \
#           [f'{stat}_{split}_{metric}' for stat in ['mean', 'std'] for split in ['train', 'valid'] for metric in metrics.keys()] + \
#           list(chain.from_iterable((f'test_{metric}_ensemble1', f'test_{metric}_ensemble2', f'test_{metric}_refit', f'train_{metric}_refit') for metric in metrics)) + \
#           [f'fold{fold+1}_theta_true_distance' for fold in range(len(folds))] + \
#           ['ensemble_theta_true_distance', 'refit_theta_true_distance']

#     history_df = pd.DataFrame(data=0.0, columns=columns, index=range(EPOCHS))

#     history_coef = {epoch: pd.DataFrame(data=0.0, columns=[f'theta{i+1}' for i in range(len(features))], 
#                                         index=[f'MultiRidge_theta_fold{n_fold+1}' for n_fold,_ in enumerate(folds)] + \
#                                               ['MultiRidge_theta_ensemble', 'MultiRidge_theta_refit']) for epoch in range(EPOCHS)}

#     Id = torch.eye(D, device=device, dtype=DTYPE)
#     ones = torch.ones(D,1, device=device, dtype=DTYPE)
#     ones_ = torch.ones(D, device=device, dtype=DTYPE)
#     if initialization is None:
#         λ = torch.tensor(np.ones(D), device=device, dtype=DTYPE)# / np.sqrt(N)
#     else:
#         λ = torch.tensor(initialization, device=device, dtype=DTYPE)
    
#     Λ_history = np.zeros(shape=(EPOCHS+1, D), dtype=np.float64)
#     Λ_history[0] = λ.cpu().numpy().ravel()

#     X_train = train[features].values
#     X_test  = test[features].values
#     Y_train = train[target].values[:,np.newaxis]
#     Y_test  = test[target].values[:,np.newaxis]
    
#     if normalize is True:
#         scaler_x = StandardScaler()
#         X_train  = scaler_x.fit_transform(X_train)
#         X_test   = scaler_x.transform(X_test)
#         scaler_y = StandardScaler()
#         Y_train  = scaler_y.fit_transform(Y_train)
#         Y_test   = scaler_y.transform(Y_test)

#     X_train = torch.tensor(X_train, device=device, dtype=DTYPE)
#     X_test  = torch.tensor(X_test, device=device, dtype=DTYPE)
#     Y_train = torch.tensor(Y_train, device=device, dtype=DTYPE)
#     Y_test  = torch.tensor(Y_test, device=device, dtype=DTYPE)
#     N_train, N_test = X_train.shape[0], X_test.shape[0]
    
#     v = torch.zeros(EPOCHS+1, D, device=device, dtype=DTYPE)
#     s = torch.zeros(EPOCHS+1, D, device=device, dtype=DTYPE)
#     v_hat = torch.zeros(EPOCHS+1, D, device=device, dtype=DTYPE)
#     s_hat = torch.zeros(EPOCHS+1, D, device=device, dtype=DTYPE)
#     beta1 = 0.9
#     beta2 = 0.999
#     eps = 1e-8

#     debug = {}

#     for k in tqdm(range(EPOCHS), total=EPOCHS):

#         grad_E_cv = 0.0
#         theta_hat = torch.zeros(D,1, device=device, dtype=DTYPE)

#         kf = KFold(n_splits=len(folds), shuffle=True, random_state=SEED*(k+1))

#         Y_test_hat_ensemble = torch.zeros(N_test,1, device=device, dtype=DTYPE)

#         for n_fold, (train_fold_idx, valid_fold_idx) in enumerate(kf.split(train[features], train[target])):
#             train_fold = train.loc[train_fold_idx].copy()
#             valid_fold = train.loc[valid_fold_idx].copy()

#             X_train_fold = train_fold[features].values
#             X_valid_fold = valid_fold[features].values
#             Y_train_fold = train_fold[target].values[:,np.newaxis]
#             Y_valid_fold = valid_fold[target].values[:,np.newaxis]

#             debug[k] = {}
#             debug[k]['X_train_fold'] = X_train_fold
#             debug[k]['Y_train_fold'] = Y_train_fold
#             debug[k]['X_valid_fold'] = X_valid_fold
#             debug[k]['Y_valid_fold'] = Y_valid_fold
            
#             if normalize is True:
#                 scaler_fold_x = StandardScaler()
#                 X_train_fold  = scaler_fold_x.fit_transform(X_train_fold)
#                 X_valid_fold  = scaler_fold_x.transform(X_valid_fold)
#                 scaler_fold_y = StandardScaler()
#                 Y_train_fold  = scaler_fold_y.fit_transform(Y_train_fold)
#                 Y_valid_fold  = scaler_fold_y.transform(Y_valid_fold)


#             debug[k]['scaler_fold_x'] = scaler_fold_x
#             debug[k]['scaler_fold_y'] = scaler_fold_y

#             X_train_fold = torch.tensor(X_train_fold, device=device, dtype=DTYPE)
#             X_valid_fold = torch.tensor(X_valid_fold, device=device, dtype=DTYPE)
#             Y_train_fold = torch.tensor(Y_train_fold, device=device, dtype=DTYPE)
#             Y_valid_fold = torch.tensor(Y_valid_fold, device=device, dtype=DTYPE)
#             N_train_fold, N_valid_fold = X_train_fold.shape[0], X_valid_fold.shape[0]

#             ## model.fit()
#             Λ = torch.diag(λ)
#             theta_fold_hat = torch.linalg.lstsq(X_train_fold.T @ X_train_fold + N_train_fold * Λ@Λ, X_train_fold.T @ Y_train_fold)[0]

#             ## model.predict()
#             Y_train_fold_hat = X_train_fold @ theta_fold_hat
#             Y_valid_fold_hat = X_valid_fold @ theta_fold_hat
#             Y_test_hat_ensemble += X_test @ theta_fold_hat

#             ## gradient computation
#             R_fold = (Y_valid_fold_hat - Y_valid_fold)
#             B_fold = torch.linalg.lstsq((X_train_fold.T @ X_train_fold + N_train_fold * Λ@Λ).T, X_valid_fold.T @ R_fold @ theta_fold_hat.T)[0]
#             # λ_mean = (1/D) * (ones_.T @λ)
#             # λ_mean_vec = λ_mean * ones_
#             # S = λ - λ_mean_vec
#             # grad_var = (1/(D-1)) * (S.T - (1/D) * (S.T @ ones_) * ones_.T)
#             # grad_mean = (1/D) * ones_.T
#             # γ = 300.0
#             # ν = 1
#             grad_E_fold = -(N_train_fold/N_valid_fold) * torch.diagonal(Λ@B_fold + B_fold@Λ) #+ γ*grad_var - ν*grad_mean
            
#             if grad_check is True:
#                 assert(torch.allclose(grad_E_fold, vec(torch.autograd.functional.jacobian(Loss_function, inputs=λ))))
            
#             grad_E_cv += grad_E_fold/len(folds)

#             ## save fold metrics
#             for metric, metric_func in metrics.items():
#                 history_df.loc[k, f'fold{n_fold+1}_train_{metric}'] = metric_func(Y_train_fold.cpu().numpy(), Y_train_fold_hat.cpu().numpy())
#                 history_df.loc[k, f'fold{n_fold+1}_valid_{metric}'] = metric_func(Y_valid_fold.cpu().numpy(), Y_valid_fold_hat.cpu().numpy())
            
#             if normalize is True:
#                 σy_fold = np.sqrt(scaler_fold_x.var_)
#                 σx_fold = np.sqrt(scaler_fold_y.var_)
#                 scale_factor_fold = σy_fold / σx_fold
#                 theta_fold_rescaled = scale_factor_fold * theta_fold_hat.cpu().numpy().squeeze()
#             else:
#                 theta_fold_rescaled = theta_fold_hat.cpu().numpy().squeeze()
#             history_df.loc[k, f'fold{n_fold+1}_theta_true_distance'] = np.linalg.norm(theta_true - theta_fold_rescaled)
#             history_coef[k].loc[f'MultiRidge_theta_fold{n_fold+1}'] = theta_fold_rescaled
#             history_coef[k].loc[f'MultiRidge_theta_ensemble'] += theta_fold_rescaled/len(folds)

#         ## save aggregated statistics
#         for metric, metric_func in metrics.items():
#             history_df.loc[k, f'mean_train_{metric}'] = history_df.loc[k, [f'fold{fold+1}_train_{metric}' for fold in range(len(folds))]].mean()
#             history_df.loc[k, f'std_train_{metric}']  = history_df.loc[k, [f'fold{fold+1}_train_{metric}' for fold in range(len(folds))]].std(ddof=0)
#             history_df.loc[k, f'mean_valid_{metric}'] = history_df.loc[k, [f'fold{fold+1}_valid_{metric}' for fold in range(len(folds))]].mean()
#             history_df.loc[k, f'std_valid_{metric}']  = history_df.loc[k, [f'fold{fold+1}_valid_{metric}' for fold in range(len(folds))]].std(ddof=0)
#             history_df.loc[k, f'test_{metric}_ensemble1'] = metric_func(Y_test.cpu().numpy(), Y_test_hat_ensemble.cpu().numpy())
#             history_df.loc[k, f'test_{metric}_ensemble2'] = metric_func(Y_test.cpu().numpy(), X_test.cpu().numpy()@history_coef[k].loc['MultiRidge_theta_ensemble'].values)
#         history_df.loc[k, 'ensemble_theta_true_distance'] = history_df.loc[k, [f'fold{fold+1}_theta_true_distance' for fold in range(len(folds))]].mean()

#         ## refit an all train and compute test score
#         theta_hat_refit = torch.linalg.lstsq(X_train.T @ X_train + N_train * Λ@Λ, X_train.T @ Y_train)[0]
#         Y_train_hat_refit = X_train @ theta_hat_refit
#         Y_test_hat_refit = X_test @ theta_hat_refit

#         for metric, metric_func in metrics.items():
#             history_df.loc[k, f'train_{metric}_refit'] = metric_func(Y_train.cpu().numpy(), Y_train_hat_refit.cpu().numpy())
#             history_df.loc[k, f'test_{metric}_refit'] = metric_func(Y_test.cpu().numpy(), Y_test_hat_refit.cpu().numpy())
        
#         if normalize is True:
#             σy = np.sqrt(scaler_x.var_)
#             σx = np.sqrt(scaler_y.var_)
#             scale_factor = σy / σx
#             theta_rescaled = scale_factor * theta_hat_refit.cpu().numpy().squeeze()
#         else:
#             theta_rescaled = theta_hat_refit.cpu().numpy().squeeze()
#         history_df.loc[k, 'refit_theta_true_distance'] = np.linalg.norm(theta_true - theta_rescaled)
#         history_coef[k].loc['MultiRidge_theta_refit'] = theta_rescaled
        
#         ## logging
#         if k%1 == 0:
#             print("Epoch {}: Train R2: {:.3f}, Valid R2: {:.3f}, Test R2: {:.3f}".format(k + 1,
#                                                                                          history_df.loc[k, f'mean_train_r2'],
#                                                                                          history_df.loc[k, f'mean_valid_r2'],
#                                                                                          history_df.loc[k, f'test_r2_refit']))
#             #plt.hist(grad_E_cv.cpu().numpy())
#             #plt.show()
#         if lr_scheduler is None:
#             ## update parameters
#             ## GD step
#             #grad_E_cv += noise_rate*(torch.std(grad_E_cv)*torch.randn_like(grad_E_cv) + torch.mean(grad_E_cv))
#             #grad_E_cv += torch.sqrt(torch.tensor(0.01/(1+(k+1))**0.55)) * torch.randn_like(grad_E_cv)
#             λ = λ - LEARNING_RATE * grad_E_cv
#             ## LR decay
#             LEARNING_RATE *= decay_rate
#             #noise_rate *= decay_rate
#             # v[k+1] = beta1*v[k] + (1-beta1)*grad_E_cv
#             # s[k+1] = beta2*s[k] + (1-beta2)*torch.square(grad_E_cv)
#             # v_hat[k+1] = v[k+1] / (1-beta1**(k+1))
#             # s_hat[k+1] = s[k+1] / (1-beta2**(k+1))
#             # grad_E_cv_hat = (LEARNING_RATE*v_hat[k+1]) / (torch.sqrt(s_hat[k+1]) + eps)
#             # λ = λ - grad_E_cv_hat
#             #print(grad_E_cv*LEARNING_RATE)
#             #print(grad_E_cv_hat)
#         else:
#             lr_scheduler.update_lr(k)
#             λ = λ - lr_scheduler.learning_rate * grad_E_cv

#         Λ_history[k+1] = λ.cpu().numpy().ravel()
    
#     return history_df, history_coef, debug#, Λ_history