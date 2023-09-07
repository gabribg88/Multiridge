import os
import math
import torch
import random
import numpy as np
import pandas as pd
from torch import nn
from itertools import chain
from tqdm.notebook import tqdm

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class LinearLR():
    def __init__(self, initial_lr, decay):
        self.learning_rate = initial_lr
        self.decay = decay
        
    def update_lr(self, epoch):
        self.learning_rate *= self.decay

class TrueModel(RegressorMixin, BaseEstimator):
    def __init__(self, coef, intercept=None):
        self.coef = coef
        self.intercept = intercept

    def fit(self, X, y, **kwargs):
        if y is None:
            raise ValueError('requires y to be passed, but the target y is None')

        X, y = check_X_y(X, y)
        self.coef_ = self.coef
        self.intercept_ = self.intercept
        self.is_fitted_ = True

        return self

    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        if self.intercept_ is None:
            return X @ self.coef_
        else:
            return X @ self.coef_ + self.intercept_
    
    def _more_tags(self):
        return {
            'poor_score': True
        }


class MultiRidge2(RegressorMixin, BaseEstimator):
    def __init__(self,
                 lambda_vector=None,
                 folds=5,
                 shuffle=True,
                 random_state=None,
                 normalize=True, epochs=100,
                 learning_rate=100,
                 scoring='r2',
                 verbose=0,
                 save_history=True,
                 device='cpu',
                 dtype=torch.float32):
        
        self.lambda_vector = lambda_vector
        self.folds = folds
        self.shuffle = shuffle
        if self.shuffle is True:
            if not isinstance(random_state, int):
                self.random_state = 0
            else:
                self.random_state = random_state
        self.normalize = normalize
        self.epochs = epochs
        self.learning_rate = learning_rate
        if isinstance(scoring, str):
            from sklearn.metrics import get_scorer
            self.scoring = {scoring: get_scorer(scoring)._score_func}
        else:
            self.scoring = scoring
        self.verbose = verbose
        self.save_history = save_history
        if self.save_history is True:
            columns = [f'fold{fold+1}_{split}_{metric}' for fold in range(self.folds) for split in ['train', 'valid'] for metric in self.scoring.keys()] + \
                      [f'{stat}_{split}_{metric}' for stat in ['mean', 'std'] for split in ['train', 'valid'] for metric in self.scoring.keys()] + \
                      list(chain.from_iterable((f'test_{metric}_ensemble', f'test_{metric}_refit', f'train_{metric}_refit') for metric in self.scoring.keys()))
            self.history_df = pd.DataFrame(data=0.0, columns=columns, index=range(self.epochs))
        self.device = device
        self.dtype = dtype
        self.coef_ = None
            

    def fit(self, X, y, eval_set=None,  **kwargs):
        
        if y is None:
            raise ValueError('requires y to be passed, but the target y is None')

        X_train, Y_train = check_X_y(X, y)
        Y_train = Y_train[:, np.newaxis]
        X_train_copy, Y_train_copy = np.copy(X_train), np.copy(Y_train)
        self.n_samples_in_  = X_train.shape[0]
        self.n_features_in_ = X_train.shape[1]

        if eval_set is not None:
            X_test, Y_test = check_X_y(eval_set[0], eval_set[1])
            Y_test = Y_test[:, np.newaxis]
            X_test_copy, Y_test_copy = np.copy(X_test), np.copy(Y_test)

        if self.normalize is True:
            scaler_x, scaler_y = StandardScaler(), StandardScaler()
            X_train, Y_train = scaler_x.fit_transform(X_train), scaler_y.fit_transform(Y_train)
            if eval_set is not None:
                X_test, Y_test = scaler_x.transform(X_test), scaler_y.transform(Y_test)
            σx, σy = np.sqrt(scaler_x.var_), np.sqrt(scaler_y.var_)
            self.scale_factor = σy / σx
        else:
            self.scale_factor = 1.0

        X_train, Y_train = torch.tensor(X_train, device=self.device, dtype=self.dtype), torch.tensor(Y_train, device=self.device, dtype=self.dtype)
        if eval_set is not None:
            X_test, Y_test = torch.tensor(X_test, device=self.device, dtype=self.dtype), torch.tensor(Y_test, device=self.device, dtype=self.dtype)
        
        Id = torch.eye(self.n_features_in_, device=self.device, dtype=self.dtype)
        ones = torch.ones(self.n_features_in_, 1, device=self.device, dtype=self.dtype)
        if self.lambda_vector is None:
            self.lambda_vector = torch.ones(self.n_features_in_, device=self.device, dtype=self.dtype)
        else:
            self.lambda_vector = torch.tensor(self.lambda_vector, device=self.device, dtype=self.dtype)

        for k in tqdm(range(self.epochs), total=self.epochs):
            grad_E_cv = 0.0
            if eval_set is not None:
                Y_test_hat_ensemble = torch.zeros(X_test.shape[0], 1, device=self.device, dtype=self.dtype)
            
            kf = KFold(n_splits=self.folds, shuffle=self.shuffle, random_state=self.random_state*(k+1))
    
            for n_fold, (train_fold_idx, valid_fold_idx) in enumerate(kf.split(X_train_copy, Y_train_copy)):

                X_train_fold, Y_train_fold = X_train_copy[train_fold_idx], Y_train_copy[train_fold_idx]
                X_valid_fold, Y_valid_fold = X_train_copy[valid_fold_idx], Y_train_copy[valid_fold_idx]

                if self.normalize is True:
                    scaler_fold_x, scaler_fold_y = StandardScaler(), StandardScaler()
                    X_train_fold, Y_train_fold  = scaler_fold_x.fit_transform(X_train_fold), scaler_fold_y.fit_transform(Y_train_fold)
                    X_valid_fold, Y_valid_fold  = scaler_fold_x.transform(X_valid_fold), scaler_fold_y.transform(Y_valid_fold)
                    σx_fold, σy_fold = np.sqrt(scaler_fold_x.var_), np.sqrt(scaler_fold_y.var_)
                    self.scale_factor_fold = σy_fold / σx_fold
                else:
                    self.scale_factor_fold = 1.0
                
                X_train_fold, Y_train_fold = torch.tensor(X_train_fold, device=self.device, dtype=self.dtype), torch.tensor(Y_train_fold, device=self.device, dtype=self.dtype)
                X_valid_fold, Y_valid_fold = torch.tensor(X_valid_fold, device=self.device, dtype=self.dtype), torch.tensor(Y_valid_fold, device=self.device, dtype=self.dtype)
                N_train_fold, N_valid_fold = X_train_fold.shape[0], X_valid_fold.shape[0]
                
                Λ = torch.diag(self.lambda_vector)
                theta_fold_hat = torch.linalg.lstsq(X_train_fold.T @ X_train_fold + N_train_fold * Λ@Λ, X_train_fold.T @ Y_train_fold)[0]

                Y_train_fold_hat = X_train_fold @ theta_fold_hat
                Y_valid_fold_hat = X_valid_fold @ theta_fold_hat
                if eval_set is not None:
                    Y_test_hat_ensemble += X_test @ theta_fold_hat
                
                ## gradient computation
                R_fold = (Y_valid_fold_hat - Y_valid_fold)
                B_fold = torch.linalg.lstsq((X_train_fold.T @ X_train_fold + N_train_fold * Λ@Λ).T, X_valid_fold.T @ R_fold @ theta_fold_hat.T)[0]
                grad_E_fold = -(N_train_fold/N_valid_fold) * torch.diagonal(Λ@B_fold + B_fold@Λ)
                grad_E_cv += grad_E_fold/self.folds

                ## save fold metrics
                if self.save_history is True:
                    for metric, metric_func in self.scoring.items():
                        self.history_df.loc[k, f'fold{n_fold+1}_train_{metric}'] = metric_func(Y_train_fold.cpu().numpy(),#*self.scale_factor_fold,
                                                                                               Y_train_fold_hat.cpu().numpy())#*self.scale_factor_fold)
                        self.history_df.loc[k, f'fold{n_fold+1}_valid_{metric}'] = metric_func(Y_valid_fold.cpu().numpy(),#*self.scale_factor_fold,
                                                                                               Y_valid_fold_hat.cpu().numpy())#*self.scale_factor_fold)

            ## save aggregated statistics
            if self.save_history is True:
                for metric, metric_func in self.scoring.items():
                    self.history_df.loc[k, f'mean_train_{metric}'] = self.history_df.loc[k, [f'fold{fold+1}_train_{metric}' for fold in range(self.folds)]].mean()
                    self.history_df.loc[k, f'std_train_{metric}']  = self.history_df.loc[k, [f'fold{fold+1}_train_{metric}' for fold in range(self.folds)]].std(ddof=0)
                    self.history_df.loc[k, f'mean_valid_{metric}'] = self.history_df.loc[k, [f'fold{fold+1}_valid_{metric}' for fold in range(self.folds)]].mean()
                    self.history_df.loc[k, f'std_valid_{metric}']  = self.history_df.loc[k, [f'fold{fold+1}_valid_{metric}' for fold in range(self.folds)]].std(ddof=0)
                    if eval_set is not None:
                        self.history_df.loc[k, f'test_{metric}_ensemble'] = metric_func(Y_test.cpu().numpy(),#*self.scale_factor,
                                                                                        Y_test_hat_ensemble.cpu().numpy())#*self.scale_factor)

            ## refit an all train and compute test score
            theta_hat_refit = torch.linalg.lstsq(X_train.T @ X_train + self.n_samples_in_ * Λ@Λ, X_train.T @ Y_train)[0]
            Y_train_hat_refit = X_train @ theta_hat_refit
            if eval_set is not None:
                Y_test_hat_refit = X_test @ theta_hat_refit

            if self.save_history is True:
                for metric, metric_func in self.scoring.items():
                    self.history_df.loc[k, f'train_{metric}_refit'] = metric_func(Y_train.cpu().numpy(),#*self.scale_factor,
                                                                                  Y_train_hat_refit.cpu().numpy())#*self.scale_factor)
                    if eval_set is not None:
                        self.history_df.loc[k, f'test_{metric}_refit'] = metric_func(Y_test.cpu().numpy(),#*self.scale_factor,
                                                                                     Y_test_hat_refit.cpu().numpy())#*self.scale_factor)
            
            self.coef_ = self.scale_factor * theta_hat_refit.cpu().numpy().squeeze()

            ## logging
            if (self.verbose > 0) and (self.save_history is True):
                if k%self.verbose == 0:
                    text = "Epoch {}: ".format(k + 1)
                    for metric, _ in self.scoring.items():
                        text += "Train {}: {:.3f}, ".format(metric, self.history_df.loc[k, f'mean_train_{metric}'])
                        text += "Valid {}: {:.3f}, ".format(metric, self.history_df.loc[k, f'mean_valid_{metric}'])
                        if eval_set is not None:
                            text += "Test {}: {:.3f}, ".format(metric, self.history_df.loc[k, f'test_{metric}_refit'])
                    print(text[:-2])

            if isinstance(self.learning_rate, float):
                self.lambda_vector = self.lambda_vector - self.learning_rate * grad_E_cv
            else:
                self.lambda_vector = self.lambda_vector - self.learning_rate.learning_rate * grad_E_cv
                self.learning_rate.update_lr(k)
        
        self.is_fitted_ = True

        return self

    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        return X @ self.coef_

    def _more_tags(self):
        return {
            'poor_score': True
        }


class MultiRidge(RegressorMixin, BaseEstimator):
    def __init__(self,
                 lambda_vector=None,
                 folds=5,
                 shuffle=True,
                 random_state=None,
                 normalize=True, epochs=100,
                 learning_rate=100,
                 scoring='r2',
                 verbose=0,
                 save_history=True,
                 device='cpu',
                 dtype=torch.float32):
        
        self.lambda_vector = lambda_vector
        self.folds = folds
        self.shuffle = shuffle
        if self.shuffle is True:
            if not isinstance(random_state, int):
                self.random_state = 0
            else:
                self.random_state = random_state
        self.normalize = normalize
        self.epochs = epochs
        self.learning_rate = learning_rate
        if isinstance(scoring, str):
            from sklearn.metrics import get_scorer
            self.scoring = {scoring: get_scorer(scoring)._score_func}
        else:
            self.scoring = scoring
        self.verbose = verbose
        self.save_history = save_history
        if self.save_history is True:
            columns = [f'fold{fold+1}_{split}_{metric}' for fold in range(self.folds) for split in ['train', 'valid'] for metric in self.scoring.keys()] + \
                      [f'{stat}_{split}_{metric}' for stat in ['mean', 'std'] for split in ['train', 'valid'] for metric in self.scoring.keys()] + \
                      list(chain.from_iterable((f'test_{metric}_ensemble', f'test_{metric}_refit', f'train_{metric}_refit') for metric in self.scoring.keys()))
            self.history_df = pd.DataFrame(data=0.0, columns=columns, index=range(self.epochs))
        self.device = device
        self.dtype = dtype
        self.coef_ = None
            

    def fit(self, X, y, eval_set=None,  **kwargs):
        
        if y is None:
            raise ValueError('requires y to be passed, but the target y is None')

        X_train, Y_train = check_X_y(X, y)
        Y_train = Y_train[:, np.newaxis]
        X_train_copy, Y_train_copy = np.copy(X_train), np.copy(Y_train)
        self.n_samples_in_  = X_train.shape[0]
        self.n_features_in_ = X_train.shape[1]

        if eval_set is not None:
            X_test, Y_test = check_X_y(eval_set[0], eval_set[1])
            Y_test = Y_test[:, np.newaxis]
            X_test_copy, Y_test_copy = np.copy(X_test), np.copy(Y_test)

        if self.normalize is True:
            scaler_x, scaler_y = StandardScaler(), StandardScaler()
            X_train, Y_train = scaler_x.fit_transform(X_train), scaler_y.fit_transform(Y_train)
            self.scaler_x, self.scaler_y = scaler_x, scaler_y
            if eval_set is not None:
                X_test, Y_test = scaler_x.transform(X_test), scaler_y.transform(Y_test)

        X_train, Y_train = torch.tensor(X_train, device=self.device, dtype=self.dtype), torch.tensor(Y_train, device=self.device, dtype=self.dtype)
        if eval_set is not None:
            X_test, Y_test = torch.tensor(X_test, device=self.device, dtype=self.dtype), torch.tensor(Y_test, device=self.device, dtype=self.dtype)
        
        Id = torch.eye(self.n_features_in_, device=self.device, dtype=self.dtype)
        ones = torch.ones(self.n_features_in_, 1, device=self.device, dtype=self.dtype)
        if self.lambda_vector is None:
            self.lambda_vector = torch.ones(self.n_features_in_, device=self.device, dtype=self.dtype)
        else:
            self.lambda_vector = torch.tensor(self.lambda_vector, device=self.device, dtype=self.dtype)

        for k in tqdm(range(self.epochs), total=self.epochs):
            grad_E_cv = 0.0
            if eval_set is not None:
                Y_test_hat_ensemble = torch.zeros(X_test.shape[0], 1, device=self.device, dtype=self.dtype)
            
            kf = KFold(n_splits=self.folds, shuffle=self.shuffle, random_state=self.random_state*(k+1))
    
            for n_fold, (train_fold_idx, valid_fold_idx) in enumerate(kf.split(X_train_copy, Y_train_copy)):

                X_train_fold, Y_train_fold = X_train_copy[train_fold_idx], Y_train_copy[train_fold_idx]
                X_valid_fold, Y_valid_fold = X_train_copy[valid_fold_idx], Y_train_copy[valid_fold_idx]

                if self.normalize is True:
                    scaler_fold_x, scaler_fold_y = StandardScaler(), StandardScaler()
                    X_train_fold, Y_train_fold  = scaler_fold_x.fit_transform(X_train_fold), scaler_fold_y.fit_transform(Y_train_fold)
                    X_valid_fold, Y_valid_fold  = scaler_fold_x.transform(X_valid_fold), scaler_fold_y.transform(Y_valid_fold)
                
                X_train_fold, Y_train_fold = torch.tensor(X_train_fold, device=self.device, dtype=self.dtype), torch.tensor(Y_train_fold, device=self.device, dtype=self.dtype)
                X_valid_fold, Y_valid_fold = torch.tensor(X_valid_fold, device=self.device, dtype=self.dtype), torch.tensor(Y_valid_fold, device=self.device, dtype=self.dtype)
                N_train_fold, N_valid_fold = X_train_fold.shape[0], X_valid_fold.shape[0]
                
                Λ = torch.diag(self.lambda_vector)
                theta_fold_hat = torch.linalg.lstsq(X_train_fold.T @ X_train_fold + N_train_fold * Λ@Λ, X_train_fold.T @ Y_train_fold)[0]

                Y_train_fold_hat = X_train_fold @ theta_fold_hat
                Y_valid_fold_hat = X_valid_fold @ theta_fold_hat
                if eval_set is not None:
                    Y_test_hat_ensemble += X_test @ theta_fold_hat
                
                ## gradient computation
                R_fold = (Y_valid_fold_hat - Y_valid_fold)
                B_fold = torch.linalg.lstsq((X_train_fold.T @ X_train_fold + N_train_fold * Λ@Λ).T, X_valid_fold.T @ R_fold @ theta_fold_hat.T)[0]
                grad_E_fold = -(N_train_fold/N_valid_fold) * torch.diagonal(Λ@B_fold + B_fold@Λ)
                grad_E_cv += grad_E_fold/self.folds

                ## save fold metrics
                if self.save_history is True:
                    for metric, metric_func in self.scoring.items():
                        self.history_df.loc[k, f'fold{n_fold+1}_train_{metric}'] = metric_func(Y_train_fold.cpu().numpy(), Y_train_fold_hat.cpu().numpy())
                        self.history_df.loc[k, f'fold{n_fold+1}_valid_{metric}'] = metric_func(Y_valid_fold.cpu().numpy(), Y_valid_fold_hat.cpu().numpy())

            ## save aggregated statistics
            if self.save_history is True:
                for metric, metric_func in self.scoring.items():
                    self.history_df.loc[k, f'mean_train_{metric}'] = self.history_df.loc[k, [f'fold{fold+1}_train_{metric}' for fold in range(self.folds)]].mean()
                    self.history_df.loc[k, f'std_train_{metric}']  = self.history_df.loc[k, [f'fold{fold+1}_train_{metric}' for fold in range(self.folds)]].std(ddof=0)
                    self.history_df.loc[k, f'mean_valid_{metric}'] = self.history_df.loc[k, [f'fold{fold+1}_valid_{metric}' for fold in range(self.folds)]].mean()
                    self.history_df.loc[k, f'std_valid_{metric}']  = self.history_df.loc[k, [f'fold{fold+1}_valid_{metric}' for fold in range(self.folds)]].std(ddof=0)
                    if eval_set is not None:
                        self.history_df.loc[k, f'test_{metric}_ensemble'] = metric_func(Y_test.cpu().numpy(), Y_test_hat_ensemble.cpu().numpy())

            ## refit an all train and compute test score
            theta_hat_refit = torch.linalg.lstsq(X_train.T @ X_train + self.n_samples_in_ * Λ@Λ, X_train.T @ Y_train)[0]
            Y_train_hat_refit = X_train @ theta_hat_refit
            if eval_set is not None:
                Y_test_hat_refit = X_test @ theta_hat_refit

            if self.save_history is True:
                for metric, metric_func in self.scoring.items():
                    self.history_df.loc[k, f'train_{metric}_refit'] = metric_func(Y_train.cpu().numpy(), Y_train_hat_refit.cpu().numpy())
                    if eval_set is not None:
                        self.history_df.loc[k, f'test_{metric}_refit'] = metric_func(Y_test.cpu().numpy(), Y_test_hat_refit.cpu().numpy())
            
            if self.normalize is True:
                σy = np.sqrt(scaler_y.var_)
                σx = np.sqrt(scaler_x.var_)
                scale_factor = σy / σx
                self.coef_ = scale_factor * theta_hat_refit.cpu().numpy().squeeze()
            else:
                self.coef_ = theta_hat_refit.cpu().numpy().squeeze()

            ## logging
            if (self.verbose > 0) and (self.save_history is True):
                if k%self.verbose == 0:
                    text = "Epoch {}: ".format(k + 1)
                    for metric, _ in self.scoring.items():
                        text += "Train {}: {:.3f}, ".format(metric, self.history_df.loc[k, f'mean_train_{metric}'])
                        text += "Valid {}: {:.3f}, ".format(metric, self.history_df.loc[k, f'mean_valid_{metric}'])
                        if eval_set is not None:
                            text += "Test {}: {:.3f}, ".format(metric, self.history_df.loc[k, f'test_{metric}_refit'])
                    print(text[:-2])

            if isinstance(self.learning_rate, float):
                self.lambda_vector = self.lambda_vector - self.learning_rate * grad_E_cv
            else:
                self.lambda_vector = self.lambda_vector - self.learning_rate.learning_rate * grad_E_cv
                self.learning_rate.update_lr(k)
        
        self.is_fitted_ = True

        return self

    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        return X @ self.coef_

    def _more_tags(self):
        return {
            'poor_score': True
        }