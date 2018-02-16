from sklearn.linear_model import SGDClassifier, SGDRegressor
import helper_functions as hf
import xgboost as xgb
import pandas as pd

class ModelLookup():

    def __init__(self, model_type, target_col):
        self.model_type = model_type
        self.target_dtype = target_col.dtype
        self.get_model()


    def get_model(self):

        '''each lookup entry MUST set:
            - model_name
            - tuning_metric
            - best_params
           Optional, if you want to do param tuning:
            - param space
            - objective_function'''

        if self.model_type == 'xgboost':

            # ----------------
            # base model attrs
            # ----------------

            self.model_name = 'xgboost_model_' + pd.to_datetime('today').strftime('%Y-%m-%d')
            # self.cores = hf.detect_cores() - 2

            if self.target_dtype == 'bool':
                objective = 'binary:logistic'
                self.tuning_metric = 'roc_auc'
                self.model = xgb.XGBClassifier()
                # self.tuning_metric = 'f1'
            else:
                objective = 'reg:linear'
                self.tuning_metric = 'neg_mean_squared_error'
                self.model = xgb.XGBRegressor()
                # self.tuning_metric = 'neg_mean_absolute_error'

            self.best_params = { 'base_score': 0.5,
                                 'booster': 'gbtree',
                                 'colsample_bylevel': 1,
                                 'colsample_bytree': 0.61993666911880152,
                                 'gamma': 3.5007109366333236,
                                 'learning_rate': 0.042247990716033385,
                                 'max_delta_step': 0,
                                 'max_depth': 9,
                                 'min_child_weight': 5,
                                 'missing': None,
                                 'n_estimators': 124,
                                 'n_jobs': -1,
                                 'nthread': -1,
                                 'objective': objective,
                                 'random_state': 0,
                                 'reg_alpha': 0,
                                 'reg_lambda': 9,
                                 'scale_pos_weight': 1,
                                 'seed': 0,
                                 'silent': True,
                                 'subsample': 1}

            # ------------------
            # model tuning attrs
            # ------------------

            self.param_space =  {
                                    'base_score': [0.5],
                                    'max_depth': (5, 10),
                                    'n_estimators': (50, 125),
                                    'learning_rate': (0.01, .3),
                                    'min_child_weight': (1, 50),
                                    'gamma': (0, 5.0),
                                    'colsample_bytree': (0.50, .999),
                                    'reg_alpha': (0, 5),
                                    'reg_lambda': (0, 10),
                                    'subsample': (.2, 1)
                                }
            self.objective_function = hf.xgb_objective

        elif self.model_type == 'reg_linear':

            # ----------------
            # base model attrs
            # ----------------

            self.model_name = 'linear_model_' + pd.to_datetime('today').strftime('%Y-%m-%d')
            # self.cores = hf.detect_cores() - 2

            if self.target_dtype == 'bool':
                loss = 'log'
                self.tuning_metric = 'roc_auc'
                self.model = SGDClassifier()
                # self.tuning_metric = 'f1'
            else:
                loss = 'squared_loss'
                self.tuning_metric = 'neg_mean_squared_error'
                self.model = SGDRegressor()
                # self.tuning_metric = 'neg_mean_absolute_error'

            self.best_params = { 'alpha': 0.0001,
                                 'average': False,
                                 'class_weight': None,
                                 'epsilon': 0.1,
                                 'eta0': 0.0,
                                 'fit_intercept': True,
                                 'l1_ratio': 0.15,
                                 'learning_rate': 'optimal',
                                 'loss': loss,
                                 'max_iter': None,
                                 'n_iter': None,
                                 'n_jobs': -1,
                                 'penalty': 'elasticnet',
                                 'power_t': 0.5,
                                 'random_state': None,
                                 'shuffle': True,
                                 'tol': None,
                                 'verbose': 0,
                                 'warm_start': False}

            # ------------------
            # model tuning attrs
            # ------------------

            self.param_space =  {
                                    'alpha': (0.0001, 10),
                                    'l1_ratio': (0, 1.0)
                                }
            self.objective_function = hf.reg_linear_objective

        else:
            raise ValueError('No ModelLookup entry defined for ' + self.model_type)
