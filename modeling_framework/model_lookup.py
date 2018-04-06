from sklearn.linear_model import SGDClassifier, SGDRegressor
import helper_functions as hf
import lightgbm as lgbm
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

        lookup_dict = {
            'xgboost': self.get_xgboost,
            'reg_linear': self.get_reg_linear,
            'lightgbm': self.get_lightgbm
        }

        if self.model_type not in lookup_dict.keys():
            raise ValueError('No ModelLookup entry defined for ' + self.model_type)

        lookup_dict[self.model_type]()


    def get_xgboost(self):
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

    def get_reg_linear(self):
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
                             'max_iter': 1000,
                             'n_iter': None,
                             'n_jobs': -1,
                             'penalty': 'elasticnet',
                             'power_t': 0.5,
                             'random_state': None,
                             'shuffle': True,
                             'tol': 0.001,
                             'verbose': 0,
                             'warm_start': False}

        # ------------------
        # model tuning attrs
        # ------------------

        self.param_space =  {
                                'alpha': (0.0001, 3),
                                'l1_ratio': (0, 1.0)
                            }
        self.objective_function = hf.reg_linear_objective

    def get_lightgbm(self):
        # ----------------
        # base model attrs
        # ----------------

        self.model_name = 'lightgbm_model_' + pd.to_datetime('today').strftime('%Y-%m-%d')
        # self.cores = hf.detect_cores() - 2

        if self.target_dtype == 'bool':
            objective = 'binary'
            self.tuning_metric = 'roc_auc'
            self.model = lgbm.LGBMClassifier()
            # self.tuning_metric = 'f1'
        else:
            objective = 'regression'
            self.tuning_metric = 'neg_mean_squared_error'
            self.model = lgbm.LGBMRegressor()
            # self.tuning_metric = 'neg_mean_absolute_error'

        self.best_params = { 'boosting_type': 'gbdt',
                             'class_weight': None,
                             'colsample_bytree': 1.0,
                             'learning_rate': 0.1,
                             'max_depth': -1,
                             'min_child_samples': 20,
                             'min_child_weight': 0.001,
                             'min_split_gain': 0.0,
                             'n_estimators': 100,
                             'n_jobs': -1,
                             'num_leaves': 31,
                             'objective': objective,
                             'random_state': None,
                             'reg_alpha': 0.0,
                             'reg_lambda': 0.0,
                             'silent': True,
                             'subsample': 1.0,
                             'subsample_for_bin': 200000,
                             'subsample_freq': 1}

        # ------------------
        # model tuning attrs
        # ------------------

        self.param_space =  {
                                'max_depth': (5, 10),
                                'n_estimators': (50, 125),
                                'learning_rate': (0.01, .3),
                                'min_child_weight': (0.01, 20),
                                'min_split_gain': (0.0, 0.1),
                                'colsample_bytree': (0.50, .999),
                                'reg_alpha': (0, 5),
                                'reg_lambda': (0, 10),
                                'subsample': (.2, 1)
                            }
        self.objective_function = hf.lgbm_objective
