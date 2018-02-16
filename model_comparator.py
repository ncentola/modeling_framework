from importlib import reload, import_module
from sklearn.metrics import roc_auc_score
from scipy import stats
from copy import copy
import data_classes
import pandas as pd
import numpy as np
import pickle
import os, sys


class ModelComparator():

    def __init__(self, model_dir_b, model_dir_a = '.'):
        self.model_dir_a = model_dir_a
        self.model_dir_b = model_dir_b

    def import_from(module, name):
        module = __import__(module, fromlist=[name])
        return getattr(module, name)

    def get_preds(self):
        with open(os.path.join(self.model_dir_a, 'model/model.pkl'), 'rb') as model_a_path:
            model_a = pickle.load(model_a_path)

        with open(os.path.join(self.model_dir_b, 'model/model.pkl'), 'rb') as model_b_path:
            model_b = pickle.load(model_b_path)

        self.model_a = model_a
        self.model_b = model_b

        starting_app_id = max(list(mc.model_a.application_ids) + list(mc.model_b.application_ids)) + 1
        application_ids = list(np.arange(starting_app_id, starting_app_id+10000))

        orig_sys_path = copy(sys.path)

        preds = {}
        for model, model_path in list(zip([model_a, model_b], [self.model_dir_a, self.model_dir_b])):

            sys.path.insert(0, os.path.join(model_path, 'model'))

            from modeling_data import ModelingData
            from scorer import Scorer
            reload(data_classes)

            # this whole class thing is super clunky but not sure how do it it any other way
            # probably not supposed to be doing things like this at all
            if hasattr(model, 'class_list'):
                classes_strings = model.class_list_text
            else:
                classes_strings = ['ApplicantFeatures',
                                   'AttributionSources',
                                   'ClarityCbb',
                                   'ClarityClearfraud',
                                   'CreditReports',
                                   'CreditAlerts',
                                   'FraudFlags',
                                   'FraudStats',
                                   'Idology',
                                   'Iovation',
                                   'IpAddresses',
                                   'Neustar',
                                   'PhoneCalls']

            classes = []
            for class_name in classes_strings:
                class_instance = getattr(import_module('data_classes'), class_name)
                classes.append(class_instance)

            m = ModelingData(class_list=classes, application_ids=application_ids)
            m.build(model_type='xgboost', target='fraud', debug = True)

            s = Scorer(model=os.path.join(model_path, 'model/model.pkl'), data=m)
            s.score()

            # remove trailing backslash
            if model_path.endswith('/'):
                model_path = model_path[:-1]

            df = pd.merge(s.preds, m.ApplicantFeatures[['account_id', 'application_id', 'initial_decision']])
            df = df.loc[(df.target) | (df.initial_decision != 'declined')]
            preds[model_path.split('/')[-1]] = df
            sys.path = copy(orig_sys_path)

        self.preds = preds

    def compare_preds(self):

        performance_comparisons = {}
        for k1, v1 in self.preds.items():
            for k2, v2 in self.preds.items():
                if k1 != k2 and (k1 + ' - ' + k2 not in performance_comparisons.keys() and k2 + ' - ' + k1 not in performance_comparisons.keys()):
                    p = {
                        'model_a': k1,
                        'model_b': k2,
                        'model_a_AUC': roc_auc_score(v1.target, v1.score),
                        'model_b_AUC': roc_auc_score(v2.target, v2.score),
                        'model_a_b_ks_stat': stats.ks_2samp(v1.score, v2.score).statistic,
                        'model_a_b_ks_p_val': stats.ks_2samp(v1.score, v2.score).pvalue
                    }

                    f = sns.distplot(v1.score, hist=False, rug=True, label=k1)
                    f = sns.distplot(v2.score, hist=False, rug=True, label=k2)

                    d = {
                        'df': p,
                        'dist_plot': f
                    }
                    performance_comparisons[k1 + ' - ' + k2] = d

        self.performance_comparisons = performance_comparisons


        
