from global_vars import query_path_beginning
from modeling_data import ModelingData
from sklearn.cluster import KMeans
from skopt import gp_minimize
import helper_functions as hf
from data_classes import *
from copy import copy
import xgboost as xgb
import pandas as pd
import numpy as np
import pickle, os, pip
from model_lookup import ModelLookup

class ModelWrapper(object):
    def __init__(self, data, model_type):

        if isinstance(data, ModelingData):
            self.data = data
        else:
            try:
                if '.pkl' not in data:
                    data = os.path.join(data, '.pkl')
                with open(data, 'rb') as pickle_file:
                    self.data = pickle.load(pickle_file)
            except:
                raise ValueError('Must pass in ModelingData object or valid path to a saved model folder')

        if len(list(self.data.modeling_data.columns)) > 1000:
            print('Probably an error - ' + str(len(list(self.data.modeling_data.columns))) + ' columns')

        self.application_ids = self.data.modeling_data.application_id
        self.class_list_text = [str(x) for x in self.data.class_list]
        # self.cores = hf.detect_cores() - 2

        # if there is a scaler attribute present, set it as train_scaler here
        try:
            self.train_imputer = self.data.imputer
            self.train_scaler = self.data.scaler
        except:
            pass

        self.model_type = model_type
        self.model = ModelLookup(self.model_type, self.data.target)

    def tune_hyperparams(self, n_calls = 100, verbose = False):

        X, y = self.data.modeling_data, self.data.target

        hf.set_objective_vars(model_in=self.model.model, X_in=X, y_in=y, tuning_metric_in=self.model.tuning_metric)
        gp_result = gp_minimize(self.model.objective_function, dimensions=list(self.model.param_space.values()), n_calls=n_calls, random_state=0, verbose = verbose, n_jobs =-1)

        tuned_params = dict(list(zip(self.model.param_space.keys(), gp_result.x)))
        default_params = copy(self.model.best_params)

        for k, v in list(zip(tuned_params.keys(), tuned_params.values())):
            default_params[k] = v

        default_params['seed'] = 0
        # default_params['nthread'] = n_cores

        self.gp_result = gp_result

    def cluster(self, n_clusters=3):
        cluster_data = copy(self.data.modeling_data)

        cluster_data['idology'] = cluster_data.account_id.isin(self.data.Idology.account_id)
        cluster_data['neustar'] = cluster_data.account_id.isin(self.data.Neustar.account_id)
        cluster_data['iovation'] = cluster_data.account_id.isin(self.data.Iovation.account_id)
        cluster_data['cbb'] = cluster_data.account_id.isin(self.data.ClarityCbb.account_id)
        cluster_data['clearfraud'] = cluster_data.account_id.isin(self.data.ClarityClearfraud.account_id)
        # cluster_data['job'] = pd.isnull(cluster_data.job_overall_fraud_rate)
        # cluster_data['phone_calls'] = cluster_data.account_id.isin(self.data.PhoneCalls.account_id)
        cluster_data['income_verification'] = ~pd.isnull(self.data.ApplicantFeatures.income_verification_method)

        cluster_features = ['idology', 'neustar', 'iovation', 'cbb', 'clearfraud', 'income_verification']
        matrix_data = cluster_data[cluster_features].replace({True: 1, False:0}).as_matrix()

        keep_clustering = True
        i = 0
        while keep_clustering:
            cluster_model = KMeans(n_clusters=n_clusters, random_state=None, max_iter=1000, n_init=3).fit(matrix_data)
            centers = pd.DataFrame(cluster_model.cluster_centers_, columns=cluster_features)
            keep_clustering = centers.apply(max, axis = 0).min() < 0.66
            i += 100

        self.cluster_labels = cluster_model.labels_
        self.cluster_centers_df = centers
        self.cluster_model = cluster_model


    def fit(self):
        if hasattr(self, 'cluster_labels'):
            cluster_labels = self.cluster_labels
        else:
            cluster_labels = [0] * len(self.data.modeling_data.index)

        fit_model = {}
        train_columns = {}
        for cluster in np.unique(cluster_labels):
            cluster_data = self.data.modeling_data.loc[cluster_labels == cluster]
            cluster_data = cluster_data.loc[:, pd.notnull(cluster_data).sum()>=len(cluster_data)*.66]

            train_data = copy(self.data.modeling_data)
            train_data = train_data[list(cluster_data.columns)]

            train_label = self.data.target.loc[cluster_labels == cluster]

            train_data = train_data.drop(['application_id', 'account_id'], axis = 1)
            print(train_data.shape)

            train_columns[cluster] = list(train_data.columns)
            keys_that_want_to_be_set = {k: self.model.best_params[k] for k in self.model.model.get_params().keys()}

            self.model.model.set_params(**keys_that_want_to_be_set)
            self.model.model.fit(X=train_data.as_matrix(), y=train_label.as_matrix())

            fit_model[cluster] = copy(self.model.model)

        self.train_columns = train_columns
        self.fit_model = fit_model

    def save(self, package_name = 'model', dir = '/Users/ncentola/saved_models'):
            package_contents_dir = os.path.join(dir, self.model_name, package_name)

            if not os.path.exists(dir):
                os.mkdir(dir)

            if not os.path.exists(os.path.join(dir, self.model_name)):
                os.mkdir(os.path.join(dir, self.model_name))

            if not os.path.exists(package_contents_dir):
                os.mkdir(package_contents_dir)

            # save packages in env to requirements.txt
            installed_packages = pip.get_installed_distributions()
            with open('requirements.txt', 'w') as f:
                for thing in installed_packages:
                    f.write("%s\n" % str(thing).replace(' ', '=='))

            # define list of files that need to be in the root dir
            root_dir_files = ['README.md', 'container_setup.sh', 'setup.py', 'requirements.txt']
            mv_files_text = ' '.join([os.path.join(package_contents_dir, file) for file in root_dir_files])

            # copy all files to the package_contents_dir which will be distributed as the model 'package'
            os.system('cp ./* ' + package_contents_dir)
            os.system('cp -r ' + query_path_beginning + 'queries ' + package_contents_dir)
            os.system('mv ' + mv_files_text + ' ' + os.path.join(dir, self.model_name))
            os.system('chmod +x ' + os.path.join(dir, self.model_name) + '/container_setup.sh')

            m = copy(self)

            del m.data

            with open(os.path.join(dir, self.model_name, package_name, 'model.pkl'), 'wb') as output:
                pickle.dump(m, output, pickle.HIGHEST_PROTOCOL)
