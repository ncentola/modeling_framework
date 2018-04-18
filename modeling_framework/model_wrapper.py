from modeling_data import ModelingData
from model_lookup import ModelLookup
from sklearn.cluster import KMeans
from skopt import gp_minimize
import helper_functions as hf
from data_classes import *
from copy import copy
from time import time
import xgboost as xgb
import pandas as pd
import numpy as np
import pickle, os, pip
import boto3

class ModelWrapper(object):
    def __init__(self, data, model_type, model_name):

        self.data = hf.read_data(data)

        if len(list(self.data.modeling_data.columns)) > 1000:
            print('Probably an error - ' + str(len(list(self.data.modeling_data.columns))) + ' columns')

        self.ids = self.data.ids
        self.model_name = model_name
        # self.class_list_text = [str(x) for x in self.data.class_list]
        # self.cores = hf.detect_cores() - 2

        self.model_type = model_type
        self.model_lookup = ModelLookup(self.model_type, self.data.target)
        self.data.target = self.model_lookup.target_col

    def tune_hyperparams(self, n_calls = 100, verbose = False):

        X, y = self.data.modeling_data, self.data.target

        hf.set_objective_vars(model_in=self.model_lookup.model, X_in=X, y_in=y, tuning_metric_in=self.model_lookup.tuning_metric, num_offers_in = X.num_offers)
        gp_result = gp_minimize(self.model_lookup.objective_function, dimensions=list(self.model_lookup.param_space.values()), n_calls=n_calls, random_state=42, verbose = verbose, n_jobs =-1)

        tuned_params = dict(list(zip(self.model_lookup.param_space.keys(), gp_result.x)))
        default_params = copy(self.model_lookup.best_params)

        for k, v in list(zip(tuned_params.keys(), tuned_params.values())):
            default_params[k] = v

        self.model_lookup.best_params = default_params
        self.gp_result = gp_result

    def cluster(self, n_clusters=5):
        df = copy(self.data.modeling_data)
        cluster_features = [x for x in list(df.columns) if 'application_details_' in x or 'fragments' in x]
        cluster_data = df[cluster_features]
        matrix_data = cluster_data.notnull().astype('int').as_matrix()

        cluster_model = KMeans(n_clusters=n_clusters, random_state=None, max_iter=1000, n_init=n_clusters).fit(matrix_data)
        centers = pd.DataFrame(cluster_model.cluster_centers_, columns=cluster_features)

        self.cluster_labels = cluster_model.labels_
        self.cluster_centers_df = centers
        self.cluster_model = cluster_model

    def fit_cluster(self):
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

            train_label = self.data.target

            train_data = train_data.drop(['application_id', 'account_id'], axis = 1)
            print(train_data.shape)

            train_columns[cluster] = list(train_data.columns)
            if self.model_type == 'xgboost':
                data = xgb.DMatrix(train_data, label = self.data.target)
                fit_model[cluster] = xgb.train(params=self.model_lookup.best_params, dtrain = data, num_boost_round=self.model_lookup.best_params['n_estimators'])
            else:
                keys_that_want_to_be_set = {k: self.model_lookup.best_params[k] for k in self.model_lookup.model.get_params().keys()}

                self.model_lookup.model.set_params(**keys_that_want_to_be_set)
                self.model_lookup.model.fit(X=train_data.as_matrix(), y=train_label.as_matrix())

                fit_model[cluster] = copy(self.model_lookup.model)

        self.train_columns = train_columns
        self.fit_model = fit_model

    def fit(self, cols_to_drop=[]):

        train_data = copy(self.data.modeling_data)
        train_data = train_data[list(train_data.columns)]

        train_label = self.model_lookup.target_col

        train_data = train_data.drop(cols_to_drop, axis = 1)
        print(train_data.shape)

        train_columns = list(train_data.columns)
        if self.model_type == 'xgboost':
            data = xgb.DMatrix(train_data, label = train_label)
            fit_model = xgb.train(params=self.model_lookup.best_params, dtrain = data, num_boost_round=self.model_lookup.best_params['n_estimators'])
        else:
            keys_that_want_to_be_set = {k: self.model_lookup.best_params[k] for k in self.model_lookup.model.get_params().keys()}

            self.model_lookup.model.set_params(**keys_that_want_to_be_set)
            self.model_lookup.model.fit(X=train_data, y=train_label)

            fit_model = copy(self.model_lookup.model)

        self.train_columns = train_columns
        self.fit_model = fit_model

    def save(self, package_name = 'model', dir = '~/saved_models'):
            self.package_contents_dir = os.path.join(dir, self.model_name, package_name)
            self.model_dir = os.path.join(dir, self.model_name)

            if not os.path.exists(dir):
                os.mkdir(dir)

            if not os.path.exists(self.model_dir):
                os.mkdir(self.model_dir)

            if not os.path.exists(self.package_contents_dir):
                os.mkdir(self.package_contents_dir)

            # save packages in env to requirements.txt
            installed_packages = pip.get_installed_distributions()
            with open('requirements.txt', 'w') as f:
                for thing in installed_packages:
                    f.write("%s\n" % str(thing).replace(' ', '=='))

            # define list of files that need to be in the root dir
            root_dir_files = ['README.md', 'container_setup.sh', 'setup.py', 'requirements.txt']
            mv_files_text = ' '.join([os.path.join(self.package_contents_dir, file) for file in root_dir_files])

            # copy all files to the self.package_contents_dir which will be distributed as the model 'package'
            os.system('cp ./* ' + self.package_contents_dir)
            os.system('cp -r queries ' + self.package_contents_dir)
            os.system('mv ' + mv_files_text + ' ' + self.model_dir)
            os.system('chmod +x ' + self.model_dir + '/container_setup.sh')

            m = copy(self)

            del m.data

            path_to_model_pkl = os.path.join(dir, self.model_name, package_name, 'model.pkl')

            with open(path_to_model_pkl, 'wb') as output:
                pickle.dump(m, output, pickle.HIGHEST_PROTOCOL)


    def to_s3(self, s3_bucket_name):
        local_directory = self.model_dir
        destination = list(filter(None, local_directory.split('/')))[-1]
        bucket = s3_bucket_name

        s3 = boto3.client('s3')

        for root, dirs, files in os.walk(local_directory):

            for filename in files:

                # construct the full local path
                local_path = os.path.join(root, filename)

                # construct the full s3 path
                relative_path = os.path.relpath(local_path, local_directory)
                s3_path = os.path.join(destination, relative_path)

                print("Uploading %s..." % s3_path)
                s3.upload_file(
                    Filename    = local_path,
                    Bucket      = bucket,
                    Key         = s3_path,
                    ExtraArgs   = {'ServerSideEncryption': 'AES256'},
                )

        with open('.model-version', 'w') as version_file:
            version_file.write(os.path.join(destination, 'model/model.pkl'))
