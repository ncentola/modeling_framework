from pandas import merge, get_dummies, read_sql, to_datetime, DataFrame
from global_vars import DB_USER, query_path_beginning
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import helper_functions as hf
from imputer import Imputer
from sqlalchemy import text
from data_classes import *
import db, os, pickle

class ModelingData():

    def __init__(self, class_list, application_ids = [], train = False, time_window = False, time_window_end = 2, time_window_weeks = 12, data_name = None):
        self.class_list = class_list
        self.train = train
        self.train_horizon = '1000-01-01'
        if data_name is None:
            self.data_name = 'train_data_' + to_datetime('today').strftime('%Y-%m-%d')
        else:
            self.data_name = data_name + '_' + to_datetime('today').strftime('%Y-%m-%d')

        # if training, assume we want a chunk of app_ids in a time window
        if time_window:
            application_ids = self.get_train_app_ids(time_window_weeks=time_window_weeks, time_window_end=time_window_end)


        self.application_ids = application_ids

    def get_train_app_ids(self, time_window_weeks, time_window_end = 2):
        start_weeks = time_window_end + time_window_weeks
        start = (datetime.today() - timedelta(weeks=start_weeks)).strftime("%Y-%m-%d")
        end = (datetime.today() - timedelta(weeks=time_window_end)).strftime("%Y-%m-%d")
        self.train_horizon = end

        print('Getting application_ids between ' + start + ' and ' + end)

        _, folio_con = db.get_folio_con(DB_USER)
        _, looker_con = db.get_looker_con(DB_USER)

        app_ids_time_window_query_path = query_path_beginning + 'queries/app_ids_time_window.sql'
        with open(app_ids_time_window_query_path) as f:
            app_ids_time_window_query = f.read()
            app_ids_time_window_query = text(app_ids_time_window_query.replace('_start_', start).replace('_end_', end))

        app_ids = read_sql(app_ids_time_window_query, folio_con)

        application_ids_to_account_ids_query_path = query_path_beginning + 'queries/app_ids_to_account_ids.sql'
        income_verification_query_path = query_path_beginning + 'queries/income_verifications.sql'

        application_ids_to_account_ids_query = hf.format_query_with_dummy_var(application_ids_to_account_ids_query_path, hf.format_id_vector(list(app_ids.application_id)))
        app_account_ids = pd.read_sql(application_ids_to_account_ids_query, folio_con)
        formatted_account_ids = hf.format_id_vector(app_account_ids.account_id)

        income_verifications_query = hf.format_query_with_dummy_var(income_verification_query_path, formatted_account_ids)
        iv = pd.read_sql(income_verifications_query, looker_con)

        return list(iv.application_id)

    def build(self, target, scale_data = False, debug = False):
        modeling_data = None

        for class_name in self.class_list:
            try:
                data_class = class_name(application_ids = self.application_ids)
            except:
                print(str(type(data_class)) + ' - Failed to instantiate')

            data_class.gather_data(debug=debug, train=self.train, train_horizon=self.train_horizon)
            data_class.process_data(debug=debug, train=self.train)

            class_name_text = type(data_class).__name__

            try:
                self.__dict__[class_name_text] = data_class.processed_data
            except:
                pass

            if modeling_data is None:
                modeling_data = data_class.processed_data
            else:
                try:
                    modeling_data = merge(modeling_data, data_class.processed_data, how = 'left', on = ['account_id', 'application_id'])
                except:
                    try:
                        modeling_data = merge(modeling_data, data_class.processed_data, how = 'left', on = ['account_id'])
                    except:
                        pass

            if debug:
                print(type(data_class).__name__ + ' ' + str(len(modeling_data.index)) + ' rows')

        self.target = modeling_data[target]
        self.raw_data = modeling_data
        cols_to_drop =  [    target,
                            'app_created_date',
                            'application_created_at',
                            'initial_decision',
                            'security_statement_present_on_report',
                            'state',
                            'active_net_monthly',
                            'correct_income',
                            'net_monthly_verified_proportion',
                            'verified_income',
                            'bureauversion'
                        ]
        for col in cols_to_drop:
            try:
                modeling_data = modeling_data.drop([col], axis = 1)
            except:
                pass

        modeling_data_dummy = get_dummies(modeling_data, prefix_sep='__', drop_first = False)

        self.modeling_data = modeling_data_dummy.loc[~pd.isnull(self.target)]
        self.target = self.target.loc[~pd.isnull(self.target)]

        if scale_data:
            self.imputer = Imputer().fit(self.modeling_data)
            self.modeling_data = self.imputer.transform(self.modeling_data)

            self.scaler = StandardScaler().fit(self.modeling_data)
            self.modeling_data = DataFrame(self.scaler.transform(self.modeling_data), columns=self.modeling_data.columns)



    def remove_outliers(self, contamination = .1):
        clustering_features = ['account_id', 'reported_income', 'verified_income', 'application_intent', 'initial_decision', 'active12', 'plus30', 'hcacc13']

        clustering_data = copy(self.raw_data[clustering_features])
        # clustering_data = self.raw_data
        clustering_data = get_dummies(clustering_data, prefix_sep='__', drop_first = False)
        clustering_data = clustering_data.dropna()

        clf = IsolationForest(max_samples=100, random_state=42, contamination=contamination)
        clf.fit(clustering_data)

        pred = clf.predict(clustering_data)
        non_outliers = clustering_data.loc[pred == 1]

        outliers = clustering_data.loc[pred == -1]

        print('Removing ' + str(len(outliers.account_id)) + ' outlier rows.')

        self.target = self.target.loc[list(self.modeling_data.account_id.isin(non_outliers.account_id))]
        self.modeling_data = self.modeling_data.loc[list(self.modeling_data.account_id.isin(non_outliers.account_id))]

        self.outliers = outliers


    def save(self, dir = 'saved_data'):
            if not os.path.exists(dir):
                os.mkdir(dir)

            if '.pkl' not in self.data_name:
                self.data_name = self.data_name + '.pkl'

            # m = copy(self)

            # del m.data

            with open(os.path.join(dir, self.data_name), 'wb') as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
