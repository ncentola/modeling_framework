from datetime import datetime, timedelta
from data_classes import *
from warnings import warn
import db, os, pickle
import pandas as pd

class ModelingData():

    def __init__(self, class_dict, ids = [], train = False, time_window=False, time_window_end = 2, time_window_weeks = 12):
        self.class_dict = class_dict
        self.train = train
        self.train_horizon = '1000-01-01'

        if time_window:
            ids = self.get_train_ids(time_window_weeks=time_window_weeks, time_window_end=time_window_end)
        self.ids = ids

    def get_train_ids(self, time_window_weeks, time_window_end = 2):
        start_weeks = time_window_end + time_window_weeks
        start = (datetime.today() - timedelta(weeks=start_weeks)).strftime("%Y-%m-%d")
        end = (datetime.today() - timedelta(weeks=time_window_end)).strftime("%Y-%m-%d")
        self.train_horizon = end

        print('Getting application_ids between ' + start + ' and ' + end)

        _, gp_con = db.get_gp_con()

        ids_time_window_query = "CREATE TEMP TABLE p_app_ids AS (SELECT partial_application_id FROM chicago.partial_applications WHERE created_at BETWEEN '_start_' AND '_end_')"
        ids_time_window_query = text(ids_time_window_query.replace('_start_', start).replace('_end_', end))

        gp_con.execute(text('DROP TABLE IF EXISTS p_app_ids'))
        gp_con.execute(ids_time_window_query)
        ids = pd.read_sql(text('SELECT * FROM p_app_ids'), gp_con)
        return ids.partial_application_id

    def build(self, target, debug = False, cols_to_drop = []):
        modeling_data = None

        for class_name in self.class_dict.keys():
            join_keys = self.class_dict[class_name]
            try:
                data_class = class_name(ids = self.ids)
            except:
                print(str(type(data_class)) + ' - Failed to instantiate')

            data_class.gather_data(debug=debug, train=self.train)
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
                    modeling_data = pd.merge(modeling_data, data_class.processed_data, how = 'left', on = join_keys)
                except:
                    warn('Unable to join '+class_name_text)

            if debug:
                print(type(data_class).__name__ + ' ' + str(len(modeling_data.index)) + ' rows')

        self.target = modeling_data[target]

        cols_to_drop = cols_to_drop + [target]
        for col in cols_to_drop:
            try:
                modeling_data = modeling_data.drop([col], axis = 1)
            except:
                pass

        modeling_data_dummy = pd.get_dummies(modeling_data, prefix_sep='__')

        self.modeling_data = modeling_data_dummy

    def save(self, dir = '~/saved_data'):
            if not os.path.exists(dir):
                os.mkdir(dir)

            if '.pkl' not in self.data_name:
                self.data_name = self.data_name + '.pkl'


            with open(os.path.join(dir, self.data_name), 'wb') as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
