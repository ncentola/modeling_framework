from global_vars import DB_USER, query_path_beginning
from datetime import datetime, timedelta
from pandas import merge, get_dummies, read_sql, to_datetime
from sqlalchemy import text
from data_classes import *
import db, os, pickle

class ModelingData():

    def __init__(self, class_dict, ids = [], train = False):
        self.class_dict = class_dict
        self.train = train

        self.ids = ids


    def build(self, target, debug = False, cols_to_drop = []):
        modeling_data = None

        for class_name in self.class_dict:
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
                    modeling_data = merge(modeling_data, data_class.processed_data, how = 'left', on = ['account_id', 'application_id'])
                except:
                    try:
                        modeling_data = merge(modeling_data, data_class.processed_data, how = 'left', on = ['account_id'])
                    except:
                        pass

            if debug:
                print(type(data_class).__name__ + ' ' + str(len(modeling_data.index)) + ' rows')

        self.target = modeling_data[target]

        for col in cols_to_drop:
            try:
                modeling_data = modeling_data.drop([col], axis = 1)
            except:
                pass

        modeling_data_dummy = get_dummies(modeling_data, prefix_sep='__')

        self.modeling_data = modeling_data_dummy

    def save(self, dir = '~/saved_data'):
            if not os.path.exists(dir):
                os.mkdir(dir)

            if '.pkl' not in self.data_name:
                self.data_name = self.data_name + '.pkl'


            with open(os.path.join(dir, self.data_name), 'wb') as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
