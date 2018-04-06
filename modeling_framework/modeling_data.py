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


    def build(self, target, debug = False):
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
        # if training, filter out non-fraud auto declines
        # if training, filter out RECORDS with CONSUMER ALERTS (up for debate whether this is right or wrong)
        if self.train:
            modeling_data = modeling_data[(modeling_data[target]) | (modeling_data.initial_decision != 'declined')]
            modeling_data = modeling_data.loc[modeling_data.security_statement_present_on_report == 0]

        # do this better - something like drop where dtype is datetime
        self.target = modeling_data[target]
        cols_to_drop =  [    target,
                            'app_created_date',
                            'application_created_at',
                            'initial_decision',
                            'security_statement_present_on_report',
                            'state',
                            'active_net_monthly'
                            # 'job_overall_fraud_rate',
                            # 'isp_overall_fraud_rate',
                            # 'org_overall_fraud_rate'
                        ]
        for col in cols_to_drop:
            try:
                modeling_data = modeling_data.drop([col], axis = 1)
            except:
                pass



        modeling_data_dummy = get_dummies(modeling_data, prefix_sep='__')

        self.modeling_data = modeling_data_dummy

    def save(self, dir = '/Users/ncentola/saved_data'):
            if not os.path.exists(dir):
                os.mkdir(dir)

            if '.pkl' not in self.data_name:
                self.data_name = self.data_name + '.pkl'

            # m = copy(self)

            # del m.data

            with open(os.path.join(dir, self.data_name), 'wb') as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
