from global_vars import query_path_beginning, DB_USER
import helper_functions as hf
from sqlalchemy import text
from json import loads
from copy import copy
import pandas as pd
import numpy as np
import db

# superclass - wrap subclass gather_data and process_data methods in try catch
class DataClass():
    def gather_data(self, debug = False, train = False, **kwargs):

        self.train_horizon = "'" + kwargs.get('train_horizon', '1000-01-01') + "'"

        if debug:
            self.gather_data_submethod(train = train)
        else:
            try:
                self.gather_data_submethod(train = train)
            except:
                print(str(type(self)) + ' - Failed to gather data')

    def process_data(self, debug = False, train = False):
        if debug:
            self.process_data_submethod(train = train)
        else:
            try:
                self.process_data_submethod(train = train)
            except:
                print(str(type(self)) + ' - Failed to process data')

# base class - should always exist
class ApplicantFeatures(DataClass):

    def __init__(self, application_ids):
        self.application_ids = application_ids
        self.formatted_application_ids = hf.format_id_vector(application_ids)

    def gather_data_submethod(self, **kwargs):

        _, folio_con = db.get_folio_con(DB_USER)
        _, rolodex_con = db.get_rolodex_con(DB_USER)

        folio_query_path = query_path_beginning + 'queries/folio.sql'
        rolodex_query_path = query_path_beginning + 'queries/rolodex.sql'

        # get folio data, use those account_ids to get rolodex data
        folio_query = hf.format_query_with_dummy_var(folio_query_path, self.formatted_application_ids)
        folio_data = pd.read_sql(folio_query, folio_con)

        self.formatted_account_ids = hf.format_id_vector(folio_data['account_id'])

        # get rolodex data
        rolodex_query = hf.format_query_with_dummy_var(rolodex_query_path, self.formatted_account_ids)
        rolodex_data = pd.read_sql(rolodex_query, rolodex_con)

        applicant_features = pd.merge(rolodex_data, folio_data, on = ['account_id'])

        self.raw_data = applicant_features

    def process_data_submethod(self, train = False):
        processed_data = copy(self.raw_data)
        processed_data = processed_data.drop(['payment_account_id'], axis = 1)
        self.processed_data = processed_data


class CreditAlerts(DataClass):

    def __init__(self, application_ids):
        self.application_ids = application_ids
        self.formatted_application_ids = hf.format_id_vector(application_ids)

    def gather_data_submethod(self, **kwargs):

        _, rolodex_con = db.get_rolodex_con(DB_USER)
        _, folio_con = db.get_folio_con(DB_USER)

        application_ids_to_account_ids_query_path = query_path_beginning + 'queries/app_ids_to_account_ids.sql'

        application_ids_to_account_ids_query = hf.format_query_with_dummy_var(application_ids_to_account_ids_query_path, self.formatted_application_ids)
        self.all_apps = pd.read_sql(application_ids_to_account_ids_query, folio_con)

        credit_alerts_query_text_path = query_path_beginning + 'queries/rolodex_credit_alerts.sql'
        credit_alerts_query = hf.format_query_with_dummy_var(credit_alerts_query_text_path, self.formatted_application_ids)

        self.raw_data = pd.read_sql(credit_alerts_query, rolodex_con)

    def process_data_submethod(self, train = False):
        processed_data = copy(self.raw_data)
        processed_data['description'] = processed_data['description'].str.replace(r'\s|-', '_').str.replace(r':|<|>', '').str.lower()
        processed_data['value'] = 1
        processed_data = processed_data.pivot_table(index=['account_id', 'application_id'], columns = 'description', values = 'value').reset_index(drop = False).rename_axis(None, axis=1)
        processed_data = pd.merge(self.all_apps, processed_data, how='left', on = ['account_id', 'application_id'])

        # if consumer alert is here, drop it
        # try:
        #     processed_data = processed_data.drop(['security_statement_present_on_report'], axis = 1)
        # except:
        #     pass

        processed_data = processed_data.fillna(0)
        self.processed_data = processed_data


class CreditReports(DataClass):

    def __init__(self, application_ids):
        self.application_ids = application_ids
        self.formatted_application_ids = hf.format_id_vector(application_ids)

    def gather_data_submethod(self, **kwargs):

        _, rolodex_con = db.get_rolodex_con(DB_USER)

        credit_reports_query_text_path = query_path_beginning + 'queries/rolodex_credit_reports.sql'
        credit_reports_query = hf.format_query_with_dummy_var(credit_reports_query_text_path, self.formatted_application_ids)

        self.raw_data = pd.read_sql(credit_reports_query, rolodex_con)

    def process_data_submethod(self, train = False):
        processed_data = copy(self.raw_data)

        processed_data['credit_error'] = ((processed_data['fico'] > 900) | (processed_data['fico'] < 0) | (processed_data['vantage_score'] < 0)).astype(int)

        processed_data.loc[(processed_data['fico'] > 900) | (processed_data['fico'] < 0), 'fico'] = np.nan
        processed_data.loc[(processed_data['vantage_score'] > 999) | (processed_data['vantage_score'] < 0), 'vantage_score'] = np.nan

        # parse and combine parsed_attributes column
        parsed_attrs = hf.fixed_parsed_attributes(processed_data['parsed_attributes'])
        parsed_attrs = parsed_attrs.apply(pd.Series)#, dtype = 'float64')
        parsed_attrs = parsed_attrs.convert_objects(convert_numeric=True)
        # parsed_attrs = parsed_attrs.replace({'Y': 1, 'N': 0})

        combined_data = pd.concat([processed_data.reset_index(drop = True), parsed_attrs.reset_index(drop = True)], axis = 1)
        colnames = list(processed_data.columns) + list(parsed_attrs.columns)

        combined_data = combined_data[colnames]
        processed_data = combined_data
        cols_to_drop = ['parsed_attributes',
                        'cbdate',
                        'unsecured_score',
                        'mixed_score',
                        'stipulated_income',
                        'credit_error',
                        'ndi']

        processed_data = hf.col_drop_try(df=processed_data, cols_to_drop=cols_to_drop)

        self.processed_data = processed_data


class FraudFlags(DataClass):

    def __init__(self, application_ids):
        self.application_ids = application_ids
        self.formatted_application_ids = hf.format_id_vector(application_ids)

    def gather_data_submethod(self, **kwargs):
        _, rolodex_con = db.get_rolodex_con(DB_USER)
        _, folio_con = db.get_folio_con(DB_USER)

        fraud_flags_query_path = query_path_beginning + 'queries/fraud_flags.sql'
        application_ids_to_account_ids_query_path = query_path_beginning + 'queries/app_ids_to_account_ids.sql'

        application_ids_to_account_ids_query = hf.format_query_with_dummy_var(application_ids_to_account_ids_query_path, self.formatted_application_ids)
        account_ids = pd.read_sql(application_ids_to_account_ids_query, folio_con).account_id
        self.formatted_account_ids = hf.format_id_vector(account_ids)

        fraud_flags_query = hf.format_query_with_dummy_var(fraud_flags_query_path, self.formatted_account_ids)
        fraud_flags = pd.read_sql(fraud_flags_query, rolodex_con)
        self.raw_data = fraud_flags

    def process_data_submethod(self, train = False):
        processed_data = copy(self.raw_data)
        # processed_data.fraud = processed_data[['rep_marked_fraud']]
        self.processed_data = processed_data[['account_id', 'fraud']]


class IncomeVerifications(DataClass):

    def __init__(self, application_ids):
        self.application_ids = application_ids
        self.formatted_application_ids = hf.format_id_vector(application_ids)

    def gather_data_submethod(self, **kwargs):
        _, looker_con = db.get_looker_con(DB_USER)
        _, folio_con = db.get_folio_con(DB_USER)

        application_ids_to_account_ids_query_path = query_path_beginning + 'queries/app_ids_to_account_ids.sql'
        income_verification_query_path = query_path_beginning + 'queries/income_verifications.sql'

        application_ids_to_account_ids_query = hf.format_query_with_dummy_var(application_ids_to_account_ids_query_path, self.formatted_application_ids)
        app_account_ids = pd.read_sql(application_ids_to_account_ids_query, folio_con)
        self.formatted_account_ids = hf.format_id_vector(app_account_ids.account_id)

        income_verifications_query = hf.format_query_with_dummy_var(income_verification_query_path, self.formatted_account_ids)
        self.raw_data = pd.read_sql(income_verifications_query, looker_con)

    def process_data_submethod(self, train = False):
        processed_data = copy(self.raw_data)
        # processed_data.fraud = processed_data[['rep_marked_fraud']]
        self.processed_data = processed_data
