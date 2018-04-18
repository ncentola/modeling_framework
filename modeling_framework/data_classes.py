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

class Offers(DataClass):

    def __init__(self, ids):
        self.partial_application_ids = ids
        self.formatted_partial_application_ids = hf.format_id_vector(ids)

    def gather_data_submethod(self, **kwargs):
        _, gp_con = db.get_gp_con()

        offers_query_text_path = '../../queries/offer_selection/offers.sql'
        offers_query = hf.format_query_with_dummy_var(offers_query_text_path, self.formatted_partial_application_ids)
        offers_data = pd.read_sql(offers_query, gp_con)

        frags = hf.jsons_to_dfs(offers_data.fragments)
        frags.columns = ['fragments_' + thing for thing in frags.columns]
        frags = frags[['fragments_applicant_monthly_income_net',
                       'fragments_applicant_primary_phone_type']]

        ads = hf.jsons_to_dfs(offers_data.application_details)
        ads.columns = ['application_details_' + thing for thing in ads.columns]
        ads = ads[['application_details_applicant_annual_gross_income',
                    'application_details_applicant_annual_income_gross',
                    'application_details_applicant_bank_account_account_type',
                    'application_details_applicant_birth_date',
                    'application_details_applicant_current_employment_employment_status',
                    'application_details_applicant_housing_status',
                    'application_details_applicant_home_value',
                    'application_details_applicant_primary_phone_type',
                    'application_details_application_url',
                    'application_details_credit_rating',
                    'application_details_reason_for_loan']]

        select_details = ads.merge(frags, how='left', left_index=True, right_index=True)
        # select_details_yn = select_details.notnull().astype('int')

        offers_data = offers_data.merge(select_details, how='left', left_index=True, right_index=True)

        self.raw_data = offers_data

    def process_data_submethod(self, **kwargs):
        processed_data = copy(self.raw_data)
        processed_data['offer_selected'] = processed_data.offer_selected + ' - '+ processed_data.sel_security

        processed_data['applicant_age_years'] = ((pd.to_datetime('today') - pd.to_datetime(processed_data.application_details_applicant_birth_date)).dt.total_seconds() / 3600 / 24 / 365)
        processed_data = processed_data.drop(['sel_security', 'application_details', 'fragments', 'application_details_applicant_birth_date', 'fragments_applicant_monthly_income_net'], axis=1)
        processed_data['offer_selected'] = processed_data.offer_selected.fillna('none')

        self.processed_data = processed_data
'''
put all data classes here:
for example...

class BlahBlah(DataClass):

    def __init__(self, blah):
        self.blah = blah

    def gather_data_submethod(self):
        ## some stuff to gather data
        self.raw_data = data_that_you_gathered

    def process_data_submethod(self):
        ## do any data cleaning/processing here
        self.processed_data = data_that_you_processed
'''
