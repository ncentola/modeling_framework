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
