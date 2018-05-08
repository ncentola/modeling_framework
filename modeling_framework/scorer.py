import helper_functions as hf
from numpy import nan, unique
from xgboost import DMatrix
import pickle, os
from model_wrapper import ModelWrapper
from copy import copy
import pandas as pd
import warnings


class Scorer(object):

    def __init__(self, model, data):

        # if passed in a model object, set to self.model, else assume path was passed in and go load it
        self.model = hf.read_model(model)
        self.data = hf.read_data(data)

        # if accidently scoring app_ids the model was trained on, throw error
        if data.modeling_data.ids.isin(self.model.ids).any():
            warnings.warn("You're trying to score applications used in model training")


        self.data = data

    def score(self, calc_contribs = True):

        model = self.model.fit_model
        scoring_data = copy(self.data.modeling_data)
        target = copy(self.data.target)

        train_columns = self.model.train_columns

        missing_cols = hf.setdiff(train_columns, list(scoring_data.columns))
        extra_cols = hf.setdiff(list(scoring_data.columns), train_columns)

        for col in missing_cols:
            if '__' in col:
                scoring_data[col] = 0
            else:
                scoring_data[col] = nan

        try:
            scoring_data = scoring_data.drop(extra_cols, axis = 1)
            #print('Dropping ' + ', '.join(extra_cols))
        except:
            pass

        # reorder the columns to same order trained on
        scoring_data = scoring_data[train_columns]
        self.scoring_data = scoring_data
        xgb_data = DMatrix(scoring_data, label = self.data.target)

        if calc_contribs:

            contribs_df = copy(scoring_data[['account_id', 'application_id']])

            c = model.predict(xgb_data, pred_contribs = True)
            c = pd.DataFrame.from_records(c, columns = list(scoring_data.columns) + ['bias'])

            contribs_df = pd.concat([contribs_df.reset_index(drop=True), c.reset_index(drop=True)], axis = 1)
            contribs = contribs_df
            self.raw_contribs = pd.concat(list(contribs.values()))

            # rearrange raw_contribs
            ordered_contribs_list = []
            for index, row in self.raw_contribs.iterrows():
                clean_row = row.drop(['account_id', 'application_id'])

                pos_contribs = clean_row.loc[clean_row > 0].sort_values(ascending = False)
                pos_contribs_ordered = [{'feature': i, 'contribution': c}  for i, c in list(zip(pos_contribs.index, pos_contribs))]

                neg_contribs = clean_row.loc[clean_row < 0].sort_values(ascending = True)
                neg_contribs_ordered = [{'feature': i, 'contribution': c}  for i, c in list(zip(neg_contribs.index, neg_contribs))]


                ordered_contribs_list.append({'account_id': row.account_id,
                                              'application_id': row.application_id,
                                              'positive_contributions_ordered': pos_contribs_ordered,
                                              'negative_contributions_ordered': neg_contribs_ordered
                                             })

            self.ordered_contribs = pd.DataFrame.from_dict(ordered_contribs_list)



        preds_df = copy(scoring_data[['account_id', 'application_id']])

        # this needs to be better but not sure. Basically every predict method has it's own idiosyncrocasies and need to be handled separately (for now)
        if self.model.model_type == 'lightgbm':
            preds_df['score'] = model.predict_proba(scoring_data.as_matrix())[:,1]
        elif self.model.model_type == 'xgboost':
            preds_df['score'] = model.predict(xgb_data)
        else:
            preds_df['score'] = model.predict(scoring_data.as_matrix())

        preds_df['target'] = target

        preds = preds_df

        self.preds = pd.concat(list(preds.values()))

    def write_db(self, db_con, schema):
        '''
        do some stuff to write results to a database
        example:

        fraud_scores = pd.merge(self.preds, self.data.ApplicantFeatures[['account_id', 'application_id', 'state']])
        fraud_scores = pd.merge(fraud_scores, self.ordered_contribs, on=['account_id', 'application_id'])
        fraud_scores.rename(columns={'score': 'fraud_score'}, inplace=True)

        fraud_scores['created_at'] = pd.to_datetime('now').strftime('%Y-%m-%d %H:%M:%S')

        db_records = find_or_create_model_records(model=self.model, db_con=db_con, schema=schema)

        fraud_scores = pd.merge(fraud_scores, db_records[['cluster', 'model_id']], on = ['cluster'])
        fraud_scores = fraud_scores.drop(['target', 'cluster'], axis = 1)

        existing_records = pd.read_sql_table(table_name='fraud_scores', con=db_con, schema=schema)

        new_fraud_scores = find_new_records(fraud_scores, existing_records, ['account_id', 'application_id', 'model_id', 'state'])


        print('Writing ' + str(len(new_fraud_scores.index)) + ' rows...')
        new_fraud_scores.to_sql('fraud_scores', con=db_con, schema=schema, if_exists='append', index=False)
        self.db_table = new_fraud_scores

        '''
        pass
