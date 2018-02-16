from helper_functions import setdiff, find_or_create_model_records, find_new_records
from numpy import nan, unique
from xgboost import DMatrix
import json, pickle, os
from model import Model
from copy import copy
import pandas as pd
import warnings


class Scorer(object):

    def __init__(self, model, data):

        # if passed in a model object, set to self.model, else assume path was passed in and go load it
        if isinstance(model, Model):
            self.model = model
        else:
            try:
                if '.pkl' not in model:
                    model = os.path.join(model, 'model.pkl')

                with open(model, 'rb') as pickle_file:
                    self.model = pickle.load(pickle_file)
            except:
                raise ValueError('Must pass in Model object or valid path to a saved model folder')

        # if accidently scoring app_ids the model was trained on, throw error
        if data.modeling_data.application_id.isin(self.model.application_ids).any():
            warnings.warn("You're trying to score applications used in model training")

        if self.model.model_type == 'reg_linear':
            data = self.model.train_imputer.transform(data)
            data = self.model.train_scaler.transform(data)

        self.data = data

    def score(self, calc_contribs = True):

        if hasattr(self.model, 'cluster_model'):
            cluster_model = self.model.cluster_model

            cluster_data = copy(self.data.modeling_data)

            try:
                cluster_data['idology'] = cluster_data.account_id.isin(self.data.Idology.account_id)
            except AttributeError:
                cluster_data['idology'] = 0

            try:
                cluster_data['neustar'] = cluster_data.account_id.isin(self.data.Neustar.account_id)
            except AttributeError:
                cluster_data['neustar'] = 0

            try:
                cluster_data['iovation'] = cluster_data.account_id.isin(self.data.Iovation.account_id)
            except AttributeError:
                cluster_data['iovation'] = 0

            try:
                cluster_data['cbb'] = cluster_data.account_id.isin(self.data.ClarityCbb.account_id)
            except AttributeError:
                cluster_data['cbb'] = 0

            try:
                cluster_data['clearfraud'] = cluster_data.account_id.isin(self.data.ClarityClearfraud.account_id)
            except AttributeError:
                cluster_data['clearfraud'] = 0

            # cluster_data['job'] = pd.isnull(cluster_data.job_overall_fraud_rate)
            # cluster_data['phone_calls'] = cluster_data.account_id.isin(self.data.PhoneCalls.account_id)
            cluster_data['income_verification'] = ~pd.isnull(self.data.ApplicantFeatures.income_verification_method)

            cluster_features = ['idology', 'neustar', 'iovation', 'cbb', 'clearfraud', 'income_verification']
            matrix_data = cluster_data[cluster_features].replace({True: 1, False:0}).as_matrix()

            self.cluster_labels = cluster_model.predict(matrix_data)
        else:
            self.cluster_labels = [0] * len(self.data.modeling_data.index)

        preds = {}
        contribs = {}
        for cluster in unique(self.cluster_labels):

            model = self.model.fit_model[cluster]
            cluster_data = self.data.modeling_data.loc[self.cluster_labels == cluster]
            cluster_target = self.data.target.loc[self.cluster_labels == cluster]

            scoring_data = copy(cluster_data)
            train_columns = self.model.train_columns[cluster]

            missing_cols = setdiff(train_columns, list(scoring_data.columns))
            extra_cols = setdiff(list(scoring_data.columns), train_columns)

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
            xgb_data = DMatrix(scoring_data, label = self.data.target)

            if calc_contribs:

                contribs_df = copy(cluster_data[['account_id', 'application_id']])
                contribs_df['cluster'] = cluster

                c = model.predict(xgb_data, pred_contribs = True)
                c = pd.DataFrame.from_records(c, columns = list(scoring_data.columns) + ['bias'])

                contribs_df = pd.concat([contribs_df.reset_index(drop=True), c.reset_index(drop=True)], axis = 1)
                contribs[cluster] = contribs_df
                self.raw_contribs = pd.concat(list(contribs.values()))

                # rearrange raw_contribs into ordered json
                ordered_contribs_list = []
                for index, row in self.raw_contribs.iterrows():
                    clean_row = row.drop(['account_id', 'application_id', 'cluster'])

                    pos_contribs = clean_row.loc[clean_row > 0].sort_values(ascending = False)
                    pos_contribs_ordered = json.dumps([{'feature': i, 'contribution': c}  for i, c in list(zip(pos_contribs.index, pos_contribs))])

                    neg_contribs = clean_row.loc[clean_row < 0].sort_values(ascending = True)
                    neg_contribs_ordered = json.dumps([{'feature': i, 'contribution': c}  for i, c in list(zip(neg_contribs.index, neg_contribs))])


                    ordered_contribs_list.append({'account_id': row.account_id,
                                                  'application_id': row.application_id,
                                                  'positive_contributions_ordered': pos_contribs_ordered,
                                                  'negative_contributions_ordered': neg_contribs_ordered
                                                 })

                self.ordered_contribs = pd.DataFrame.from_dict(ordered_contribs_list)



            preds_df = copy(cluster_data[['account_id', 'application_id']])
            preds_df['cluster'] = cluster
            preds_df['score'] = model.predict(xgb_data)
            preds_df['target'] = cluster_target

            preds[cluster] = preds_df

        self.preds = pd.concat(list(preds.values()))

    def write_db(self, db_con, schema):

        fraud_scores = pd.merge(self.preds, self.data.ApplicantFeatures[['account_id', 'application_id', 'state']])
        fraud_scores = pd.merge(fraud_scores, self.ordered_contribs, on=['account_id', 'application_id'])
        fraud_scores.rename(columns={'score': 'fraud_score'}, inplace=True)

        fraud_scores['created_at'] = pd.to_datetime('now').strftime('%Y-%m-%d %H:%M:%S')

        db_records = find_or_create_model_records(model=self.model, db_con=db_con, schema=schema)
        db_records['cluster'] = db_records.model_cluster.str.split('_').str[-1].map(int)
        fraud_scores = pd.merge(fraud_scores, db_records[['cluster', 'model_id']], on = ['cluster'])
        fraud_scores = fraud_scores.drop(['target', 'cluster'], axis = 1)

        existing_records = pd.read_sql_table(table_name='fraud_scores', con=db_con, schema=schema)

        new_fraud_scores = find_new_records(fraud_scores, existing_records, ['account_id', 'application_id', 'model_id', 'state'])


        print('Writing ' + str(len(new_fraud_scores.index)) + ' rows...')
        new_fraud_scores.to_sql('fraud_scores', con=db_con, schema=schema, if_exists='append', index=False)
        self.db_table = new_fraud_scores
