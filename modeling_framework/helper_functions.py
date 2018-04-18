from sklearn.model_selection import cross_val_score, cross_val_predict
from sqlalchemy import create_engine, text
from geopy.distance import great_circle
from operator import itemgetter
from itertools import product
from copy import copy
import pandas as pd
import numpy as np
import re, ast, os, pickle, collections, json

def find_new_records(new_df, existing_df, subset_cols):
    join = pd.merge(new_df, existing_df, how='left', on=subset_cols)
    new = join.loc[pd.isnull(join.iloc[:,-1])]
    new_clean = new_df.iloc[new.index]
    return new_clean

# this fixes stupid parentheses in json value fields
def fixed_parsed_attributes(parsed_attrs):
    n = []
    for d in parsed_attrs:
        for k, v in d.items():
            try:
                d[k] = ast.literal_eval(v)
            except:
                d[k] = v
        n.append(d)
    return pd.Series(n)

def format_id_vector(ids):

    if isinstance(ids, pd.Series):
        ids = list(ids)

    if isinstance(ids, (list,)):
        # if a list of numbers, just convert each element to string
        ids = list(map(str, ids))

        # if a list of strings, need to add quotes inbetween
        if re.match('[a-zA-Z]', ''.join(list(ids))):
            ids = ', '.join(["'" + thing.replace("'", "''") + "'" for thing in ids])
            # ids = ', '.join(['"' + thing + '"' for thing in ids])
            return ids

    else:
        # if not a list already, make a list
        ids = [str(ids)]
    return ', '.join(ids)

def format_query_with_dummy_var(query_path, replace_string):
    with open(query_path) as f:
        query = f.read()
        query = text(query.replace('_dummy_variable_', replace_string))

    return query

def rbind(a, b):
    return pd.concat([a.reset_index(drop = True), b.reset_index(drop = True)], axis = 1)

def regex_remove_cols(df, regex):
    r = re.compile(regex)
    filtered_cols = list(filter(r.match, list(df.columns)))
    return df.drop(filtered_cols, axis = 1)

def setdiff(first, second):
        second = set(second)
        return [item for item in first if item not in second]

def col_drop_try(df, cols_to_drop):
    for col in cols_to_drop:
        try:
            df = df.drop(col, axis = 1)
        except:
            pass

    return df

def read_data(data):
    # if isinstance(data, modeling_data.ModelingData):
    if type(data).__name__ == 'ModelingData':
        data = data
    else:
        try:
            if '.pkl' not in data:
                data = os.path.join(data, '.pkl')
            with open(data, 'rb') as pickle_file:
                data = pickle.load(pickle_file)
        except:
            raise ValueError('Must pass in ModelingData object or valid path to a saved model folder')

    return data

def read_model(model):
    # if isinstance(model, model_wrapper.ModelWrapper):
    if type(model).__name__ == 'ModelWrapper':
        model = model
    else:
        try:
            if '.pkl' not in model:
                model = os.path.join(model, 'model.pkl')

            with open(model, 'rb') as pickle_file:
                model = pickle.load(pickle_file)
        except:
            raise ValueError('Must pass in Model object or valid path to a saved model folder')

    return model

def multiclass_lookup(target_raw, target_numeric):
    lookup = pd.DataFrame(list(zip(target_numeric, target_raw)))
    lookup = lookup.loc[~lookup.duplicated()]

    lookup_dict = {}
    lookup_dict_inverse = {}
    for i, r in lookup.iterrows():
        lookup_dict[r[1]] = r[0]
        lookup_dict_inverse[r[0]] = r[1]
    return lookup_dict, lookup_dict_inverse

def rank_preds(preds_df, target, num_offers):
    def myf(x):
        x = sorted(x.items(), key=itemgetter(1), reverse=True)
        return [z[0] for z in x]

    ordered_preds = list(map(myf, preds_df.to_dict(orient='records')))

    # find the rank of the highest predicted offer
    # cap the rank by the number of offers a customer received (can't have a rank higher than number of offers)
    rank=[]
    for t, cap, preds in list(zip(target, num_offers, ordered_preds)):
        r = min(preds.index(t) + 1, cap)
        rank.append(r)

    rank=pd.Series(rank)
    return rank

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def jsons_to_dfs(jsons):
    jsons = jsons.loc[~pd.isnull(jsons)]
    j = jsons.apply(json.loads).apply(flatten)
    df = pd.concat(list(j.apply(pd.DataFrame, index=[0])))
    df.index=jsons.index
    return df

# --------------------------------------
# --------------------------------------
# ----------Objective Functions---------
# --------------------------------------
# --------------------------------------


def set_objective_vars(model_in, X_in, y_in, tuning_metric_in, num_offers_in):
    global model, tuning_metric
    global X, y, num_offers

    model = model_in
    X = X_in
    y = y_in
    tuning_metric = tuning_metric_in
    num_offers = num_offers_in


def xgb_objective(params):
    print(params)
    base_score, max_depth, n_estimators, learning_rate, min_child_weight, gamma, colsample_bytree, reg_alpha, reg_lambda, subsample = params

    model.set_params(   base_score=base_score,
                        max_depth=max_depth,
                        n_estimators = n_estimators,
                        learning_rate=learning_rate,
                        min_child_weight=min_child_weight,
                        gamma=gamma,
                        colsample_bytree = colsample_bytree,
                        reg_alpha = reg_alpha,
                        reg_lambda = reg_lambda,
                        subsample = subsample
                    )

    return -np.mean(cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring=tuning_metric, verbose= 11))


def reg_linear_objective(params):
    print(params)
    alpha, l1_ratio = params

    model.set_params(   alpha=alpha,
                        l1_ratio=l1_ratio,
                        max_iter=1000,
                        tol=0.001
                    )

    return -np.mean(cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring=tuning_metric, verbose= 11))


def lgbm_objective(params):
    print(params)
    max_depth, n_estimators, learning_rate, min_child_weight, min_split_gain, colsample_bytree, reg_alpha, reg_lambda, subsample = params

    model.set_params(   max_depth=max_depth,
                        n_estimators = n_estimators,
                        learning_rate=learning_rate,
                        min_child_weight=min_child_weight,
                        min_split_gain=min_split_gain,
                        colsample_bytree = colsample_bytree,
                        reg_alpha = reg_alpha,
                        reg_lambda = reg_lambda,
                        subsample = subsample
                    )

    return -np.mean(cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring=tuning_metric, verbose= 11))

def lgbm_rank_objective(params):
    print(params)
    max_depth, n_estimators, learning_rate, min_child_weight, min_split_gain, colsample_bytree, reg_alpha, reg_lambda, subsample = params

    model.set_params(   max_depth=max_depth,
                        n_estimators = n_estimators,
                        learning_rate=learning_rate,
                        min_child_weight=min_child_weight,
                        min_split_gain=min_split_gain,
                        colsample_bytree = colsample_bytree,
                        reg_alpha = reg_alpha,
                        reg_lambda = reg_lambda,
                        subsample = subsample
                    )

    preds = pd.DataFrame(cross_val_predict(model, X, y, cv=5, n_jobs=-1, verbose= 11, method='predict_proba'))
    avg_rank = rank_preds(preds_df=preds, target=y, num_offers=num_offers).mean()
    print(avg_rank)
    return avg_rank
