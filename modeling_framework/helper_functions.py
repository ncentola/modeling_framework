from sklearn.model_selection import cross_val_score
from sqlalchemy import create_engine, text
from geopy.distance import great_circle
from operator import itemgetter
from itertools import product
from copy import copy
import pandas as pd
import numpy as np
import re, ast, os

def clean_data(df):
    def clean_col(col_name):
        col = df[col_name]
        #if not any(col.str.contains('[a-z|A-Z]')):
        try:
            col = pd.to_numeric(col)
        except:
            print('failed to convert - ' + col_name)
            pass
        return col

    clean_df = pd.DataFrame.from_records(list(map(clean_col, list(df.columns))))
    clean_df = clean_df.transpose()
    clean_df.columns = df.columns
    return clean_df

def calculate_fraud_stats(df, groupby_col):
    df = df.drop_duplicates()
    gb = df.groupby(groupby_col).agg({'fraud': 'mean', 'account_id': 'count'}).reset_index(drop = False)
    gb.columns = ['query_group', groupby_col + '_overall_fraud_rate', groupby_col + '_overall_count']
    return gb


def detect_cores():
    if hasattr(os, "sysconf"):
        if os.sysconf_names.get("SC_NPROCESSORS_ONLN"):
            ncpus = os.sysconf("SC_NPROCESSORS_ONLN")
            if isinstance(ncpus, int) and ncpus > 0:
                return ncpus
        else:
             return int(os.popen("sysctl -n hw.ncpu").read())


def find_new_records(new_df, existing_df, subset_cols):
    join = pd.merge(new_df, existing_df, how='left', on=subset_cols)
    new = join.loc[pd.isnull(join.iloc[:,-1])]
    new_clean = new_df.iloc[new.index]
    return new_clean


def find_or_create_model_group(model, db_con, schema):
    try:
        model_group_table = pd.read_sql_table('model_groups', con=db_con, schema=schema)
        if any(model.model_name == model_group_table.model_name):
            return int(model_group_table.loc[model_group_table.model_name == model.model_name].model_group_id)
    except:
        print('Unable to read model_groups table, creating table now...')

    try:
        num_clusters = model.cluster_centers_df.shape[0]
        cluster_centers_json = model.cluster_centers_df.to_json(orient = 'records')
    except:
        num_clusters = 1
        cluster_centers_json = np.nan

    try:
        model_group_id = model_group_table.model_group_id.max() + 1
    except:
        model_group_id = 1

    new_model_group = {
        'model_group_id': model_group_id,
        'model_name': model.model_name,
        'number_of_clusters': num_clusters,
        'cluster_centers': cluster_centers_json
    }

    new_model_group_df = pd.DataFrame(new_model_group, index = [0])
    new_model_group_df.to_sql('model_groups', con=db_con, schema=schema, if_exists='append', index=False)

    return int(new_model_group_df.model_group_id)

def find_or_create_model_records(model, db_con, schema):
    model_group_id = find_or_create_model_group(model, db_con, schema)

    try:
        models_table = pd.read_sql_table('models', con=db_con, schema=schema)

        if any(models_table.model_group_id == model_group_id):
            return models_table.loc[models_table.model_group_id == model_group_id]
    except:
        print('Unable to read models table, creating table now...')

    try:
        model_id = models_table.model_id.max() + 1
    except:
        model_id = 1

    new_model_records = []
    for k, v in model.fit_model.items():

        cluster_centers = model.cluster_centers_df.loc[[k]].to_json(orient='records')
        importance_dict = model.fit_model[k].get_score(importance_type='gain')
        sorted_importance = sorted(importance_dict.items(), key=itemgetter(1), reverse=True)
        sorted_importance_json = pd.DataFrame(sorted_importance, columns = ['Feature', 'Importance']).to_json(orient='records')

        new_model_record = {
            'model_id': model_id,
            'model_group_id': model_group_id,
            'model_cluster': model.model_name + '/cluster_' + str(k),
            'cluster_center': cluster_centers,
            'importance': sorted_importance_json
        }

        new_model_records.append(new_model_record)
        model_id += 1

    df = pd.DataFrame(new_model_records)
    df.to_sql('models', con=db_con, schema=schema, if_exists='append', index=False)
    return(df)

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

def process_ip_data_split(split):
    account_id = split.account_id.iloc[0]
    # get rid of rows with NaN ip_lat or ip_lon
    split = split[np.isfinite(split[['ip_lat', 'ip_lon']]).apply(all, axis = 1)]

    if len(split.index) == 0:
        aggs = {
                    'account_id'         : account_id,
                    'ip_ip_distance'     : np.nan,
                    'address_ip_distance': np.nan,
                    'ip_velocity'        : np.nan,
                    'ip_seq_dist_avg'    : np.nan,
                    'days_per_ip'        : np.nan
               }
        return aggs

    # order sequentially and reset index
    split = split.sort_values('created_at')
    split = split.reset_index(drop = True)

    # function for vectorized calculation of great circle distance
    def gcv(start_lat, start_lon, end_lat, end_lon):
        start = (start_lat, start_lon)
        end = (end_lat, end_lon)
        return great_circle(start, end).miles

    # ----------------------
    # create the cartesian prod of ip coords for IP to IP distance
    latlon = split.ip_lat.map(str) + ',' + split.ip_lon.map(str)
    prod = pd.DataFrame.from_records(product(latlon, latlon))

    p0 = pd.DataFrame.from_records(list(prod[0].str.split(',')))

    p0.columns = ['start_lat', 'start_lon']
    p0 = p0.apply(pd.to_numeric)

    p1 = pd.DataFrame.from_records(list(prod[1].str.split(',')))
    p1.columns = ['end_lat', 'end_lon']
    p1 = p1.apply(pd.to_numeric)

    pc = rbind(p0, p1)
    # -----------------------

    # -----------------------
    # create sequential IP dataset for IP velocity
    split_a = copy(split)
    split_a['join'] = np.arange(0, split_a.shape[0])
    split_b = copy(split_a)
    split_b['join'] = split_b['join'] - 1

    sequential_data = pd.merge(split_a, split_b, on = 'join')
    # -----------------------


    # calculate distance and time delta vectors
    ip_ip_dist_grid = np.vectorize(gcv)(pc['start_lat'], pc['start_lon'], pc['end_lat'], pc['end_lon'])
    address_ip_dists = np.vectorize(gcv)(split['ip_lat'], split['ip_lon'], split['address_lat'], split['address_lon'])

    # calc these things only if there's more than 1 ip_address record
    if sequential_data.shape[0] > 1:
        ip_ip_sequential_dists = np.vectorize(gcv)(sequential_data['ip_lat_x'], sequential_data['ip_lon_x'], sequential_data['ip_lat_y'], sequential_data['ip_lon_y'])
        ip_ip_sequential_time_deltas = (sequential_data.created_at_y - sequential_data.created_at_x).dt.total_seconds() / 3600 # time delta in hours
        ip_ip_sequential_mph = ip_ip_sequential_dists / ip_ip_sequential_time_deltas
        ip_ip_sequential_mph = ip_ip_sequential_mph[~(np.isinf(ip_ip_sequential_mph) | np.isnan(ip_ip_sequential_mph))]
    else:
        ip_ip_sequential_dists = 0
        ip_ip_sequential_time_deltas = 0
        ip_ip_sequential_mph = 0

    days_per_ip = ((max(split.created_at) - min(split.created_at)).total_seconds() / 3600 / 24) / len(split.ip_address.unique())
#     days_per_ip = ((max(split.created_at) - min(split.created_at)).days + 1) / len(split.ip_address.unique())

    # aggregate and wrap up into dict
    aggs = {
                'account_id'         : account_id,
                'ip_ip_distance'     : max(ip_ip_dist_grid),
                'address_ip_distance': max(address_ip_dists),
                'ip_velocity'        : np.mean(ip_ip_sequential_mph),
                'ip_seq_dist_avg'    : np.mean(ip_ip_sequential_dists),
                'days_per_ip'        : days_per_ip
           }

    return aggs

def setdiff(first, second):
        second = set(second)
        return [item for item in first if item not in second]

def set_objective_vars(model_in, X_in, y_in, tuning_metric_in):
    global model, tuning_metric
    global X, y

    model = model_in
    X = X_in
    y = y_in
    tuning_metric = tuning_metric_in

def col_drop_try(df, cols_to_drop):
    for col in cols_to_drop:
        try:
            df = df.drop(col, axis = 1)
        except:
            pass

    return df

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
