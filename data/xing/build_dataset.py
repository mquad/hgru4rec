import numpy as np
import pandas as pd
import subprocess
import argparse

def remap_columns(data, columns):
    """Remap the values in each to the fields in `columns` to the range [0, number of unique values]"""
    maps = {}
    if not isinstance(columns, list):
        columns = [columns]
    for c in columns:
        # remap column to the range (0:num_unique_values)
        uniques = data[c].unique()
        col_map = pd.Series(index=uniques, data=np.arange(len(uniques)))
        data[c] = col_map[data[c]].values
        maps[c] = col_map
    return data, maps


def make_sessions(data, session_th=30 * 60, is_ordered=False, user_key='user_id', item_key='item_id', time_key='ts'):
    """Assigns session ids to the events in data without grouping keys"""
    if not is_ordered:
        # sort data by user and time
        data.sort_values(by=[user_key, time_key], ascending=True, inplace=True)
    # compute the time difference between queries
    tdiff = np.diff(data[time_key].values)
    # check which of them are bigger then session_th
    split_session = tdiff > session_th
    split_session = np.r_[True, split_session]
    # check when the user chenges is data
    new_user = data['user_id'].values[1:] != data['user_id'].values[:-1]
    new_user = np.r_[True, new_user]
    # a new sessions stars when at least one of the two conditions is verified
    new_session = np.logical_or(new_user, split_session)
    # compute the session ids
    session_ids = np.cumsum(new_session)
    data['session_id'] = session_ids
    return data


def write_dataset_to_hdf(filename, datasets, keys=['train', 'test', 'valid_train', 'valid_test']):
    assert len(datasets) == len(keys)
    for ds, k in zip(datasets, keys):
        ds.to_hdf(filename, k)


def write_dict_to_hdf(filename, d, use_keys=None):
    if use_keys:
        iterator = zip(use_keys, d.values())
    else:
        iterator = d
    for k, ds in iterator:
        ds.to_hdf(filename, k)


def last_session_out_split(data,
                           user_key='user_id',
                           item_key='item_id',
                           session_key='session_id',
                           time_key='ts',
                           clean_test=True,
                           min_session_length=2):
    """
    last-session-out split
    assign the last session of every user to the test set and the remaining ones to the training set
    """
    sessions = data.sort_values(by=[user_key, time_key]).groupby(user_key)[session_key]
    last_session = sessions.last()
    train = data[~data.session_id.isin(last_session.values)].copy()
    test = data[data.session_id.isin(last_session.values)].copy()
    if clean_test:
        train_items = train[item_key].unique()
        test = test[test[item_key].isin(train_items)]
        #  remove sessions in test shorter than min_session_length
        slen = test[session_key].value_counts()
        good_sessions = slen[slen >= min_session_length].index
        test = test[test[session_key].isin(good_sessions)].copy()
    return train, test


def last_n_days_out_split(data, n=1,
                          user_key='user_id',
                          item_key='item_id',
                          session_key='session_id',
                          time_key='ts',
                          clean_test=True,
                          min_session_length=2):
    """
    last n-days out split
    assign the sessions in the last n days to the test set and remaining to the training one
    """
    DAY = 24 * 60 * 60
    data.sort_values(by=[user_key, time_key], inplace=True)
    sessions_start = data.groupby(session_key)[time_key].agg('min')
    end_time = data[time_key].max()
    test_start = end_time - n * DAY
    train = data[data.session_id.isin(sessions_start[sessions_start < test_start].index)].copy()
    test = data[data.session_id.isin(sessions_start[sessions_start >= test_start].index)].copy()
    if clean_test:
        train_items = train[item_key].unique()
        test = test[test[item_key].isin(train_items)]
        #  remove sessions in test shorter than min_session_length
        slen = test[session_key].value_counts()
        good_sessions = slen[slen >= min_session_length].index
        test = test[test[session_key].isin(good_sessions)].copy()
    return train, test


parser = argparse.ArgumentParser()
parser.add_argument('interactions_path')
args = parser.parse_args()

print('Loading {}'.format(args.interactions_path))
interactions = pd.read_csv(args.interactions_path, header=0, sep='\t')
# remove interactions of type 'delete'
interactions = interactions[interactions.interaction_type != 4].copy()

print('Building sessions')
# partition interactions into sessions with 30-minutes idle time
interactions = make_sessions(interactions, session_th=30 * 60, time_key='created_at', is_ordered=False)
print('Original data:')
print('Num items: {}'.format(interactions.item_id.nunique()))
print('Num users: {}'.format(interactions.user_id.nunique()))
print('Num sessions: {}'.format(interactions.session_id.nunique()))

print('Filtering data')
# drop duplicate interactions within the same session
interactions.drop_duplicates(subset=['item_id', 'session_id', 'interaction_type'], keep='first', inplace=True)
# keep items with >=20 interactions
item_pop = interactions.item_id.value_counts()
good_items = item_pop[item_pop >= 20].index
inter_dense = interactions[interactions.item_id.isin(good_items)]
# remove sessions with length < 3
session_length = inter_dense.session_id.value_counts()
good_sessions = session_length[session_length >= 3].index
inter_dense = inter_dense[inter_dense.session_id.isin(good_sessions)]
# let's keep only returning users (with >= 5 sessions) and remove overly active ones (>=200 sessions)
sess_per_user = inter_dense.groupby('user_id')['session_id'].nunique()
good_users = sess_per_user[(sess_per_user >= 5) & (sess_per_user < 200)].index
inter_dense = inter_dense[inter_dense.user_id.isin(good_users)]
print('Filtered data:')
print('Num items: {}'.format(inter_dense.item_id.nunique()))
print('Num users: {}'.format(inter_dense.user_id.nunique()))
print('Num sessions: {}'.format(inter_dense.session_id.nunique()))

print('Partitioning data')
# last-session-out partitioning
train_full_sessions, test_sessions = last_session_out_split(inter_dense,
                                                            user_key='user_id',
                                                            item_key='item_id',
                                                            session_key='session_id',
                                                            time_key='created_at',
                                                            clean_test=True)
train_valid_sessions, valid_sessions = last_session_out_split(train_full_sessions,
                                                              user_key='user_id',
                                                              item_key='item_id',
                                                              session_key='session_id',
                                                              time_key='created_at',
                                                              clean_test=True)

print('Write to disk')
# write to disk
subprocess.call(['mkdir', '-p', 'dense/last-session-out'])
train_full_sessions.to_hdf('dense/last-session-out/sessions.hdf','train')
test_sessions.to_hdf('dense/last-session-out/sessions.hdf','test')
train_valid_sessions.to_hdf('dense/last-session-out/sessions.hdf', 'valid_train')
valid_sessions.to_hdf('dense/last-session-out/sessions.hdf','valid_test')

