# -*- coding: utf-8 -*-
"""
@author: Balázs Hidasi
Edited by: Massimo Quadrana

"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")


def evaluate_sessions_batch(pr, train_data, test_data, items=None, cut_off=20, batch_size=100, break_ties=False,
                            output_rankings=False,
                            session_key='SessionId', user_key='UserId', item_key='ItemId', time_key='Time'):
    """
    Evaluates the GRU4Rec network wrt. recommendation accuracy measured by recall@N and MRR@N.

    Parameters
    --------
    pr : gru4rec.GRU4Rec
        A trained instance of the GRU4Rec network.
    train_data : pandas.DataFrame
        Train data. It contains the transactions of the test set. It has one column for session IDs, one for item IDs
        and one for the timestamp of the events (unix timestamps).
        It must have a header. Column names are arbitrary, but must correspond to the keys you use in this function.
        (Actually not used by this function, kept only for interface compatibility)
    test_data : pandas.DataFrame
        Test data. Same format of train_data.
    items : 1D list or None
        The list of item ID that you want to compare the score of the relevant item to.
        If None, all items of the training set are used. Default value is None.
    cut_off : int
        Cut-off value (i.e. the length of the recommendation list; N for recall@N and MRR@N). Default value is 20.
    batch_size : int
        Number of events bundled into a batch during evaluation. Speeds up evaluation.
        If it is set high, the memory consumption increases. Default value is 100.
    break_ties : boolean
        Whether to add a small random number to each prediction value in order to break up possible ties,
        which can mess up the evaluation.
        Defaults to False, because (1) GRU4Rec usually does not produce ties, except when the output saturates;
        (2) it slows down the evaluation.
        Set to True is you expect lots of ties.
    output_rankings: boolean
        If True, stores the predicted ranks of every event in test data into a Pandas DataFrame
        that is returned by this function together with the metrics.
        Notice that predictors models do not provide predictions for the first event in each session. (default: False)
    session_key : string
        Header of the session ID column in the input file (default: 'SessionId')
    user_key : string
        Header of the user ID column in the input file (default: 'UserId')
    item_key : string
        Header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file (default: 'Time')

    Returns
    --------
    out : tuple
        (Recall@N, MRR@N[, DataFrame with the detailed predicted ranks])

    """

    # In case someone would try to run with both items=None and not None on the same model
    # without realizing that the predict function needs to be replaced
    pr.predict = None
    test_data.sort_values([session_key, time_key], inplace=True)
    offset_sessions = np.zeros(test_data[session_key].nunique() + 1, dtype=np.int32)
    offset_sessions[1:] = test_data.groupby(session_key).size().cumsum()
    evalutation_point_count = 0
    mrr, recall = 0.0, 0.0

    # get the other columns in the dataset
    columns = [user_key, session_key, item_key]
    other_columns = test_data.columns.values[np.in1d(test_data.columns.values, columns, invert=True)].tolist()

    if output_rankings:
        rank_list = []

    if len(offset_sessions) - 1 < batch_size:
        batch_size = len(offset_sessions) - 1
    iters = np.arange(batch_size).astype(np.int32)
    maxiter = iters.max()
    start = offset_sessions[iters]
    end = offset_sessions[iters + 1]
    in_item_id = np.zeros(batch_size, dtype=np.int32)
    in_user_id = np.zeros(batch_size, dtype=np.int32)
    in_session_id = np.zeros(batch_size, dtype=np.int32)

    np.random.seed(42)
    perc = 10
    session_cnt = 0
    n_sessions = len(offset_sessions)
    while True:
        valid_mask = iters >= 0
        if valid_mask.sum() == 0:
            break
        start_valid = start[valid_mask]
        minlen = (end[valid_mask] - start_valid).min()
        in_item_id[valid_mask] = test_data[item_key].values[start_valid]
        in_user_id[valid_mask] = test_data[user_key].values[start_valid] if user_key is not None else -1
        in_session_id[valid_mask] = test_data[session_key].values[start_valid]

        for i in range(minlen - 1):
            out_item_idx = test_data[item_key].values[start_valid + i + 1]
            if items is not None:
                uniq_out = np.unique(np.array(out_item_idx, dtype=np.int32))
                preds = pr.predict_next_batch(iters, in_item_id, in_user_id,
                                              np.hstack([items, uniq_out[~np.in1d(uniq_out, items)]]),
                                              batch_size)
            else:
                preds = pr.predict_next_batch(iters, in_item_id, in_user_id, None, batch_size)
            if break_ties:
                preds += np.random.rand(*preds.values.shape) * 1e-8
            preds.fillna(0, inplace=True)
            in_item_id[valid_mask] = out_item_idx

            if items is not None:
                others = preds.ix[items].values.T[valid_mask].T
                targets = np.diag(preds.ix[in_item_id].values)[valid_mask]
                ranks = (others > targets).sum(axis=0) + 1
            else:
                ranks = (preds.values.T[valid_mask].T > np.diag(preds.ix[in_item_id].values)[valid_mask]).sum(
                    axis=0) + 1

            if output_rankings:
                eval_record = np.vstack([in_user_id[valid_mask],
                                         in_session_id[valid_mask],
                                         in_item_id[valid_mask],
                                         ranks])
                others_record = np.vstack([test_data[c].values[start_valid + i + 1] for c in other_columns])
                batch_results = np.vstack([eval_record, others_record]).T
                rank_list.append(batch_results)

            rank_ok = ranks <= cut_off
            recall += rank_ok.sum()
            mrr += (1.0 / ranks[rank_ok]).sum()
            evalutation_point_count += len(ranks)
        start = start + minlen - 1
        mask = np.arange(len(iters))[(valid_mask) & (end - start <= 1)]
        for idx in mask:
            session_cnt += 1
            if session_cnt > int(perc * n_sessions / 100):
                logger.info('Sessions {}/{} ({}% completed)'.format(session_cnt, n_sessions, perc))
                perc += 10
            maxiter += 1
            if maxiter >= len(offset_sessions) - 1:
                iters[idx] = -1
            else:
                iters[idx] = maxiter
                start[idx] = offset_sessions[maxiter]
                end[idx] = offset_sessions[maxiter + 1]
    print(evalutation_point_count)

    if output_rankings:
        columns = [user_key, session_key, item_key, 'rank'] + other_columns
        df_ranks = pd.DataFrame.from_records(np.vstack(rank_list), columns=columns)
        return recall / evalutation_point_count, mrr / evalutation_point_count, df_ranks
    else:
        return recall / evalutation_point_count, mrr / evalutation_point_count


def evaluate_sessions_batch_hier_bootstrap(pr, train_data, test_data, items=None, cut_off=20, batch_size=100,
                                           break_ties=False,
                                           output_rankings=False,
                                           bootstrap_length=-1,
                                           session_key='SessionId', user_key='UserId', item_key='ItemId',
                                           time_key='Time'):
    """
    Evaluates the HGRU4Rec network wrt. recommendation accuracy measured by recall@N and MRR@N.
    Concatenates train sessions to test sessions to bootstrap the hidden states of the HGRU.
    The number of the last sessions of each user that are used in the bootstrapping is controlled by `bootstrap_length`.

    Parameters
    --------
    pr : gru4rec.HGRU4Rec
        A trained instance of the HGRU4Rec network.
    train_data : pandas.DataFrame
        Train data. It contains the transactions of the test set. It has one column for session IDs,
        one for item IDs and one for the timestamp of the events (unix timestamps).
        It must have a header. Column names are arbitrary, but must correspond to the keys you use in this function.
    test_data : pandas.DataFrame
        Test data. Same format of train_data.
    items : 1D list or None
        The list of item ID that you want to compare the score of the relevant item to.
        If None, all items of the training set are used. Default value is None.
    cut_off : int
        Cut-off value (i.e. the length of the recommendation list; N for recall@N and MRR@N). Default value is 20.
    batch_size : int
        Number of events bundled into a batch during evaluation. Speeds up evaluation.
         If it is set high, the memory consumption increases. Default value is 100.
    break_ties : boolean
        Whether to add a small random number to each prediction value in order to break up possible ties,
        which can mess up the evaluation.
        Defaults to False, because (1) GRU4Rec usually does not produce ties, except when the output saturates;
        (2) it slows down the evaluation.
        Set to True is you expect lots of ties.
    output_rankings: boolean
        If True, stores the predicted ranks of every event in test data into a Pandas DataFrame
        that is returned by this function together with the metrics.
        Notice that predictors models do not provide predictions for the first event in each session. (default: False)
    bootstrap_length: int
        Number of sessions in train data used to bootstrap the hidden state of the predictor,
        starting from the last training session of each user.
        If -1, consider all sessions. (default: -1)
    session_key : string
        Header of the session ID column in the input file (default: 'SessionId')
    user_key : string
        Header of the user ID column in the input file (default: 'UserId')
    item_key : string
        Header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file (default: 'Time')

    Returns
    --------
    out : tuple
        (Recall@N, MRR@N[, DataFrame with the detailed predicted ranks])

    """
    # In case someone would try to run with both items=None and not None on the same model
    # without realizing that the predict function needs to be replaced
    pr.predict = None

    # use the training sessions of the users in test_data to bootstrap the state of the user RNN
    test_users = test_data[user_key].unique()
    train_data = train_data[train_data[user_key].isin(test_users)].copy()
    # select the bootstrap_length recent sessions in training data to bootstrap the hidden state of the predictor
    if bootstrap_length >= 0:
        user_sessions = train_data.sort_values(by=[user_key, time_key])[[user_key, session_key]].drop_duplicates()
        session_order = user_sessions.groupby(user_key, sort=False).cumcount(ascending=False)
        last_sessions = user_sessions[session_order < bootstrap_length][session_key]
        train_data = train_data[train_data[session_key].isin(last_sessions)].copy()
    # concatenate training and test sessions
    train_data['in_eval'] = False
    test_data['in_eval'] = True
    test_data = pd.concat([train_data, test_data])
    # pre-process the session data
    user_indptr, offset_sessions = pr.preprocess_data(test_data)
    offset_users = offset_sessions[user_indptr]

    # get the other columns in the dataset
    columns = [user_key, session_key, item_key]
    other_columns = test_data.columns.values[np.in1d(test_data.columns.values, columns, invert=True)].tolist()
    other_columns.remove('in_eval')

    evalutation_point_count = 0
    mrr, recall = 0.0, 0.0

    if output_rankings:
        rank_list = []

    # here we use parallel minibatches over users
    if len(offset_users) - 1 < batch_size:
        batch_size = len(offset_users) - 1

    # variables used to iterate over users
    user_iters = np.arange(batch_size).astype(np.int32)
    user_maxiter = user_iters.max()
    user_start = offset_users[user_iters]
    user_end = offset_users[user_iters + 1]

    # variables to manage iterations over sessions
    session_iters = user_indptr[user_iters]
    session_start = offset_sessions[session_iters]
    session_end = offset_sessions[session_iters + 1]

    in_item_id = np.zeros(batch_size, dtype=np.int32)
    in_user_id = np.zeros(batch_size, dtype=np.int32)
    in_session_id = np.zeros(batch_size, dtype=np.int32)

    np.random.seed(42)
    perc = 10
    n_users = len(offset_users)
    user_cnt = 0
    while True:
        # iterate only over the valid entries in the minibatch
        valid_mask = np.logical_and(user_iters >= 0, session_iters >= 0)
        if valid_mask.sum() == 0:
            break

        session_start_valid = session_start[valid_mask]
        session_end_valid = session_end[valid_mask]
        session_minlen = (session_end_valid - session_start_valid).min()
        in_item_id[valid_mask] = test_data[item_key].values[session_start_valid]
        in_user_id[valid_mask] = test_data[user_key].values[session_start_valid]
        in_session_id[valid_mask] = test_data[session_key].values[session_start_valid]

        for i in range(session_minlen - 1):
            out_item_idx = test_data[item_key].values[session_start_valid + i + 1]
            if items is not None:
                uniq_out = np.unique(np.array(out_item_idx, dtype=np.int32))
                preds = pr.predict_next_batch(in_session_id, in_item_id, in_user_id,
                                              np.hstack([items, uniq_out[~np.in1d(uniq_out, items)]]),
                                              batch_size)
            else:
                preds = pr.predict_next_batch(in_session_id, in_item_id, in_user_id, None, batch_size)
            if break_ties:
                preds += np.random.rand(*preds.values.shape) * 1e-8

            preds.fillna(0, inplace=True)

            in_item_id[valid_mask] = out_item_idx
            in_eval_mask = np.zeros(batch_size, dtype=np.bool)
            in_eval_mask[valid_mask] = test_data['in_eval'].values[session_start_valid + i + 1]

            if np.any(in_eval_mask):
                if items is not None:
                    others = preds.ix[items].values.T[in_eval_mask].T
                    targets = np.diag(preds.ix[in_item_id].values)[in_eval_mask]
                    ranks = (others > targets).sum(axis=0) + 1
                else:
                    ranks = (preds.values.T[in_eval_mask].T > np.diag(preds.ix[in_item_id].values)[in_eval_mask]).sum(
                        axis=0) + 1
                if output_rankings:
                    session_start_eval = session_start[in_eval_mask]
                    eval_record = [in_user_id[in_eval_mask],  # user id
                                   in_session_id[in_eval_mask],  # session id
                                   in_item_id[in_eval_mask],  # OUTPUT item id (see line 261)
                                   ranks]
                    others_record = np.vstack([test_data[c].values[session_start_eval + i + 1] for c in other_columns])
                    batch_results = np.vstack([eval_record, others_record]).T
                    rank_list.append(batch_results)

                rank_ok = ranks <= cut_off
                recall += rank_ok.sum()
                mrr += (1.0 / ranks[rank_ok]).sum()
                evalutation_point_count += len(ranks)

        session_start[valid_mask] = session_start[valid_mask] + session_minlen - 1
        session_start_mask = np.arange(len(user_iters))[valid_mask & (session_end - session_start <= 1)]
        for idx in session_start_mask:
            session_iters[idx] += 1
            if session_iters[idx] + 1 >= len(offset_sessions):
                session_iters[idx] = -1
                user_iters[idx] = -1
                break
            session_start[idx] = offset_sessions[session_iters[idx]]
            session_end[idx] = offset_sessions[session_iters[idx] + 1]

        user_change_mask = np.arange(len(user_iters))[valid_mask & (user_end - session_start <= 0)]
        for idx in user_change_mask:
            user_cnt += 1
            if user_cnt > int(perc * n_users / 100):
                logger.info('User {}/{} ({}% completed)'.format(user_cnt, n_users, perc))
                perc += 10
            user_maxiter += 1
            if user_maxiter + 1 >= len(offset_users):
                session_iters[idx] = -1
                user_iters[idx] = -1
                break
            user_iters[idx] = user_maxiter
            user_start[idx] = offset_users[user_maxiter]
            user_end[idx] = offset_users[user_maxiter + 1]
            session_iters[idx] = user_indptr[user_maxiter]
            session_start[idx] = offset_sessions[session_iters[idx]]
            session_end[idx] = offset_sessions[session_iters[idx] + 1]

    if output_rankings:
        columns = [user_key, session_key, item_key, 'rank'] + other_columns
        df_ranks = pd.DataFrame.from_records(np.vstack(rank_list), columns=columns)
        return recall / evalutation_point_count, mrr / evalutation_point_count, df_ranks
    else:
        return recall / evalutation_point_count, mrr / evalutation_point_count


def evaluate_sessions(pr, train_data, test_data, items=None, cut_off=20, output_rankings=False, session_key='SessionId',
                      user_key='UserId', item_key='ItemId', time_key='Time'):
    """
    Evaluates the baselines wrt. recommendation accuracy measured by recall@N and MRR@N. Has no batch evaluation capabilities. Breaks up ties.

    Parameters
    --------
    pr : baseline predictor
        A trained instance of a baseline predictor.
    train_data : pandas.DataFrame
        Train data. It contains the transactions of the test set. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
        It must have a header. Column names are arbitrary, but must correspond to the keys you use in this function.
    test_data : pandas.DataFrame
        Test data. Same format of train_data.
    items : 1D list or None
        The list of item ID that you want to compare the score of the relevant item to. If None, all items of the training set are used. Default value is None.
    cut_off : int
        Cut-off value (i.e. the length of the recommendation list; N for recall@N and MRR@N). Defauld value is 20.
    output_rankings: boolean
        If True, stores the predicted ranks of every event in test data into a Pandas DataFrame
        that is returned by this function together with the metrics.
        Notice that predictors models do not provide predictions for the first event in each session. (default: False)
    session_key : string
        Header of the session ID column in the input file (default: 'SessionId')
    user_key : string
        Header of the user ID column in the input file (default: 'UserId')
    item_key : string
        Header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file (default: 'Time')

    Returns
    --------
    out : tuple
        (Recall@N, MRR@N[, DataFrame with the detailed predicted ranks])

    """

    test_data.sort_values([session_key, time_key], inplace=True)
    items_to_predict = train_data[item_key].unique()
    evalutation_point_count = 0
    prev_iid, prev_uid, prev_sid = -1, -1, -1
    mrr, recall = 0.0, 0.0

    if output_rankings:
        rank_list = []

    for i in range(len(test_data)):
        sid = test_data[session_key].values[i]
        iid = test_data[item_key].values[i]
        uid = test_data[user_key].values[i] if user_key is not None else -1
        if prev_sid != sid:
            prev_sid = sid
        else:
            if items is not None:
                if np.in1d(iid, items):
                    items_to_predict = items
                else:
                    items_to_predict = np.hstack(([iid], items))
            preds = pr.predict_next(sid, prev_iid, prev_uid, items_to_predict)
            preds[np.isnan(preds)] = 0
            preds += 1e-8 * np.random.rand(len(preds))  # Breaking up ties
            rank = (preds > preds[iid]).sum() + 1

            if output_rankings:
                if user_key is not None:
                    rank_list.append((uid, sid, iid, rank))
                else:
                    rank_list.append((sid, iid, rank))
            assert rank > 0
            if rank <= cut_off:
                recall += 1
                mrr += 1.0 / rank
            evalutation_point_count += 1
        prev_iid = iid
        prev_uid = uid

    if output_rankings:
        if user_key is not None:
            columns = [user_key, session_key, item_key, 'rank']
        else:
            columns = [session_key, item_key, 'rank']
        df_ranks = pd.DataFrame.from_records(np.vstack(rank_list), columns=columns)
        return recall / evalutation_point_count, mrr / evalutation_point_count, df_ranks
    else:
        return recall / evalutation_point_count, mrr / evalutation_point_count
