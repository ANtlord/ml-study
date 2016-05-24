#!/usr/bin/env python
import datetime
from typing import Iterable, List, Set, Dict

import numpy as np
import pandas
from dask.dataframe import Series
from pandas import DataFrame
import sys

from sklearn.cross_validation import KFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN_NAME = 'radiant_win'


def _get_cols_with_nans(in_data: DataFrame):
    for col_name in in_data.keys():
        if in_data[col_name].hasnans:
            yield col_name


def _fill_cols(in_data: DataFrame, names: Iterable[str],
               nan_replacement):
    for col_name in names:
        print(col_name)
        val = nan_replacement
        in_data[col_name].fillna(val, inplace=True)

    return in_data


class GradientBoostResults:
    def __init__(self, used_trees_count, execution_time, validation_scores):
        self.scores = validation_scores
        self.execution_time = execution_time
        self.trees_count = used_trees_count


def _cross_validation(in_clf, in_data, results):
    start_time = datetime.datetime.now()
    kf = KFold(in_data.shape[0], 5, shuffle=True)
    scores = []
    for train_index, test_index in kf:
        X_train = in_data[train_index]
        Y_train = results[train_index]
        X_test = in_data[test_index]
        Y_test = results[test_index]

        in_clf.fit(X_train, Y_train)
        scores.append(roc_auc_score(Y_test, in_clf.predict_proba(X_test)[:, 1]))
    stop_time = datetime.datetime.now()
    execution_time = stop_time - start_time

    return scores, execution_time


def _gradient_classify(in_data: DataFrame, results: Series):
    for trees_count in [10, 20, 30, 100]:
        clf = GradientBoostingClassifier(n_estimators=trees_count, verbose=False)
        scores, execution_time = _cross_validation(clf, in_data, results)
        print('Time with %s trees: %s. Score: %s' % (trees_count, execution_time,
                                                     sum(scores) / len(scores)))


def _logistic_regression_classify(in_data: np.array, results: np.array):
    res_clf = None
    best_accuracy = 0
    for C in [0.1, 0.5, 1.0, 10]:
        clf = LogisticRegression(C=C)
        scores, execution_time = _cross_validation(clf, in_data, results)
        accuracy = sum(scores) / len(scores)
        print('Time with %s C-value: %s. Score: %s' % (clf.C, execution_time, accuracy))
        if best_accuracy < accuracy:
            res_clf = clf
    return res_clf


def _get_games_filter(of_hero_id: int, hero_fields: List[str], in_data: DataFrame) -> Series:
    filter_key = None
    for field_name in hero_fields:
        val = in_data[field_name] == of_hero_id
        if filter_key is None:
            filter_key = val
        else:
            filter_key |= val

    return filter_key


def _get_win_games(is_radiant: int, filter_key, in_data: DataFrame):
    games_count = in_data[filter_key].shape[0]
    win_count = in_data[filter_key & (in_data[TARGET_COLUMN_NAME] == is_radiant)].shape[0]
    return games_count, win_count


def _collect_heroes(in_data: DataFrame) -> Set[int]:
    heroes_ids = []
    for key in drop_keys:
        heroes_ids += in_data[key].unique().tolist()
    return set(heroes_ids)


def _compute_win_coeffs(for_horoes: Set[int], in_data: DataFrame) -> Dict[str, float]:
    heroes_win_coeff_set = {}
    for hero_id in for_horoes:
        radiant_games = _get_games_filter(hero_id, radiant_heroes, in_data)
        drew_games = _get_games_filter(hero_id, drew_heroes, in_data)

        radiant_games, radiant_wins = _get_win_games(1, radiant_games, in_data)
        drew_games, drew_wins = _get_win_games(0, drew_games, in_data)
        heroes_win_coeff = (radiant_wins + drew_wins) / (radiant_games + drew_games)
        heroes_win_coeff_set[hero_id] = heroes_win_coeff

    return heroes_win_coeff_set


def _encode_fraction(of_heroes: List[int], in_data: DataFrame) -> np.array:
    """
    unused
    :param of_heroes:
    :param in_data:
    :return:
    """
    x_pick = np.zeros((in_data.shape[0], len(of_heroes)))
    for i, match_id in enumerate(in_data.index):
        for p in range(5):
            hero = in_data.ix[match_id, 'r%d_hero' % (p+1)]
            x_pick[i, of_heroes.index(hero)] = 1
            hero = in_data.ix[match_id, 'd%d_hero' % (p+1)]
            x_pick[i, of_heroes.index(hero)] = -1
    return x_pick


if __name__ == '__main__':
    learn_data = pandas.read_csv('features.csv', index_col='match_id')  # type: DataFrame
    test_data = pandas.read_csv('features_test.csv', index_col='match_id')  # type: DataFrame
    unusable_headers = [x for x in learn_data.keys() if x not in test_data.keys()]
    Y = learn_data[TARGET_COLUMN_NAME]
    unusable_headers += [TARGET_COLUMN_NAME]

    radiant_heroes = ['r%i_hero' % x for x in range(1, 6)]
    drew_heroes = ['d%i_hero' % x for x in range(1, 6)]
    drop_keys = radiant_heroes + drew_heroes + ['lobby_type']

    hero_fractions = _encode_fraction([x for x in _collect_heroes(learn_data)], learn_data)
    test_hero_fractions = _encode_fraction([x for x in _collect_heroes(test_data)], test_data)

    data = learn_data.drop(unusable_headers, axis=1)  # type: DataFrame
    test_data.drop(drop_keys, axis=1, inplace=True)
    data = _fill_cols(data, _get_cols_with_nans(data), sys.maxsize)
    test_data = _fill_cols(test_data, _get_cols_with_nans(test_data), sys.maxsize)
    _gradient_classify(data.values, Y.values)

    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    _logistic_regression_classify(data, Y.values)

    data = learn_data.drop(unusable_headers + drop_keys, axis=1)  # type: DataFrame
    data = _fill_cols(data, _get_cols_with_nans(data), sys.maxsize)
    data = scaler.fit_transform(data)
    test_data = scaler.transform(test_data)
    _logistic_regression_classify(data, Y.values)

    data = np.hstack((data, hero_fractions))
    test_data = np.hstack((test_data, test_hero_fractions))
    fitted_clf = _logistic_regression_classify(data, Y.values)
    res = fitted_clf.predict_proba(test_data)[:, 1]

    print(res.min())
    print(res.max())
