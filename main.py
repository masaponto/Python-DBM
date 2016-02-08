#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.datasets import fetch_mldata
from sklearn import cross_validation
from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVC
from sklearn.utils import shuffle

from dbm.dbm import DBM
from dbm.three_layer_mlp import TLMLP

from multiprocessing import Pool


def argwrapper(args):
    '''
    ラッパー関数
    '''
    return args[0](*args[1:])


def my_cv(estimator, data_set, cv=5):
    '''
    find cross_validation accuracy
    '''
    from sklearn.utils import shuffle
    X, y = shuffle(data_set.data, data_set.target,
                   random_state=np.random.RandomState())

    n = data_set.data.shape[0]
    k = n // cv
    # print(n,cv,k)
    scores = []

    for index in range(0, n - (n % cv), k):
        #e = estimator
        x_test = X[index: index + k]
        y_test = y[index: index + k]

        x_train1 = X[:index]
        y_train1 = y[:index]

        x_train2 = X[index + k:]
        y_train2 = y[index + k:]

        x_train = np.r_[x_train1, x_train2]
        y_train = np.r_[y_train1, y_train2]

        estimator.fit(x_train, y_train)
        re = estimator.predict(x_test)
        score = sum([r == y for r, y in zip(re, y_test)]) / len(y_test)
        scores.append(score)

    # print(scores)
    return np.average(scores)


def pararell_cv_mlp(data_set, hid_num=10, epochs=1000):
    p = Pool(8)
    print('==MLP', 'hid_num', hid_num, 'epochs', epochs)
    cv = 5
    mlp = TLMLP(hid_num, epochs)
    func_args = [(my_cv, mlp, data_set, cv) for i in range(10)]
    score_list = p.map(argwrapper, func_args)
    print("Accuracy %0.3f (+/- %0.3f)" %
          (np.average(score_list), np.std(score_list) * 2))


def pararell_cv_dbm(data_set, hid_num=10, epochs=1000):
    p = Pool(8)
    print('==DBM', 'hid_num', hid_num, 'epochs', epochs)
    cv = 5
    dbm = DBM(hid_num=hid_num, epochs=epochs)
    func_args = [(my_cv, dbm, data_set, cv) for i in range(10)]
    score_list = p.map(argwrapper, func_args)
    print("Accuracy %0.3f (+/- %0.3f)" %
          (np.average(score_list), np.std(score_list) * 2))


def main():
    db_name = 'australian'
    data_set = fetch_mldata(db_name)
    data_set.data = preprocessing.scale(data_set.data)

    pararell_cv_mlp(data_set)
    pararell_cv_dbm(data_set)


if __name__ == "__main__":
    main()
