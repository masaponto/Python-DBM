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

import sys
sys.path.append('./dbm')
from three_layer_mlp import TLMLP


class DBM(BaseEstimator):

    def __init__(self, hid_num=10, delta=0.1, out=0.1, eps=0.1, n=10, epochs=1000, is_del_outlier=True):
        self.hid_num = hid_num
        self.delta = delta
        self.out = out
        self.eps = eps
        self.n = n
        self.epochs = epochs
        self.is_del_outlier = is_del_outlier

    def fit(self, X, y):
        svm = SVC(kernel='rbf')
        svm.fit(X, y)

        # delete outlier
        if self.is_del_outlier:
            outer_indexs = [i for i, (x, y) in enumerate(
                zip(X, y)) if svm.decision_function(x.reshape(1, -1)) * y < - self.out]
            X = np.delete(X, outer_indexs, 0)
            y = np.delete(y, outer_indexs)

        support_vecs = svm.support_vectors_
        X = np.r_[X, support_vecs]
        y = np.r_[y, svm.predict(support_vecs)]

        for svec in support_vecs:
            svec = svec.reshape(1, -1)
            for i in range(self.n):
                np.random.seed()
                new_vec = np.array(
                    [np.random.uniform(x - self.eps, x + self.eps) for x in svec]).reshape(1, -1)

                if self.delta <= abs(svm.decision_function(new_vec)) <= svm.decision_function(svec):
                    X = np.r_[X, new_vec]
                    y = np.r_[y, svm.predict(new_vec)]

        X, y = shuffle(X, y, random_state=np.random.RandomState())
        self.mlp = TLMLP(self.hid_num, self.epochs)
        self.mlp.fit(X, y)

    def predict(self, X):
        return self.mlp.predict(X)


def main():
    db_name = 'australian'
    data_set = fetch_mldata(db_name)
    data_set.data = preprocessing.scale(data_set.data)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        data_set.data, data_set.target, test_size=0.4, random_state=0)

    dbm = DBM(hid_num=10, epochs=1000)
    dbm.fit(X_train, y_train)

    re = dbm.predict(X_test)
    score = sum([r == y for r, y in zip(re, y_test)]) / len(y_test)
    print("DBM Accuracy %0.3f " % score)


if __name__ == "__main__":
    main()
