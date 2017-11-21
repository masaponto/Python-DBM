#!/usr/bin/env python

import numpy as np
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, ClassifierMixin


class DBM(BaseEstimator, ClassifierMixin):

    """
    This script is for Decision Boundary Making Algorithm.
    SVM is used as high performance model.
    A compact model is from argument 'estimator'.
    Here, DBM can classify ONLY for binary classification problems.
    """

    def __init__(self,
                 estimator,
                 delta=0.1,
                 out=0.1,
                 eps=0.1,
                 n=10,
                 svm=SVC(kernel='rbf')):
        """
        Args:
        estimator : Estimator object implementing 'fit'.
        delta (float) : The vale for definition of neighbor of a decision boundary
        out (float) : The value for definition of neighbor of outlier
        eps (float) : The value for definition of neighbor of a support vector
        n (int) : The number of data that are geneted around a support vector
        """

        self.estimator = estimator
        self.delta = delta
        self.out = out
        self.eps = eps
        self.n = n
        self.svm = svm

    def __delete_outlier(self, X, y, svm):
        cond = svm.decision_function(X) * y < - self.out
        X = self.__delete_vectors(X, cond)
        y = self.__delete_vectors(y, cond)
        return X, y

    def __add_vectors(self, X, y, vectors, svm):
        X = np.r_[X, vectors]
        y = np.r_[y, svm.predict(vectors)]
        return X, y

    def __delete_vectors(self, X, cond):
        return np.delete(X, np.where(cond), 0)

    def __generate_data(self, X, y):

        self.svm.fit(X, y)

        X, y = self.__delete_outlier(X, y, self.svm)

        support_vectors = self.svm.support_vectors_
        X, y = self.__add_vectors(X, y, support_vectors, self.svm)

        for sv in support_vectors:
            np.random.seed()

            # Generate some vectors around a support vector
            gv = np.random.uniform(-self.eps, self.eps, (self.n, X.shape[1]))
            svs = np.array([sv for i in range(self.n)])
            gv = gv + svs

            # Delete generated vectors that are too colse to decision bounday
            gv_distance = self.svm.decision_function(gv)
            gv = self.__delete_vectors(gv,  abs(gv_distance) < self.delta)

            if gv.shape[0] == 0:
                continue

            # Delete generated vectors that are more distance from decision
            # boundary than that of a support vector
            gv_distance = self.svm.decision_function(gv)
            svs = np.array([sv for i in range(gv.shape[0])])
            sv_distance = self.svm.decision_function(svs)

            gv = self.__delete_vectors(gv, abs(gv_distance) > abs(sv_distance))

            if gv.shape[0] == 0:
                continue

            # Add generated vectors
            X, y = self.__add_vectors(X, y, gv, self.svm)

        return X, y

    def fit(self, X, y):
        _X, _y = self.__generate_data(X, y)
        self.estimator.fit(_X, _y)
        return self

    def predict(self, X):
        return self.estimator.predict(X)


def main():
    from sklearn.neural_network import MLPClassifier
    from sklearn import preprocessing, model_selection
    from sklearn.datasets import fetch_mldata

    db_name = 'australian'

    data_set = fetch_mldata(db_name)
    data_set.data = preprocessing.normalize(data_set.data)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        data_set.data, data_set.target, test_size=0.4)

    np.random.seed()
    mlp = MLPClassifier(hidden_layer_sizes=(
        10,), max_iter=10000, activation='logistic')

    mlp.fit(X_train, y_train)
    print("MLP Accuracy %0.3f " % mlp.score(X_test, y_test))

    np.random.seed()
    svm = SVC(kernel='rbf', C=500)
    svm.fit(X_train, y_train)
    print("SVM Accuracy %0.3f " % svm.score(X_test, y_test))

    np.random.seed()
    mlp = MLPClassifier(hidden_layer_sizes=(
        10,), max_iter=10000, activation='logistic')
    dbm = DBM(mlp, svm=SVC(kernel='rbf', C=500))
    dbm.fit(X_train, y_train)
    print("DBM-MLP Accuracy %0.3f " % dbm.score(X_test, y_test))


if __name__ == "__main__":
    main()
