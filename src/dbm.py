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
    Here, DBM can classify ONLY for binaly classification problems.
    """

    def __init__(self,
                 estimator,
                 delta=0.1,
                 out=0.1,
                 eps=0.1,
                 n=10):
        """
        Args:
        estimator : Estimator object implementing 'fit'.
        delta (float) : The vale for definision of neighbor of a decision boundary
        out (float) : The value for definision of neighbor of outlier
        eps (float) : The value for definision of neighbor of a support vector
        n (int) : The number of data that are geneted around a support vector
        """

        self.estimator = estimator
        self.delta = delta
        self.out = out
        self.eps = eps
        self.n = n

    def generate_data(self, X, y):

        svm = SVC(kernel='rbf')
        svm.fit(X, y)

        cond = svm.decision_function(X) * y < - self.out
        X = np.delete(X, np.where(cond)[0], 0)
        y = np.delete(y, np.where(cond)[0], 0)

        support_vectors = svm.support_vectors_

        X = np.r_[X, support_vectors]
        y = np.r_[y, svm.predict(support_vectors)]

        for sv in support_vectors:
            np.random.seed()
            svs = np.array([sv for i in range(self.n)])
            gv = np.random.uniform(-self.eps, self.eps, (self.n, X.shape[1]))
            gv = svs + gv

            d = svm.decision_function(svs)[0]
            cond1 = abs(svm.decision_function(gv)) < self.delta
            cond2 = abs(svm.decision_function(gv)) > abs(d)

            gv = np.delete(gv, np.where(cond1), 0)
            gv = np.delete(gv, np.where(cond2), 0)

            if gv.shape[0] != 0:
                X = np.r_[X, gv]
                y = np.r_[y, svm.predict(gv)]

        X, y = shuffle(X, y, random_state=np.random.RandomState())

        return X, y

    def fit(self, X, y):
        _X, _y = self.generate_data(X, y)
        return self.estimator.fit(X, y)

    def predict(self, X):
        return self.estimator.predict(X)


def main():
    from sklearn.neural_network import MLPClassifier
    from sklearn import preprocessing, model_selection
    from sklearn.datasets import fetch_mldata

    db_name = 'australian'

    data_set = fetch_mldata(db_name)
    data_set.data = preprocessing.scale(data_set.data)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        data_set.data, data_set.target, test_size=0.4)

    mlp = MLPClassifier(solver='sgd', alpha=1e-5,
                        hidden_layer_sizes=(2,), activation='logistic', learning_rate_init=0.5)

    mlp = mlp.fit(X_train, y_train)
    print("MLP Accuracy %0.3f " % mlp.score(X_test, y_test))

    mlp = MLPClassifier(solver='sgd', alpha=1e-5,
                        hidden_layer_sizes=(2,), activation='logistic', learning_rate_init=0.5)

    dbm = DBM(mlp).fit(X_train, y_train)
    print("DBM-MLP Accuracy %0.3f " % dbm.score(X_test, y_test))


if __name__ == "__main__":
    main()
