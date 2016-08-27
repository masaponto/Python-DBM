#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.datasets import fetch_mldata
from sklearn import cross_validation
from sklearn.datasets import load_svmlight_file
from sklearn.utils import shuffle


class TLMLP(BaseEstimator):
    """
    Multi Layer Perceptron (Sngle hidden layer)

    """

    def __init__(self, hid_num=10, epochs=1000, r=0.5):
        """
        mlp using sigmoid
        Args:
        hid_num int : number of hidden neuron
        out_num int : number of output neuron
        epochs int: number of epoch
        r float: learning rate
        """
        self.hid_num = hid_num
        self.epochs = epochs
        self.r = r

    def _sigmoid(self, x, a=1):
        """
        sigmoid function
        Args:
        x float
        Returns:
        float
        """

        return 1 / (1 + np.exp(-a * x))

    def _dsigmoid(self, x, a=1):
        """
        diff sigmoid function
        Args:
        x float
        Returns:
        float
        """
        return a * x * (1.0 - x)

    def _add_bias(self, x_vs):
        """
        add bias to list

        Args:
        x_vs [[float]] Array: vec to add bias

        Returns:
        [float]: added vec
        """

        return np.c_[x_vs, np.ones(len(x_vs))]

    def _ltov(self, n, label):
        """
        trasform label scalar to vector
        Args:
        n (int) : number of class, number of out layer neuron
        label (int) : label
        Exmples:
        >>> mlp = MLP(10, 3)
        >>> mlp._ltov(3, 1)
        [1, -1, -1]
        >>> mlp._ltov(3, 2)
        [-1, 1, -1]
        >>> mlp._ltov(3, 3)
        [-1, -1, 1]
        """
        return [-1 if i != label else 1 for i in range(1, n + 1)]

    def _vtol(self, vec):
        """
        tranceform vector (list) to label
        Args:
        v: int list, list to transform

        Returns:
        int : label of classify result

        Exmples:
        >>> p = MLP(10, 3)
        >>> p.out_num = 3
        >>> p._vtol([1, -1, -1])
        1
        >>> p._vtol([-1, 1, -1])
        2
        >>> p._vtol([-1, -1, 1])
        3
        """

        if self.out_num == 1:
            return 1 if 1 <= round(vec[0], 0) else -1
        else:
            v = list(vec)
            return int(v.index(max(v))) + 1

    def _calc_out(self, w, x):
        return self._sigmoid(np.dot(w, x))

    def _out_error(self, z, y):
        return (z - y) * self._dsigmoid(z)

    def _hid_error(self, z, eo):
        return np.dot(self.wo.T, eo) * self._dsigmoid(z)

    def _w_update(self, w, e, z):
        e = np.atleast_2d(e)
        z = np.atleast_2d(z)
        return w - self.r * np.dot(e.T, z)

    def fit(self, X, y):
        """
        学習
        Args:
        X [[float]] array : featur vector
        y [int] array : class labels
        """

        self.out_num = max(y)

        y = np.array([self._ltov(self.out_num, _y)
                      for _y in y]) if self.out_num != 1 else y
        X = self._add_bias(X)

        np.random.seed()
        self.wh = np.random.uniform(-1.0, 1.0, (self.hid_num, X.shape[1]))
        self.wo = np.random.uniform(-1.0, 1.0, (self.out_num, self.hid_num))

        for n in range(self.epochs):
            X, y = shuffle(X, y, random_state=np.random.RandomState())
            for _x, _y in zip(X, y):

                # forward phase
                # 中間層の結果
                zh = self._calc_out(self.wh, _x)
                zh[-1] = -1.

                # 出力層の結果
                zo = self._calc_out(self.wo, zh)

                # backward phase
                # 出力層の誤差
                eo = self._out_error(zo, _y)

                # 中間層
                eh = self._hid_error(zh, eo)

                # weight update
                # 出力層
                self.wo = self._w_update(self.wo, eo, zh)
                # 中間層
                self.wh = self._w_update(self.wh, eh, _x)

    def predict(self, x):
        """
        Args:
        x_vs [[float]] array
        """
        x = self._add_bias(x)
        y = self._calc_out(self.wo, self._calc_out(self.wh, x.T))

        return np.array([self._vtol(_y) for _y in y.T])


def main():
    #db_names = ['iris', 'australian']
    db_names = ['australian']
    hid_nums = [5]

    for db_name in db_names:
        print(db_name)
        for hid_num in hid_nums:
            print(hid_num)
            # load iris data set
            data_set = fetch_mldata(db_name)
            data_set.data = preprocessing.scale(data_set.data)

            mlp = TLMLP(hid_num, 1000)
            mlp.fit(data_set.data, data_set.target)
            re = mlp.predict(data_set.data)
            score = sum([r == y for r, y in zip(re, data_set.target)]
                        ) / len(data_set.target)
            print("Accuracy %0.3f " % score)


if __name__ == "__main__":
    #import doctest
    # doctest.testmod()
    main()
