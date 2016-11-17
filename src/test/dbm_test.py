#!/usr/bin/env python
import sys
import unittest
import numpy as np

sys.path.append('../')
from svm_dbm import DBM
from sklearn.neural_network import MLPClassifier


class TestDbmMethods(unittest.TestCase):

    def test_delete_vectors(self):
        x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        y = np.array([1, 2, 3, 4])

        _x = np.array([[4, 5, 6], [7, 8, 9]])

        new_x = self.dbm._DBM__delete_vectors(x, y < 2)
        self.assertEqual(np.array_equal(new_x, _x), True)

    def setUp(self):
        self.dbm = DBM(MLPClassifier())


if __name__ == "__main__":
    unittest.main()
