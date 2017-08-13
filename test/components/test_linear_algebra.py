""" components.linear_algebra.py"""
import unittest
import numpy as np
import models.components.linear_algebra as la

# pylint: disable=missing-docstring, invalid-name
class TestCalculations(unittest.TestCase):
    """ Test small calculations """
    def test_transpose(self):
        A1 = [[1, 2], [3, 4]]
        self.assertEqual(la.transpose(A1), np.transpose(A1).tolist())

    def test_shape(self):
        A1 = [[1, 2], [3, 4]]
        self.assertEqual(la.shape(A1), np.shape(A1))

    def test_dot(self):
        v1 = [1, 3, 5]
        v2 = [2, 4, 1]
        val = la.dot(v1, v2)
        self.assertEqual(val, 19)

    def test_matmul(self):
        A1 = [[1, 2], [3, 4]]
        A2 = [[5, 6], [7, 8]]
        M = la.matmul(A1, A2)
        self.assertEqual(M, [[19, 22], [43, 50]])


class TestUpper(unittest.TestCase):
    """ Test functionality of matrix elimination to upper

    Tests
    -----
    test_upper_identity

    """
    def setUp(self):
        self.I = [[1, 0], [0, 1]]

    def test_upper_identity(self):
        self.assertEqual(la.upper(self.I), la.upper(self.I))

class TestDeterminant(unittest.TestCase):
    """ Test output of determinant algorithm

    Tests
    -----
    test_determinant_identity

    """
    def setUp(self):
        self.I = [[1, 0], [0, 1]]
        self.U = [[2, 3], [0, 5]]

    def test_determinant_identity(self):
        self.assertEqual(la.det(self.I), np.linalg.det(self.I))

    def test_determinant_upper(self):
        self.assertEqual(la.det(self.I), np.linalg.det(self.I))

if __name__ == "__main__":
    unittest.main()
