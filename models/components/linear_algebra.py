""" Implementations of linear algebra operations

Note: Doesn't check for a rectangular-ness (i.e A = [[1,2],[3],[4,5,6]])
"""
import numpy as np

# pylint: disable=invalid-name, missing-docstring
def _is_square(A):
    """ Check to see if the matrix has the same number of cols and rows"""
    A_s = shape(A)
    return A_s[0] == A_s[1]

def shape(A):
    row_size = len(A)
    col_size = len(A[0])
    return (row_size, col_size)


def dot(v1, v2):
    val = 0
    for v1_i, v2_i in zip(v1, v2):
        val += v1_i*v2_i
    return val

def transpose(A):
    """ WIP """
    return np.transpose(A).tolist()

def matmul(A, B):
    A_s = shape(A)
    B_s = shape(B)
    if not A_s[1] == B_s[0]:
        raise Exception("Input matrices incompatible shape")
    M = []
    B_T = transpose(B)
    for row_i in range(A_s[0]):
        row = []
        for col_i in range(B_s[1]):
            row.append(dot(A[row_i][:], B_T[col_i][:]))
        M.append(row)
    return M



def upper(A):
    """ Make the matrix upper triangular"""
    return A

def det(A):
    """ Calculate the determinant of A """
    A_upper = upper(A)
    if _is_square(A_upper):
        n = len(A)
        det_val = 1
        for i in range(n):
            det_val *= A[i][i]
    return det_val
