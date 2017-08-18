""" Principal Components Analysis

Decomposition Algorithm. Maps high dimensional data to a lower dimension.
Lossy compression.

Reference:
    I. Goodfellow, Y. Bengio, A. Courville. "Deep Learning".
    Ch.2, pg.45-49
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
# pylint: disable=invalid-name, no-member, missing-docstring

def pca(data, r=2):
    """ Eigendecomposition of the covariance matrix """
    # Calculate covariance matrix/make it symmetric
    cov = np.matmul(data.T, data)
    # Find the eigenvectors of the covariance matrix
    _, eigvec = np.linalg.eig(cov)
    # Use `r` truncated eigenvectors to get truncated representation
    return np.matmul(data, eigvec[:r].T)


def save_as_scatterplot(data, labels, savepath="images/output.png"):
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("Iris PCA")
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.savefig(savepath, bbox_inches="tight")
    plt.clf()

if __name__ == "__main__":
    iris = load_iris()
    representation = pca(iris.data)
    save_as_scatterplot(representation, iris.target)
