"""
Quick and dirty implementation of logistic regression on n-dimensional data
using NumPy
"""
import numpy as np


# pylint: disable=invalid-name
def ohe(vec, K=None):
    """ One Hot Encoding"""
    if not K:
        K = np.unique(vec)
    labels = []
    for label in vec.astype(np.int):
        temp = np.zeros(K)
        temp[label] = 1
        labels.append(temp)
    return np.array(labels)

def sigmoid(A):
    """ Maps from R to [0, 1]

    Function Behaviour
    ------------------
    If the input approaches positive infinity, output approaches 1
    If the input is 0, output is 1/2
    If the input approaches negative infinity, output approaches 0
    """
    return 1/(1+np.exp(-(A)))

def loss(y, y_hat):
    """ Calculates the categorical classification loss

    Parameters
    ----------
    y:
        Ground truth label
    y_hat:
        Prediction from model
    """
    return -(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))


class LogisticRegression:
    """
    Logistic regression on n-dimensional data for 2 classes


    Parameters
    ----------
    K: float
        Number of classes
    """
    def __init__(self, K=3):
        self.W = []
        self.B = []
        self.K = K

    def fit(self, data, labels):
        """ Fit model weights to given data using maximum likelihood

        Parameters
        ----------
        data: matrix
            Shape (m, n) where m = # training points and n = # features
        labels: list
            Shape (m x l) where l = # of labels

        Returns
        -------
        weights: list Ex. - [B_0, B_1,...,B_n]
            Model weights
        """
        labels = ohe(labels, self.K)
        n_x, n_features = np.shape(data)
        n_x, n_labels = np.shape(labels)
        self.W = np.random.normal(size=(n_features, n_labels))
        self.B = np.random.normal(size=(n_labels))
        Z = sigmoid(np.matmul(data, self.W) + self.B)
        # Derivative of cost with respect to output
        d_z = Z - labels
        self.W += (1/n_x)*np.matmul(data.T, d_z)
        self.B += (1/n_x)*np.sum(d_z, axis=0)


    def predict(self, data):
        """ Use weights to predict labels of datapoints

        Parameters
        ----------
        data: matrix (n x m)

        Returns
        ------
        predictions: list Ex. - [y_pred_0, y_pred_1...y_pred_n]
            List of predictions
        """
        return sigmoid(np.matmul(data, self.W) + self.B)

if __name__ == "__main__":
    # Each row is a datapoint, last index is label
    sample_data = np.array([[3, 3, 1],
                            [0, 2, 0],
                            [1, 4, 0],
                            [4, 6, 1],
                            [3, 2, 2],
                            [4, 4, 2],
                            [5, 6, 2],
                            [2, 2, 2]])
    x_train = sample_data[:-1, :-1]
    x_test = sample_data[-2:, :-1]
    y_train = sample_data[:-1, -1]
    y_test = sample_data[-2:, -1]
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    preds = lr.predict(x_test)
    total, acc = 0, 0
    for pred, y_test_i in zip(preds, y_test):
        total += 1.
        if np.argmax(pred) == int(y_test_i):
            acc += 1.
    print("accuracy score: {}".format(acc/total))
