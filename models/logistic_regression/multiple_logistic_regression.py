"""
Quick and dirty implementation of logistic regression on n-dimensional data
using NumPy
"""
import numpy as np
from sklearn.metrics import accuracy_score


class LogisticRegression:
    """
    Logistic regression on n-dimensional data for 2 classes


    PARAMETERS
    ----------
    K: float
        Number of classes
    """
    def __init__(self, K=2):
        self.W = []
        self.K = K

    def fit(self, data, labels):
        """ Fit model weights to given data using maximum likelihood

        PARAMETERS
        ----------
        data: matrix (n x m)
        labels: list (n x 1)

        RETURNS
        -------
        weights: list Ex. - [B_0, B_1,...,B_n]
            Model weights
        """
        weights = np.array([])
        self.W = weights

    def predict(self, data):
        """ Use weights to predict labels of datapoints

        PARAMETERS
        ----------
        data: matrix (n x m)

        RETURNS
        ------
        predictions: list Ex. - [y_pred_0, y_pred_1...y_pred_n]
            List of predictions
        """
        if len(self.W) != (len(data[0]) - 1):
            raise Exception("Model isn't fit to this dimensional data")
        predictions = []
        for dp in data:
            model = np.exp(self.W[0] + np.dot(self.W[1:], dp))
            pr = []
            pr.append(1/(1+model))
            pr.append(model/(1+model))
            predictions.append(np.argmax(pr))
        return predictions


if __name__ == "__main__":
    # Each row is a datapoint, last index is label
    data = [[3, 3, 1],
            [0, 2, 0],
            [1, 4, 0],
            [4, 6, 1],
            [2, 2, 0]]
    data = np.array(data)
    x_train = data[:-1, :-1]
    x_test = data[-1, :-1]
    y_train = data[:-1, -1]
    y_test = data[-1, -1]
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    predictions = lr.predict(x_test)
    print(accuracy_score(predictions, y_test))
