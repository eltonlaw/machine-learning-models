"""
Quick and dirty implementation of linear regression on 1D data using NumPy
"""
import numpy as np


def fit_linear_regression_1d(data):
    """
    Linear Regression on 1D Data

    Using formula 3.4 from 'An introduction to statistical learning' by
    Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani

    PARAMETERS
    ----------
    data: np.ndarray [[x_0, y_0]...[x_n, y_n]]
        1D labelled data

    RETURNS
    -------
    list: [B_0, B_1]
        Model parameters
    """
    n = len(data)
    x_bar = data.T[0].sum()/n
    y_bar = data.T[1].sum()/n
    x_bar_dev = data.T[0] - x_bar
    y_bar_dev = data.T[1] - y_bar
    B_1 = np.dot(x_bar_dev, y_bar_dev)/np.dot(x_bar_dev, y_bar_dev)
    B_0 = y_bar - B_1 * x_bar
    beta = [B_0, B_1]
    return beta


def predict_linear_regression_1d(data, beta):
    predictions = beta[1] * data.T[0] + beta[1]
    MSE = np.square(data.T[1] - predictions).mean()
    act_pred = list(zip(data.T[1], predictions))
    return MSE, act_pred


if __name__ == "__main__":
    data = np.array([[1, 1],
                    [2, 4],
                    [5, 7],
                    [0, -1],
                    [5, 5]])
    beta = fit_linear_regression_1d(data)
    print("Beta: {}".format(beta))
    # Beta: [0.60000000000000009, 1.0]
    MSE, act_pred = predict_linear_regression_1d(data, beta)
    print("Mean Squared Error: {}".format(MSE))
    # Mean Squared Error: 1.6
    print("(Actual, Prediction): {}".format(act_pred))
    # (Actual, Prediction):[(1, 2.0), (4, 3.0), (7, 6.0), (-1, 1.0), (5, 6.0)]
