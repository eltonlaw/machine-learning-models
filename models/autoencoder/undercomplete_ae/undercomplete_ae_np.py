"""No-library implementation of a standard 1-layer undercomplete autoencoder"""
# import numpy as np
from utils import Plotter
from sklearn.datasets import fetch_mldata


# pylint: disable=invalid-name
class UAE:
    """ Undercomplete Autoencoder

    Attributes
    ----------
    params: dict, optional, default True
        A dictionary containing any custom hyperparameters you want to pass
        to the model

    Methods
    -------
    fit(X):
        Learn non-linear mapping from X back to X

    """
    # pylint: disable=dangerous-default-value
    def __init__(self, parameters={}):
        """ If it exists, use user-defined parameters, otherwise use defaults

        Parameters
        ---------
        parameters: dict
            Model hyperparameters. Supports 'lr', 'batch_size'.

        """
        defaults = {
            "lr": 0.0005,
            "batch_size": 256
            }
        for param_name, value in defaults.items():
            if param_name in parameters:
                value = parameters[param_name]
            setattr(self, param_name, value)

    def train(self, X):
        """ Fit model to training data """
        pass

    def test(self, X):
        """ Reconstruct input/test data """
        pass

if __name__ == "__main__""":
    mnist = fetch_mldata('MNIST original')
    plt = Plotter()
    print("=== Job Complete ===")
