"""
Ordinary Least Squares Regression with Minibatch Gradient Descent and
in closed form (using projections)
"""
import numpy as np

# pylint: disable=invalid-name
class LinearRegression:
    """
    Outputs/predictions are modelled as linear combinations of outputs.

    Attributes
    ----------
    normalized: boolean, optional, default True
        Normalize data
    method: str
        Optimization method, options are 'gd' (for Gradient Descent)
        and 'cf (for Closed Form)'

    Methods
    -------
    fit(X, y):
        Learn linear mapping form X (some data) to y (some label)
    predict(X):
        Use previously learning mapping on novel data, X

    """
    def __init__(self, normalized=True, method="gd"):
        self.method = method
        self.fitted = False
        self.normalized = normalized

    # pylint:disable=too-many-arguments,too-many-locals
    def fit(self, X, y, batch_size=1, lr=0.0001, epochs=100):
        """ Learn linear mapping from X (some data) to y (some label)

        The hyperparameters `batch_size`, `lr` and `epochs` are only needed
        if using the method 'gd' (Gradient Descent).

        Parameters
        ----------
        X: array
            Input data of shape [m, n] where `m` is the number of data
            points and `n` is the number of features/columns
        y: array
            Input data of shape [m, 1] or [m] where `m` is the number of
            data points
        batch_size : int, optional, default 1
            The number of data points used in each update of the weights.
            Keep in mind that `batch_size` must be divisible by the number
            of data points.
        lr : float, optional, default 0.001
            Learning rate/alpha, the percentage of the gradient that gets
            subtracted from the weights.
        epochs : int, optional, default 100
            Number of times entire training set is looped over
        Returns
        ------
        None

        """
        if self.method == "gd":
            n_data, n_features = np.shape(X)
            # Number of data points must be divisable by batch size
            assert n_data % batch_size == 0
            n_batches = n_data//batch_size
            if self.normalized:
                X = _normalize(X)

            weights = np.random.standard_normal((n_features+1))
            for _ in range(epochs):
                for i in range(n_batches):
                    # Get the next batch of training data
                    x_i, y_i = [X[i*batch_size:(i+1)*batch_size],
                                y[i*batch_size:(i+1)*batch_size]]
                    # Add 1 to the beginning for the intercept
                    x_i = np.insert(x_i, 0, 1)
                    # Use weights to get prediction
                    y_hat = [np.matmul(x_i, weights)]
                    loss_derivative = (y_hat - y_i)*x_i
                    # Update weights
                    weights = weights - (lr * loss_derivative)
        elif self.method == "cf":
            if self.normalized:
                X = _normalize(X)
            # pylint: disable=line-too-long
            weights = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)

        # pylint: disable=attribute-defined-outside-init
        self.weights = weights
        self.fitted = True

    def predict(self, X):
        """ Use previously learning mapping on novel data, X

        Parameters
        ----------
        X: array
            Input data of shape [m, n] where `m` is the number of data
            points and `n` is the number of features/columns

        Returns
        ------
        y_hat: array
            Vector of predictions for `X`, each element corresponds with
            one of the rows in `X`. (Ex. y_[i] is the predicted label for
            X[i])

        """
        assert self.fitted
        if self.normalized:
            X = _normalize(X)
        if self.method == "gd":
            X = np.insert(X, 0, 1, axis=1)
        y_hat = np.matmul(X, self.weights)
        return y_hat

def _normalize(vector):
    norm = np.linalg.norm(vector)
    if norm != 0:
        out = vector/norm
    else:
        out = vector
    return out

def test_run(dataset_name, method):
    """ Run the model on a toy dataset """
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import explained_variance_score
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    ds = getattr(datasets, ("load_{}").format(dataset_name))()
    X_train, X_test, y_train, y_test = train_test_split(ds.data,
                                                        ds.target,
                                                        test_size=0.3)
    model = LinearRegression(method=method)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("="*79)
    print("{}-{}".format(dataset_name.upper(), method))
    print("Weights:", model.weights)
    print("MSE:", mean_squared_error(predictions, y_test))
    print("EV:", explained_variance_score(predictions, y_test))


if __name__ == "__main__":
    test_run("boston", method="gd")
    test_run("boston", method="cf")
    # _test_run("linnerud")
