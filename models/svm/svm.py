""" Support Vector Machine Implementation """
# import tensorflow as tf
# pylint: disable=import-error
from kernel import Kernel


def load_split_iris():
    """ Use sklearn to load iris datasets """
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    iris = load_iris()
    # pylint: disable=no-member
    return train_test_split(iris.data, iris.target)


# pylint: disable=invalid-name
class SVM:
    """ Support Vector Machine

    A machine learning algorithm that finds the hyperplane which maximizes the
    seperation between classes.
    """
    def __init__(self, ktype="linear"):
        self.k = Kernel()
        self.ktype = ktype
        # getattr(self.k, self.ktype)()

    def fit(self, X, y):
        """ Fit SVM to data given"""
        pass

    def predict(self, X):
        """ Predict data given using fitted model """
        pass

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_split_iris()
    model = SVM(ktype="linear")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
