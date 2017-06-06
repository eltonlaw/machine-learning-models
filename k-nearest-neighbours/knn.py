import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


def load_iris():
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    iris = load_iris()
    return train_test_split(iris.data, iris.target)


def load_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    m = input_data.read_data_sets("../MNIST_data/", one_hot=False)
    return m.train.images, m.test.images, m.train.labels, m.test.labels


def distance(v1, v2):
    """ Calculate euclidean distance between vector 1 and vector 2 """
    assert np.shape(v1) == np.shape(v2)
    return np.linalg.norm(v1-v2)


def mode(A):
    return np.argmax(np.bincount(A))


def fit_predict(train_data, train_labels, test_data, nn=3):
    """ K-Nearest Neighbour Classifier

    """
    predictions = []
    for x_test in test_data:
        distances = []
        for x_train, y_train in zip(train_data, train_labels):
            distances.append(distance(x_test, x_train))
        neighbours_i = np.argsort(distances)[:nn]
        neighbours_labels = [train_labels[i] for i in neighbours_i]
        predictions.append(mode(neighbours_labels))
    return predictions


if __name__ == "__main__":
    # train_data, test_data, train_labels, test_labels = load_iris()
    train_data, test_data, train_labels, test_labels = load_mnist()
    predictions = fit_predict(train_data, train_labels, test_data)
    score = accuracy_score(test_labels, predictions)
    print("Test Set Accuracy: {}".format(score))
    # clf = KNeighborsClassifier(n_neighbors=3)
    # clf.fit(train_data, train_labels)
    # predictions = clf.predict(test_data)
    # score = accuracy_score(test_labels, predictions)
    # print("Test Set Accuracy: {}".format(score))
