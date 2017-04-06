""" ...Significant portions taken from sentdex's Deep Learning tut"""
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf


def get_data(dataset):
    """
    Fetches datasets from 'mldata.org' using sklearn.

    Parameters
    ----------
    dataset: string
        Name of dataset to pull from mldata.org

    Returns
    -------
    tensor
        Training set datapoints
    tensor
        Test set datapoints
    tensor
        Training set labels
    tensor
        Test set labels

    """
    mnist = fetch_mldata(dataset)
    # One hot encoding
    distinct_outputs = 10
    temp_labels = np.zeros((70000, distinct_outputs))
    for i, label in enumerate(mnist.target):
        temp_labels[i][int(label)] = 1
    labels = temp_labels
    # Split data into train/test set
    x_train, x_test, y_train, y_test = train_test_split(mnist.data,
                                                        labels,
                                                        test_size=0.3,
                                                        random_state=1)
    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = get_data("MNIST original")

# Parameters
n_inputs = len(x_train[0])
n_outputs = len(y_train[0])
batch_size = 100

x = tf.placeholder(tf.float32, shape=[1, n_inputs])
y_ = tf.placeholder(tf.float32, shape=[1, n_outputs])


def add_layer(node_i, node_o):
    W = tf.Variable(tf.random_uniform([node_i, node_o], minval=-1, maxval=1))
    B = tf.Variable(tf.random_uniform([1, node_o], minval=0, maxval=1))
    return W, B


def model(data):
    W1, B1 = add_layer(784, 300)
    W2, B2 = add_layer(300, 200)
    W3, B3 = add_layer(200, 10)

    L1 = tf.matmul(data, W1) + B1
    L1 = tf.nn.relu(L1)
    L2 = tf.matmul(L1, W2) + B2
    L2 = tf.nn.relu(L2)
    output = tf.matmul(L2, W3) + B3
    return output

#    layers = [n_inputs, 300, 200, n_outputs]
#    wbs = []
#    for node_i, node_o in zip(layers[:-1], layers[1:]):
#        W, B = add_layer(node_i, node_o)
#        wbs.append({"weights": np.transpose(W), "biases": B})
#    for i, _ in enumerate(layers):
#        if i == 0:
#            z = tf.matmul(data, wbs[i]["weights"])
#            z = z + wbs[i]["biases"]
#            a = tf.nn.relu(z)
#            continue
#        z = tf.matmul(a, wbs[i]["weights"])
#        z = z + wbs[i]["biases"]
#        a = tf.nn.relu(z)
#    return z


def next_batch(x, y, batch_i, batch_size):
    i_start = batch_i*batch_size
    i_stop = (batch_i+1)*batch_size
    return x[i_start:i_stop], y[i_start:i_stop]


def train(x):
    y_ = model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_, y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    batch_size = 100
    epochs = 10

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_i in range(int(len(x_train)/batch_size)):
                epoch_x, epoch_y = next_batch(x_train, y_train, batch_i, batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
        score = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(score, "float"))
        print("Accuracy:{}".format(accuracy.eval({x: x_test, y: y_test})))

        save_path = saver.save(sess, "./temp/model.ckpt")
        print("Model saved in file: {}".format(save_path))


train(x)
