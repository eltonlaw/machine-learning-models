import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

n_in = np.shape(mnist.train.images)[1]
n_out = 1
epochs = 100


X = tf.placeholder(tf.float32, [None, n_in])
A = tf.Variable(tf.random_normal([n_out, n_in]))
B = tf.Variable(tf.random_normal([n_out, 1]))

y_pred = tf.matmul(A, tf.transpose(X)) + B
y_actual = tf.placeholder(tf.float32, [n_out, None])
cost = tf.reduce_sum(tf.pow(y_pred - y_actual, 2))
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

correct_prediction = tf.equal(y_pred, y_actual)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    feed_dict = {X: mnist.train.images,
                 y_actual: np.reshape(mnist.train.labels, (1, 55000))}
    # Train
    for i in range(epochs):
        sess.run(optimizer, feed_dict=feed_dict)
        print("EPOCH {}".format(i))
    # Test
    print(sess.run(accuracy, feed_dict={X: mnist.test.images,
                                        y_actual: np.reshape(mnist.test.labels,
                                                             (1, 10000))}))
