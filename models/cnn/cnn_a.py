""" Wide/Shallow Convolutional Net with Overlapping Pooling"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
log_path = "./logs"
data_dir = "../MNIST_data"

# Session Parameters
config = tf.ConfigProto(allow_soft_placement=True,
                        log_device_placement=True)

mnist = input_data.read_data_sets(data_dir, one_hot=True)
n_data = mnist.train.images.shape[1]
n_labels = mnist.train.labels.shape[1]
im_w = 28
im_h = 28
im_d = 1

# Hyperparameters

epochs = 10
batch_size = 256
learning_rate = 1e-4
c1_kw = 3
c1_kh = 3
c1_out = 256
c1_stride = [1, 1, 1, 1]
c1_padding = "SAME"
p1_ksize = [1, 2, 2, 1] #  2x2 Pooling
p1_stride = [1, 1, 1, 1]
p1_padding ="SAME"

mnist_graph = tf.Graph()
with mnist_graph.as_default():
    x = tf.placeholder(tf.float32, [None, n_data])
    x_im = tf.reshape(x, [-1, im_h, im_w, im_d])
    y = tf.placeholder(tf.float32, [None, n_labels])

    W, B = {}, {}
    # Convolutional Layer
    W["conv1"] = tf.Variable(tf.random_normal([c1_kw, c1_kh, im_d, c1_out]))
    B["conv1"] = tf.Variable(tf.random_normal([c1_out]))
    conv1 = tf.nn.conv2d(x_im, W["conv1"], c1_stride, c1_padding)
    conv1 = tf.nn.relu(conv1+B["conv1"])

    # Max Pool
    pool1 = tf.nn.max_pool(conv1, ksize=p1_ksize, strides=p1_stride, padding=p1_padding)

    # Reshape
    pool1_1d = tf.reshape(pool1, (-1, im_w*im_h*c1_out))

    # Fully Connected Layer
    W["fc1"] = tf.Variable(tf.random_normal((im_w*im_h*c1_out, n_labels)))
    B["fc1"] = tf.Variable(tf.random_normal([n_labels]))
    y_ = tf.matmul(pool1_1d, W["fc1"]) + B["fc1"]

    # Loss
    cross_entropy = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                                            logits=y_))

    # Optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    # Evaluate
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()

with tf.Session(graph=mnist_graph, config=config) as sess:
    sess.run(init)
    # Train
    for i in range(epochs):
        batch = mnist.train.next_batch(batch_size)
        _, a = sess.run([optimizer, accuracy],
                        feed_dict={x: batch[0], y: batch[1]})
        if i % 2 == 0:
            print("{}) Training Set Accuracy: {}".format(i, a))
    # Test
    test_score = sess.run(accuracy, feed_dict={x: mnist.test.images, 
                                               y: mnist.test.labels})
    print("Test Set Accuracy: {}".format(test_score))
    print("=== DONE ===")
