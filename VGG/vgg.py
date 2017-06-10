import tensorflow as tf
import numpy as np
from load_cifar import extract
from load_cifar import load_batch
from load_cifar import one_hot_encode

tarpath = "cifar-10-python.tar.gz"
data_path = extract(tarpath)
images, labels_1d = load_batch(data_path[0])

# Images are (32, 32, 3)
n_data = (32, 32, 3)
# There are 10 possible outputs
n_labels = len(np.unique(labels_1d))
labels = one_hot_encode(labels_1d, n_labels)

# LAYER ARCHITECTURE
# Layer 1
n = 224


def rotate_reshape(img):
    for i, img in enumerate(images):
        img = np.reshape(img, [32, 32, 3], order="F")
        img = np.rot90(img, k=3)
        images[i] = img
    return images


def conv_layer(X, filters, kernel, stride, name="convolutional"):
    with tf.name_scope(name):
        pass


def fc_layer(X, n_in, n_out, name="fully_connected"):
    with tf.name_scope(name):
        with tf.name_scope("weights"):
            W = tf.Variable(tf.truncated_normal([n_in, n_out]),
                            name="weights")
        with tf.name_scope("biases"):
            B = tf.Variable(tf.ones(n_out)/10, name="biases")
        with tf.name_scope("activations"):
            A = tf.nn.relu(tf.matmul(X, W) + B, name="activations")
        return A


def logits_layer(X, n_in, n_out, name="logits"):
    with tf.name_scope(name):
        with tf.name_scope("weights"):
            W = tf.Variable(tf.truncated_normal([n_in, n_out]),
                            name="weights")
        with tf.name_scope("biases"):
            B = tf.Variable(tf.zeros(n_out)/10, name="biases")
        with tf.name_scope("activations"):
            Z = tf.matmul(X, W) + B
        return Z


g = tf.Graph()
with g.as_default():
    X = tf.placeholder(tf.float32, [None, n_data], name="image")
    y = tf.placeholder(tf.float32, [None, n_labels], name="label")
    A = conv_layer(X, 3, (11, 11), 4)


with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
