import tensorflow as tf
import numpy as np
from load_cifar import extract
from load_cifar import load_batch
from load_cifar import one_hot_encode
from preprocess import rotate_reshape
from preprocess import subtract_mean_rgb
from preprocess import rescale

tarpath = "cifar-10-python.tar.gz"
data_path = extract(tarpath)
images_1d, labels_1d = load_batch(data_path[0])

# Images are (32, 32, 3)
n_data = [32, 32, 3]
# There are 10 possible outputs
n_labels = len(np.unique(labels_1d))
labels = one_hot_encode(labels_1d, n_labels)
# Reshape and rotate 1d vector into image
images_raw = rotate_reshape(images_1d, n_data)
# Rescale images to 224,244
images_rescaled = rescale(images_raw, [224, 224, 3])
# Subtract mean RGB value from every pixel
images = subtract_mean_rgb(images_rescaled)

g = tf.Graph()
with g.as_default():
    X = tf.placeholder(tf.float32, [None, n_data], name="image")
    y = tf.placeholder(tf.float32, [None, n_labels], name="label")


with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
