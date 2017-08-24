""" Simplest possible example of a Generative Adversarial Network """
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# from models.components.plot_util import Plotter


# pylint:disable=invalid-name, missing-docstring, too-few-public-methods
flags = tf.app.flags
flags.DEFINE_integer("epochs", 10, "")
flags.DEFINE_float("learning_rate", 0.0001, "")
flags.DEFINE_integer("batch_size", 64, "")
FLAGS = flags.FLAGS

def dense_relu(X, weight_shape, bias_shape):
    """ Dense ReLU layer"""
    weights = tf.get_variable("weights", weight_shape,
                              initializer=tf.random_normal_initializer())
    biases = tf.get_variable("biases", bias_shape,
                             initializer=tf.constant_initializer(0.01))
    return tf.nn.relu(tf.matmul(X, weights) + biases)

def logits(X, weight_shape, bias_shape):
    """ Output Layer """
    weights = tf.get_variable("weights", weight_shape,
                              initializer=tf.random_normal_initializer())
    biases = tf.get_variable("biases", bias_shape,
                             initializer=tf.constant_initializer(0.01))
    return tf.matmul(X, weights) + biases

def conv_relu(X, kernel_shape, bias_shape):
    """ Convolutional ReLU layer"""
    weights = tf.get_variable("weights", kernel_shape,
                              initializer=tf.random_normal_initializer())
    biases = tf.get_variable("biases", bias_shape,
                             initializer=tf.constant_initializer(0.01))
    conv = tf.nn.conv2d(X, weights, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.relu(conv + biases)

def max_pool_2d(X, shape=(1, 2, 2, 1), stride=(1, 2, 2, 1)):
    """ Max pooling with stride and pool 2"""
    return tf.nn.max_pool(X, shape, stride, padding="SAME")

# pylint: disable=no-member, too-many-locals
class Generator:
    """ Generate data from RNG generated vectors """
    def __init__(self, batch_size, lr, sess):
        """ Build Graph """
        self.lr = lr
        self.sess = sess
        self.batch_size = batch_size
        # Layer nodes
        L = [1000, 3136, 3136]
        # Conv kernel channel outs
        C = [16, 16, 1]
        self.x = tf.random_normal([batch_size, L[0]])
        self.y = tf.random_normal([batch_size, 1])

        with tf.variable_scope("h1_g"):
            h1 = dense_relu(self.x, [L[0], L[1]], L[1])
        with tf.variable_scope("h2_g"):
            h2 = dense_relu(h1, [L[1], L[2]], L[2])

        h2_2d = tf.reshape(h2, [batch_size, 56, 56, 1])
        with tf.variable_scope("h3_g"):
            h3 = conv_relu(h2_2d, [3, 3, 1, C[0]], C[0])
        with tf.variable_scope("h4_g"):
            h4 = conv_relu(h3, [3, 3, C[0], C[1]], C[1])
            h4_p = max_pool_2d(h4)
        with tf.variable_scope("h5_g"):
            image_ = conv_relu(h4_p, [3, 3, C[1], C[2]], C[2])


        self.image_ = image_
        self.dim_in = L[0]

    def train(self, discriminator):
        images_2d = np.reshape(self.sess.run(self.image_), (64, 784))
        y_ = discriminator.predict(images_2d)

        with tf.name_scope("cross_entropy_g"):
            cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.y,
                    logits=y_),
                name="cross_entropy_g")


        all_vars = tf.trainable_variables()
        vars_to_train = [var for var in all_vars if "_g" in var.name]
        print("====== Vars to Train ========")
        print(vars_to_train)
        print("=============================")
        with tf.name_scope("train_g"):
            train_op = tf.train.RMSPropOptimizer(self.lr) \
                       .minimize(cost, var_list=vars_to_train)
        self.sess.run(train_op,
                      feed_dict={self.x:self.generate(),
                                 self.y:np.zeros(self.batch_size)
                                })

    def generate(self):
        batch = np.random.normal(size=(self.batch_size, self.dim_in))
        return self.sess.run(self.image_, feed_dict={self.x: batch})

class Discriminator:
    def __init__(self, batch_size, lr, sess):
        """ Binary Classifier """
        self.sess = sess
        self.x = tf.placeholder(tf.float32, shape=(batch_size, 784))
        x_2d = tf.reshape(self.x, (batch_size, 28, 28, 1))
        self.y = tf.placeholder(tf.float32, shape=(batch_size, 1))

        C = [32, 64, 128]
        L = [1000, 1]

        with tf.variable_scope("h1_d"):
            h1 = conv_relu(x_2d, [3, 3, 1, C[0]], C[0])
        with tf.variable_scope("h2_d"):
            h2 = conv_relu(h1, [3, 3, C[0], C[1]], C[1])
        with tf.variable_scope("h3_d"):
            h3 = conv_relu(h2, [3, 3, C[1], C[2]], C[2])
            h3_1d = tf.reshape(h3, [batch_size, -1])
        with tf.variable_scope("h4_d"):
            h4 = dense_relu(h3_1d, [h3_1d.shape[1], L[0]], L[0])
        with tf.variable_scope("out_d"):
            y_ = logits(h4, [L[0], L[1]], L[1])

        with tf.name_scope("cross_entropy_d"):
            cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=self.y,
                    logits=y_),
                name="cross_entropy_d")

        all_vars = tf.trainable_variables()
        vars_to_train = [var for var in all_vars if "_d" in var.name]
        with tf.name_scope("train_d"):
            train_op = tf.train.RMSPropOptimizer(lr) \
                       .minimize(cost, var_list=vars_to_train)

        self.y_ = y_
        self.train_op = train_op

    def train(self, batch):
        self.sess.run(self.train_op,
                      feed_dict={self.x:batch[0],
                                 self.y:batch[1]
                                })

    def predict(self, x_test):
        return self.sess.run(self.y_, feed_dict={self.x: x_test})

class BatchLoader:
    """ Generates a shuffled batch of generated and real data"""
    def __init__(self, data):
        self.data = data
    def next_batch(self, G, batch_size):
        split = int(np.random.random() * batch_size)

        real = self.data.train.next_batch(batch_size-split)[0]
        gen = G.generate()[:split]
        gen = np.reshape(gen, (split, 784))

        data = np.append(gen, real, axis=0)
        # Assign 0 to generated images and 1 to real images
        labels = np.append(np.zeros((split, 1)),
                           np.ones((batch_size - split, 1)),
                           axis=0)
        print("============ Batch Loader ===========")
        print("labels.shape: {}".format(np.shape(labels)))
        print("gen.shape: {}".format(np.shape(gen)))
        print("data.shape: {}".format(np.shape(data)))
        print("real.shape: {}".format(np.shape(real)))
        print("=====================================")
        return (data, labels)


def main(_):
    """ Main Function"""
    batch_size = FLAGS.batch_size
    epochs = FLAGS.epochs
    lr = FLAGS.learning_rate

    with tf.Session() as sess:
        G = Generator(batch_size, lr, sess)
        D = Discriminator(batch_size, lr, sess)

        sess.run(tf.global_variables_initializer())

        mnist = input_data.read_data_sets("MNIST_data")
        BL = BatchLoader(mnist)
        for _ in range(epochs):
            batch = BL.next_batch(G, batch_size)
            D.train(batch)
            G.train(D)

if __name__ == "__main__":
    tf.app.run()
