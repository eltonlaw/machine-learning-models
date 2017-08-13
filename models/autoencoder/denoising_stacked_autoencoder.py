"""" Denoising Stacked Autoencoder
Paper Reference:
    Pascal Vincent, Hugo Larochelle, Isabelle Lajoie, Yoshua Bengio, and
    Pierre-Antoine Manzagol. 2010. Stacked Denoising Autoencoders: Learning
    Useful Representations in a Deep Network with a Local Denoising Criterion.
    J. Mach. Learn. Res. 11 (December 2010), 3371-3408.
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Parameters
train_dir = "../MNIST_data"
log_path = "./logs"
n_h1 = 512
n_h2 = 256
n_h3 = 128
batch_size = 256
display_step = 1
learning_rate = 0.01
training_epochs = 100


mnist = input_data.read_data_sets(train_dir, one_hot=True)
n_data = mnist.train.images.shape[1]
mnist_graph = tf.Graph()


def ae_layer(X_in, channel_in, channel_out, name="autoencoder_layer"):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal([channel_in, channel_out]),
                        name="weights")
        tf.summary.histogram("weights", W)
        B = tf.Variable(tf.truncated_normal([channel_out]), name="biases")
        tf.summary.histogram("biases", B)
        return tf.nn.sigmoid(tf.matmul(X_in, W) + B)


with mnist_graph.as_default():
    X = tf.placeholder("float", [None, n_data], name="X")
    E1 = ae_layer(X, n_data, n_h1, name="encoder_1")
    E2 = ae_layer(E1, n_h1, n_h2, name="encoder_2")
    E3 = ae_layer(E2, n_h2, n_h3, name="encoder_3")
    D1 = ae_layer(E3, n_h3, n_h2, name="decoder_1")
    D2 = ae_layer(D1, n_h2, n_h1, name="decoder_2")
    output = ae_layer(D2, n_h1, n_data, name="decoder_3")

    with tf.name_scope("cost"):
        cost = tf.reduce_mean(tf.pow(X - output, 2))
    tf.summary.scalar("cost", cost)

    with tf.name_scope("train"):
        train = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = train.minimize(cost)

    init = tf.global_variables_initializer()
    summaries = tf.summary.merge_all()


with tf.Session(graph=mnist_graph) as sess:
    # from tensorflow.python import debug as tf_debug
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    writer = tf.summary.FileWriter(log_path, sess.graph)
    sess.run(init)
    for i in range(training_epochs):
        batch = mnist.train.next_batch(batch_size)
        _, s, c = sess.run([optimizer, summaries, cost],
                           feed_dict={X: batch[0]})
        writer.add_summary(s, training_epochs * i)
        if training_epochs % display_step == 0:
            print("Epoch {} Training Error = {}".format(i+1, c))
    # Save variables to disk
    saver = tf.train.Saver()
    save_path = saver.save(sess, "/tmp/trained_model.ckpt")
    print("Model saved to: '{}'".format(save_path))
