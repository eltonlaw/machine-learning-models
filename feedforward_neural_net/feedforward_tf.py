""" Feedforward neural network with Python3 & TensorFlow
Weights initialized using a truncated normal distribution - N(0, 1)
Biases initialized at 0.1
Cost function: Cross Entropy
Layers: [Input - 784,
         Dense - 500 - relu
         Dense - 300 - relu,
         Output - 10 - softmax - one hot encoding]
Learning Rate: Start at 0.02, 0.95 decay rate, 20000 decay steps
Optimizer: RMSProp
Batch Size = 256
Training Epochs = 100
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python import debug as tf_debug

# Session Parameters

config = tf.ConfigProto(allow_soft_placement=True,
                        log_device_placement=True)

# Model Parameters ###############
n_h1 = 500
n_h2 = 300

batch_size = 256
epochs = 100

starting_lr = 0.02
decay_rate = 0.95
decay_steps = 20000

display_step = 1
log_path = "./summaries"
train_dir = "../MNIST_data"

# Get Data
mnist = input_data.read_data_sets(train_dir, one_hot=True)

n_in = mnist.train.images.shape[1]
n_out = mnist.train.labels.shape[1]
g = tf.Graph()


def variable_summaries(var):
    with tf.name_scope("summary"):
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean", mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar("stddev", stddev)
        tf.summary.histogram("histogram", var)


with g.as_default():
    with tf.name_scope("input") as scope:
        x = tf.placeholder(tf.float32, [None, n_in], name="label")
        y = tf.placeholder(tf.float32, [None, n_out], name="data")

    with tf.name_scope("hidden1") as scope:
        W1 = tf.Variable(tf.truncated_normal([n_in, n_h1]), name="weights")
        B1 = tf.Variable(tf.ones(n_h1)/10, name="biases")
        H1 = tf.nn.relu(tf.matmul(x, W1) + B1, name="H1")
        variable_summaries(W1)
        variable_summaries(B1)
        variable_summaries(H1)

    with tf.name_scope("hidden2") as scope:
        W2 = tf.Variable(tf.truncated_normal([n_h1, n_h2]), name="weights")
        B2 = tf.Variable(tf.ones(n_h2)/10, name="biases")
        H2 = tf.nn.relu(tf.matmul(H1, W2) + B2, name="H2")
        variable_summaries(W2)
        variable_summaries(B2)
        variable_summaries(H2)

    with tf.name_scope("output") as scope:
        W3 = tf.Variable(tf.truncated_normal([n_h2, n_out]), name="weights")
        B3 = tf.Variable(tf.ones(n_out)/10, name="biases")
        y_ = tf.matmul(H2, W3) + B3
        variable_summaries(W3)
        variable_summaries(B3)
        variable_summaries(y_)

    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                                                logits=y_),
                        name="cross_entropy_mean")
    tf.summary.scalar("cross_entropy", cross_entropy)

    # Global step refers to the number of batches seen by the graph
    global_step = tf.Variable(0, name="global_step", trainable=False)
    decayed_lr = tf.train.exponential_decay(starting_lr, global_step,
                                            decay_steps, decay_rate)

    with tf.name_scope("train"):
        train = tf.train.RMSPropOptimizer(decayed_lr)
        optimizer = train.minimize(cross_entropy, global_step=global_step)

    with tf.name_scope("accuracy"):
        with tf.name_scope("correct_prediction"):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    summary = tf.summary.merge_all()


with tf.Session(graph=g, config=config) as sess:
    # Wrap TensorFlow session with tfdbg
    # To use, uncomment below, then add --debug flag when running this script
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    # Create summary writer
    # On CLI: tensorboard --logdir=log_path
    writer = tf.summary.FileWriter(log_path, sess.graph)

    # Initialize variables
    sess.run(init)
    # Train Step
    for i in range(epochs):
        batch = mnist.train.next_batch(batch_size)
        _, s, a = sess.run([optimizer, summary, accuracy],
                           feed_dict={x: batch[0], y: batch[1]})
        writer.add_summary(s, epochs * i)
        if epochs % display_step == 0:
            print("Epoch {} Training Error = {}".format(i+1, a))
    print("Final Training Error = {}".format(a))
    # Test Step
    print(accuracy.eval(feed_dict={x: mnist.test.images,
                                   y: mnist.test.labels}))
    # Save variables to disk
    save_path = saver.save(sess, "/tmp/trained_model.ckpt")
    print("Model saved to: '{}'".format(save_path))
