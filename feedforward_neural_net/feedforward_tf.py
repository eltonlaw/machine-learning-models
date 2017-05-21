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
# from tensorflow.python import debug as tf_debug

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
log_path = "./logs"
train_dir = "../MNIST_data"
# Get Data
mnist = input_data.read_data_sets(train_dir, one_hot=True)
n_data = mnist.train.images.shape[1]
n_labels = mnist.train.labels.shape[1]


def variable_summaries(var):
    with tf.name_scope("summary"):
        tf.summary.histogram("histogram", var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar("mean", mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar("stddev", stddev)


def fc_layer(X, channels_in, channels_out, name="fully_connected"):
    with tf.name_scope(name):
        with tf.name_scope("weights"):
            W = tf.Variable(tf.truncated_normal([channels_in, channels_out]),
                            name="weights")
            variable_summaries(W)
        with tf.name_scope("biases"):
            B = tf.Variable(tf.ones(channels_out)/10, name="biases")
            variable_summaries(B)
        with tf.name_scope("activation"):
            A = tf.nn.relu(tf.matmul(X, W) + B, name="activation")
            variable_summaries(A)
        return A


def logits_layer(X, channels_in, channels_out, name="logits"):
    with tf.name_scope(name):
        with tf.name_scope("weights"):
            W = tf.Variable(tf.truncated_normal([n_h2, n_labels]),
                            name="weights")
            variable_summaries(W)
        with tf.name_scope("biases"):
            B = tf.Variable(tf.ones(n_labels)/10, name="biases")
            variable_summaries(B)
        with tf.name_scope("activation"):
            Z = tf.matmul(A2, W) + B
            variable_summaries(Z)
        return Z


mnist_graph = tf.Graph()
with mnist_graph.as_default():
    x = tf.placeholder(tf.float32, [None, n_data], name="x")
    y = tf.placeholder(tf.float32, [None, n_labels], name="labels")
    # img = tf.reshape(x, [-1, 28, 28, 1])
    # tf.summary.image("", img)
    A1 = fc_layer(x, n_data, n_h1, name="fully_connected_1")
    A2 = fc_layer(A1, n_h1, n_h2, name="fully_connected_2")
    y_ = logits_layer(A2, n_h2, n_labels, name="logit")

    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                                                logits=y_),
                        name="cross_entropy_mean")
    tf.summary.scalar("cross_entropy", cross_entropy)

    with tf.name_scope("decay_learning_rate"):
        # Global step refers to the number of batches seen by the graph
        global_step = tf.Variable(0, name="global_step", trainable=False)
        decayed_lr = tf.train.exponential_decay(starting_lr, global_step,
                                                decay_steps, decay_rate)
    tf.summary.scalar("decay_learning_rate", decayed_lr)

    with tf.name_scope("train"):
        train = tf.train.RMSPropOptimizer(decayed_lr)
        optimizer = train.minimize(cross_entropy, global_step=global_step)

    with tf.name_scope("accuracy"):
        with tf.name_scope("correct_prediction"):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

    merged_summary = tf.summary.merge_all()


with tf.Session(graph=mnist_graph, config=config) as sess:
    # Wraps TensorFlow session with tfdbg
    # To use, uncomment below, then add --debug flag running script
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    # Create summary writer
    # On CLI: tensorboard --logdir=log_path
    writer = tf.summary.FileWriter(log_path, sess.graph)
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    # Train Step
    for i in range(epochs):
        batch = mnist.train.next_batch(batch_size)
        _, s, a = sess.run([optimizer, merged_summary, accuracy],
                           feed_dict={x: batch[0], y: batch[1]})
        writer.add_summary(s, epochs * i)
        if epochs % display_step == 0:
            print("Epoch {} Training Error = {}".format(i+1, a))
    print("Final Training Error = {}".format(a))
    # Test Step
    print(accuracy.eval(feed_dict={x: mnist.test.images,
                                   y: mnist.test.labels}))
    # Save variables to disk
    saver = tf.train.Saver()
    save_path = saver.save(sess, log_path+"/model.ckpt")
    print("Model saved to: '{}'".format(save_path))
