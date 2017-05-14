"""
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

mnist = input_data.read_data_sets("../MNIST_data", one_hot=True)

n_in = mnist.train.images.shape[1]
n_out = mnist.train.labels.shape[1]
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
#############################
g = tf.Graph()

with g.as_default():
    with tf.name_scope("input") as scope:
        x = tf.placeholder(tf.float32, [None, n_in], name="label")
        y = tf.placeholder(tf.float32, [None, n_out], name="data")

    with tf.name_scope("hidden1") as scope:
        W1 = tf.Variable(tf.truncated_normal([n_in, n_h1]), name="weights")
        B1 = tf.Variable(tf.ones(n_h1)/10, name="biases")

    with tf.name_scope("hidden2") as scope:
        W2 = tf.Variable(tf.truncated_normal([n_h1, n_h2]), name="weights")
        B2 = tf.Variable(tf.ones(n_h2)/10, name="biases")

    with tf.name_scope("output") as scope:
        W3 = tf.Variable(tf.truncated_normal([n_h2, n_out]), name="weights")
        B3 = tf.Variable(tf.ones(n_out)/10, name="biases")

    H1 = tf.nn.relu(tf.matmul(x, W1) + B1, name="H1")
    H2 = tf.nn.relu(tf.matmul(H1, W2) + B2, name="H2")
    y_ = tf.matmul(H2, W3) + B3

    cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_),
            name="cost")
    tf.summary.scalar("cost", cost)

    global_step = tf.Variable(0, trainable=False)
    decayed_lr = tf.train.exponential_decay(starting_lr, global_step,
                                            decay_steps, decay_rate)
    train = tf.train.RMSPropOptimizer(decayed_lr)
    optimizer = train.minimize(cost, global_step=global_step)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    summary = tf.summary.merge_all()

with tf.Session(graph=g) as sess:
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
        _, c, s = sess.run([optimizer, cost, summary],
                           feed_dict={x: batch[0], y: batch[1]})
        writer.add_summary(s, epochs * i)
        if epochs % display_step == 0:
            print("Epoch {} Training Error = {}".format(i+1, c))
    # Test Step
    print(accuracy.eval(feed_dict={x: mnist.test.images,
                                   y: mnist.test.labels}))
    # Save variables to disk
    save_path = saver.save(sess, "/tmp/trained_model.ckpt")
    print("Model saved to: '{}'".format(save_path))
