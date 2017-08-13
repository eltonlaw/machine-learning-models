from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Parameters
train_dir = "../MNIST_data"
log_path = "./logs"
n_h = 1024
batch_size = 256
display_step = 1
learning_rate = 0.01
training_epochs = 100
regularization_constant = 0.01

mnist = input_data.read_data_sets(train_dir, one_hot=True)
n_data = mnist.train.images.shape[1]


def ae_layer(X_in, channel_in, channel_out, name="autoencoder_layer"):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal([channel_in, channel_out]),
                        name="weights")
        tf.losses.add_loss(regularization_constant*tf.nn.l2_loss(W))
        tf.summary.histogram("weights", W)
        B = tf.Variable(tf.truncated_normal([channel_out]), name="biases")
        tf.summary.histogram("biases", B)
        return tf.nn.sigmoid(tf.matmul(X_in, W) + B)


mnist_graph = tf.Graph()
with mnist_graph.as_default():
    X = tf.placeholder("float", [None, n_data], name="X")
    E1 = ae_layer(X, n_data, n_h, name="encoder")
    D1 = ae_layer(E1, n_h, n_data, name="decoder")
    X_ = D1

    with tf.name_scope("cost"):
        mse = tf.reduce_mean(tf.pow(X - X_, 2))
        tf.losses.add_loss(mse)
        cost = tf.losses.get_total_loss()
    tf.summary.scalar("cost", cost)

    with tf.name_scope("train"):
        train = tf.train.GradientDescentOptimizer(learning_rate)
        optimizer = train.minimize(cost)
    init = tf.global_variables_initializer()
    summaries = tf.summary.merge_all()


with tf.Session(graph=mnist_graph) as sess:
    sess.run(init)
    for i in range(training_epochs):
        batch = mnist.train.next_batch(batch_size)
        _, s, c = sess.run([optimizer, summaries, cost],
                           feed_dict={X: batch[0]})
        if training_epochs % display_step == 0:
            print("Epoch {} Training Error = {}".format(i+1, c))
    # Predict
    test_cost = sess.run(cost, feed_dict={X: mnist.test.images})
    print("Test Cost: {}".format(test_cost))
