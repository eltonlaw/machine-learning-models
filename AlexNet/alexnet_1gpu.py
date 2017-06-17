import tensorflow as tf

# HYPERPARAMETERS
batch_size = 128
initial_lr = 0.01
epochs = 90
dropout_p = 0.5
momentum = 0.9
weight_decay = 0.0005
decay_rate = 0.95
decay_steps = 1000


def conv_layer(X, K_n, strides, B_init=1):
    """
    PARAMETERS
    ----------
    X: Tensor
        Input
    K_n: shape (1,4)
        Size of kernel
    strides: shape(1,4)
        Number of Strides to take
    RETURNS
    -------
    Activations: Tensor
    """
    W = tf.get_variable("weights", K_n,
                        initializer=tf.random_normal_initializer(mean=0.0,
                                                                 stddev=0.01))
    with tf.variable_scope('regularization'):
        norm = tf.reduce_sum(tf.nn.l2_loss(W))
        tf.add_to_collection("losses", weight_decay*norm)
    B = tf.get_variable("biases", [1, K_n[-1]],
                        initializer=tf.constant_initializer(B_init))
    conv = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + B)


def maxpool_layer(X):
    ksize = [1, 3, 3, 1]
    strides = [1, 2, 2, 1]
    return tf.nn.max_pool(X, ksize, strides, "SAME")


def fc_layer(X, in_n, out_n):
    W = tf.get_variable("weights", [in_n, out_n],
                        initializer=tf.random_normal_initializer(mean=0.0,
                                                                 stddev=0.01),
                        collections=['variables'])
    with tf.variable_scope('regularization'):
        norm = tf.reduce_sum(tf.nn.l2_loss(W))
        tf.add_to_collection("losses", weight_decay*norm)
    B = tf.get_variable("biases", [1, out_n],
                        initializer=tf.constant_initializer(1))
    return tf.nn.relu(tf.xw_plux_b(X, W, B))


def logits_layer(X, in_n, out_n):
    W = tf.get_variable("weights", [in_n, out_n],
                        initializer=tf.random_normal_initializer())
    with tf.variable_scope('regularization'):
        norm = tf.reduce_sum(tf.nn.l2_loss(W))
        tf.add_to_collection("losses", weight_decay*norm)
    B = tf.get_variable("biases", [1, out_n],
                        initializer=tf.constant_initializer(1))
    return tf.xw_plux_b(X, W, B)


def total_loss(y, y_):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y,
                                                            logits=y_)
    cross_entropy_loss = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection("losses", cross_entropy_loss)
    return tf.add_n(tf.get_collection("losses"), name="total_loss")


g = tf.Graph()
with g.as_default():
    X = tf.placeholder(tf.float32, [None, 224, 224, 3])
    y = tf.placeholder(tf.float32, [None, 1000])

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.inverse_time_decay(initial_lr, global_step,
                                                decay_steps, decay_rate,
                                                staircase=True)

    with tf.variable_scope("conv1"):
        conv1 = conv_layer(X, [11, 11, 3, 96], [1, 4, 4, 1], B_init=0)
    lrn1 = tf.nn.lrn(conv1, depth_radius=5, bias=2, alpha=0.0001, beta=0.75)
    with tf.variable_scope("conv2"):
        conv2 = conv_layer(lrn1, [5, 5, 96, 256], [1, 1, 1, 1])
    pool2 = maxpool_layer(conv2)
    with tf.variable_scope("conv3"):
        conv3 = conv_layer(pool2, [3, 3, 256, 384], [1, 1, 1, 1], B_init=0)
    pool3 = maxpool_layer(conv3)
    with tf.variable_scope("conv4"):
        conv4 = conv_layer(pool3, [3, 3, 384, 384], [1, 1, 1, 1])
    with tf.variable_scope("conv5"):
        conv5 = conv_layer(conv4, [3, 3, 384, 256], [1, 1, 1, 1])
    conv5_1d = tf.reshape(conv5, [batch_size, -1])
    conv5_1d_shape = conv5_1d.get_shape()[1].value
    with tf.variable_scope("fc1"):
        fc1 = fc_layer(conv5_1d, conv5_1d_shape, 4096)
        fc1 = tf.nn.dropout(fc1, dropout_p)
    with tf.variable_scope("fc2"):
        fc2 = fc_layer(fc1, 4096, 4096)
        fc2 = tf.nn.dropout(fc2, dropout_p)
    with tf.variable_scope("logits"):
        y_ = logits_layer(fc2, 4096, 1000)
    loss = total_loss(y, y_)
    train = tf.train.MomentumOptimizer(learning_rate, momentum)
    optimizer = train.minimize(loss, global_step=global_step)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    # ...Not added: ImageNet Dataset hooks
    for i in range(epochs):
        sess.run(optimizer)
    print(accuracy.eval(feed_dict={X: TEST_DATA,
                                   Y: TEST_LABELS}))
