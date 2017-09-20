import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def variable_summaries(var):
    """
    This helper function, taken from the official TensorFlow documentation,
    simply adds some ops that take care of logging summaries
    :type var: tf.Variable
    :return:
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def run():
    # Import MNIST data
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

    # Define some parameters
    element_size = 28
    time_steps = 28
    num_classes = 10
    batch_size = 128
    hidden_layer_size = 128

    # Where to save TensorBoard model summaries
    LOG_DIR = "tensor_log/RNN_with_summaries"

    # Create placeholders for inputs, labels
    _inputs = tf.placeholder(tf.float32, shape=[None, time_steps, element_size], name='inputs')
    y = tf.placeholder(tf.float32, shape=[None, num_classes], name='labels')

    # TensorFlow built-in functions
    rnn_cell = tf.contrib.rnn.BasicRNNCell(hidden_layer_size)
    outputs, state = tf.nn.dynamic_rnn(rnn_cell, _inputs, dtype=tf.float32)

    # Weights and bias for input and hidden layer
    # with tf.name_scope('rnn_weights'):
    #     with tf.name_scope("W_x"):
    #         Wx = tf.Variable(tf.zeros([element_size, hidden_layer_size]))
    #         variable_summaries(Wx)
    #
    #     with tf.name_scope("W_h"):
    #         Wh = tf.Variable(tf.zeros([hidden_layer_size, hidden_layer_size]))
    #         variable_summaries(Wh)
    #
    #     with tf.name_scope("Bias"):
    #         b_rnn = tf.Variable(tf.zeros([hidden_layer_size]))
    #         variable_summaries(b_rnn)

    # def rnn_step(previous_hidden_state, x):
    #     return tf.tanh(tf.matmul(previous_hidden_state, Wh) + tf.matmul(x, Wx) + b_rnn)


    # Processing inputs to work with scan function
    # Current input shape: (batch_size, time_steps, element_size)
    # processed_input = tf.transpose(_inputs, perm=[1, 0, 2])
    #
    # # Current input shape now: (time_steps, batch_size, element_size)
    # initial_hidden = tf.zeros([batch_size, hidden_layer_size])
    #
    # # Getting all state vectors across time
    # all_hidden_states = tf.scan(
    #     lambda prev, x: tf.tanh(tf.matmul(prev, Wh) + tf.matmul(x, Wx) + b_rnn),
    #     processed_input,
    #     initializer=initial_hidden,
    #     name='states'
    # )

    # Weights for output layers
    with tf.name_scope('linear_layer_weights') as scope:
        with tf.name_scope("W_Linear"):
            weights = tf.Variable(tf.truncated_normal(
                [hidden_layer_size, num_classes],
                mean=0, stddev=0.01))
            variable_summaries(weights)
        with tf.name_scope("Bias_Linear"):
            biases = tf.Variable(tf.truncated_normal([num_classes], mean=0, stddev=.01))
            variable_summaries(biases)

    # Apply linear layer to state vector    
    # def get_linear_layer(hidden_state):
    #     return tf.matmul(hidden_state, Wl) + bl

    with tf.name_scope('linear_layer_weights') as scope:
        # Iterate across time, apply linear layer to all RNN outputs
        # all_outputs = tf.map_fn(
        #     lambda hidden_state: tf.matmul(hidden_state, weights) + biases,
        #     all_hidden_states
        # )
        # Get last output    
        # output = all_outputs[-1]
        last_rnn_output = outputs[:, -1, :]
        logits = tf.matmul(last_rnn_output, weights) + biases
        tf.summary.histogram('RNN_Output', logits)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        # Using RMSPropOptimizer
        train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

    with tf.name_scope('accuracy'):
        #Todo logits or final output
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1))
        accuracy = (tf.reduce_mean(tf.cast(correct_prediction, tf.float32))) * 100
        tf.summary.scalar('accuracy', accuracy)

    # Merge all the summaries
    merged = tf.summary.merge_all()

    # Get a small test set
    test_data = mnist.test.images[: batch_size].reshape((-1, time_steps, element_size))
    test_label = mnist.test.labels[: batch_size]
    with tf.Session() as sess:
        # Write summaries to LOG_DIR -- used by TensorBoard
        train_writer = tf.summary.FileWriter(LOG_DIR + '/ train', graph=tf.get_default_graph())
        test_writer = tf.summary.FileWriter(LOG_DIR + '/ test', graph=tf.get_default_graph())
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Reshape data to get 28 sequences of 28 pixels
            batch_x = batch_x.reshape(
                (batch_size, time_steps, element_size)
            )
            summary, _ = sess.run([merged, train_step], feed_dict={_inputs: batch_x, y: batch_y})
            # Add to summaries
            train_writer.add_summary(summary, i)

            if i % 1000 == 0:
                acc, loss, = sess.run([accuracy, cross_entropy], feed_dict={_inputs: batch_x, y: batch_y})
                train_loss = "Minibatch Loss= {:.6f}, ".format(loss)
                train_acc = "Training Accuracy= {:.5f}".format(acc)
                print(("Iter %d, " % i) + train_loss + train_acc )

            if i % 10:
                # Calculate accuracy for 128 MNIST test images and add to summaries
                summary, acc = sess.run([merged, accuracy], feed_dict={_inputs: test_data, y: test_label})
                test_writer.add_summary(summary, i)

        test_acc = sess.run(accuracy, feed_dict={_inputs: test_data, y: test_label})
        print(" Test Accuracy:", test_acc)
