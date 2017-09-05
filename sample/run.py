#!/usr/bin/env python3
import functools
from math import exp
from os.path import basename, splitext

import tensorflow as tf
import data_helpers

from tensor_functions import bias_variable, weight_variable
import numpy as np


def build_batch(data, batch_size=64, num_epochs=200, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def run():
    """
    Convolutional NN Text
    Activation function: relu
    Optimizer: AdamOptimizer
    :return:
    """

    # ----- Data -------
    percent_test = 0.1
    print("Loading data...")
    positive_data_file = "data/rt-polaritydata/rt-polarity.pos"
    negative_data_file = "data/rt-polaritydata/rt-polarity.neg"

    data, target = data_helpers.load_data_and_labels(positive_data_file, negative_data_file)
    max_document_length = max([len(x.split(" ")) for x in data])
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(data)))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(target)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = target[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use `1cross-validation
    dev_sample_index = -1 * int(percent_test * float(len(target)))
    train_data, test_data = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    train_target, test_target = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    train_batch = build_batch(list(zip(train_data, train_target)))

    # ------ Constants -------


    # Data
    test_freq = 20

    # Learning Rate Values
    lrmax = 0.003
    lrmin = 0.00001
    decay_speed = 2000.0

    # Drop-off (Less for text)
    keep_ratio = 0.5

    # Text Values
    sequence_length = train_data.shape[1]
    output_size = train_target.shape[1]

    # Todo Dimensionality of character embedding
    embedding_dim = 128

    vocab_size = len(vocab_processor.vocabulary_)

    print("Vocabulary Size: {:d}".format(vocab_size))
    print("Train/Dev split: {:d}/{:d}".format(len(train_target), len(test_target)))

    # Layers (Single Dimension for text)
    filters = [
        5,
        4,
        3,
    ]

    # channels = [1, 4, 8, 12]
    # channels = [1, 6, 12, 24]
    num_filters = 128

    # Always `1` for text
    # strides = [1, 2, 2]
    stride = 1
    stride_shape = [1, stride, stride, 1]

    # Tensor Board Log
    logs_path = "tensor_log/%s/" % splitext(basename(__file__))[0]

    fully_connecting_nodes = num_filters * len(filters)

    # Target classifications for nodes
    output_nodes = 2

    # Place holders
    X = tf.placeholder(tf.int32, [None, sequence_length], name="Input_PH")
    Y_ = tf.placeholder(tf.float32, [None, output_size], name="Output_PH")
    L = tf.placeholder(tf.float32, name="Learning_Rate_PH")
    keep_prob = tf.placeholder(tf.float32, name="Per_Keep_PH")

    # Initialize Activation
    with tf.device('/cpu:0'), tf.name_scope("embedding"):
        embedding = tf.Variable(
            tf.random_uniform([vocab_size, embedding_dim], -1.0, 1.0),
            name="Text_Embedding"
        )
        embedded_chars = tf.nn.embedding_lookup(embedding, X)
        embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

    # ----- Weights and Bias -----
    weights = []
    biases = []
    for i in range(len(filters)):
        with tf.name_scope('Layer'):
            # weight_shape = [filters[i], embedding_dim] + channels[i:i+2]
            weight_shape = [filters[i], embedding_dim, 1, num_filters]
            weights.append(weight_variable(weight_shape))
            biases.append(bias_variable(weight_shape[-1:]))

    with tf.name_scope('Layer'):
        WOutput = weight_variable([fully_connecting_nodes, output_nodes])
        BOutput = bias_variable([output_nodes])

    # ---------------- Operations ----------------

    # ------- Activation Function -------
    """
    This method creates 3 separate layers with different filter sizes that get concatenated.
    Other networks have taking a layer results and fed them into the next layer.
    """
    pooled_outputs = []
    for i in range(len(filters)):
        with tf.name_scope('Wx_plus_b'):
            # Todo Same input for each layer?
            preactivate = tf.nn.conv2d(
                embedded_chars_expanded,
                weights[i],
                strides=stride_shape,
                padding="VALID",
                name="conv"
            )
            tf.summary.histogram('Pre_Activations', preactivate)

            # Apply nonlinearity
            activations = tf.nn.relu(tf.nn.bias_add(preactivate, biases[i]), name="relu")
            tf.summary.histogram('Activations', activations)

            # Todo same stride shape for conv2d and max_pool (stride_shape and stride_pool_shape)
            # Valid Padding dimension size: (input_size - filter_size + 1) / stride
            next_dim = sequence_length - filters[i] + 1

            # Ksize reduces the conv dimensions by conv2d[0] - pool_shape[0] +1
            # with strides: (conv2d[0] - pool_shape[0] +1) / stride[0]
            # Example:  conv2d = [1, 8, 8, 2]
            #           pool_shape = [1, 8, 1, 1]
            #           stride_pool_shape = [1, 1, 4, 1]
            # Result:   [1, 1, 2, 2]
            pool_shape = [1, next_dim, 1, 1]
            stride_pool_shape = [1, 1, 1, 1]

            pooled = tf.nn.max_pool(
                activations,
                ksize=pool_shape,
                strides=stride_pool_shape,
                padding='VALID',
                name="pool"
            )
            # Todo Output is not cycled through next layer
            pooled_outputs.append(pooled)

    # Combine all the pooled features
    pool_results = tf.concat(pooled_outputs, 3)
    pool_flat = tf.reshape(pool_results, [-1, fully_connecting_nodes])

    fully_connected_dropout = tf.nn.dropout(pool_flat, keep_prob)

    # ------- Regression Functions -------
    with tf.name_scope('Wx_plus_b'):
        logits = tf.nn.xw_plus_b(fully_connected_dropout, WOutput, BOutput, name="Product")
        tf.summary.histogram('Pre_Activations', logits)
    predictions = tf.nn.softmax(logits, name="Output_Result")

    # ------- Loss Function -------
    with tf.name_scope('Loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=Y_, name="Cross_Entropy")
        with tf.name_scope('Total'):
            loss = tf.reduce_mean(cross_entropy, name="loss") * 100
    tf.summary.scalar('Losses', loss)

    # ------- Optimizer -------
    with tf.name_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer(L)
        train_step = optimizer.minimize(loss, name="minimize")

    # ------- Accuracy -------
    with tf.name_scope('Accuracy'):
        with tf.name_scope('correct_prediction'):
            is_correct = tf.equal(
                tf.argmax(predictions, 1, name="Max_Result"),
                tf.argmax(Y_, 1, name="Target")
            )
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    tf.summary.scalar('Accuracies', accuracy)

    # ------- Tensor Graph -------
    # Start Tensor Graph
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    # Tensor Board
    merged_summary_op = tf.summary.merge_all()

    tensor_graph = tf.get_default_graph()
    train_writer = tf.summary.FileWriter(logs_path + "train", graph=tensor_graph)
    test_writer = tf.summary.FileWriter(logs_path + "test")

    # ------- Training -------
    train_operations = [train_step, loss, merged_summary_op]
    test_operations = [accuracy, loss, merged_summary_op]
    test_data = {X: test_data, Y_: test_target, keep_prob: 1.0, L: 0}

    avg_cost = 0.
    for step, batch in enumerate(train_batch):

        # ----- Train step -----
        batch_X, batch_Y = zip(*batch)

        learning_rate = lrmin + (lrmax - lrmin) * exp(-step / decay_speed)
        train_data = {
            X: batch_X,
            Y_: batch_Y,
            L: learning_rate,
            keep_prob: keep_ratio
        }

        # Record execution stats
        if step % 100 == 99:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            _, cross_loss, summary = sess.run(
                train_operations,
                feed_dict=train_data,
                options=run_options,
                run_metadata=run_metadata
            )

        else:
            _, cross_loss, summary = sess.run(
                train_operations,
                feed_dict=train_data
            )

        # ----- Test Step -----
        if step % test_freq == 0:
            acc, cross_loss, summary = sess.run(
                test_operations,
                feed_dict=test_data
            )
            test_writer.add_summary(summary, step)
            print('Accuracy at step %s: %s' % (step, acc))
    #
    #     avg_cost += cross_loss / batch_total
    #     train_writer.add_summary(summary, step)
    #
    # # Display logs per epoch step
    # print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))


if __name__ == "__main__":
    run()
