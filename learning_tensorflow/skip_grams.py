#!/usr/bin/env python3
import os, math
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

batch_size = 64
embedding_demensions = 5
negative_samples = 8
LOG_DIR = "logs/word2vec_intro"

digit_to_word_map = {
    1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
    6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"
}

sentences = []

# Create two kinds of sentences - sequences of odd and even digits
for i in range(10000):
    rand_odd_ints = np.random.choice(range(1, 10, 2), 3)
    sentences.append(" ".join([digit_to_word_map[r] for r in rand_odd_ints]))
    rand_even_ints = np.random.choice(range(2, 10, 2), 3)
    sentences.append(" ".join([digit_to_word_map[r] for r in rand_even_ints]))

word2index_map = {}
index = 0
for sent in sentences:
    for word in sent.lower().split():
        if word not in word2index_map:
            word2index_map[word] = index
            index += 1

index2word_map = {index: word for word, index in word2index_map.items()}
vocabulary_size = len(index2word_map)


# Generate skip-gram pairs
skip_gram_pairs = []
for sent in sentences:
    tokenized_sent = sent.lower().split()
    for i in range(1, len(tokenized_sent)-1):
        word_id = word2index_map[tokenized_sent[i]]
        next_id = word2index_map[tokenized_sent[i+1]]
        prev_id = word2index_map[tokenized_sent[i-1]]

        skip_gram_pairs.append([word_id, prev_id])
        skip_gram_pairs.append([word_id, next_id])


def get_skipgram_batch(batch_size):
    instance_indices = list(range(len(skip_gram_pairs)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    x = [skip_gram_pairs[i][0] for i in batch]
    y = [[skip_gram_pairs[i][1]] for i in batch]
    return x, y

train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

with tf.name_scope("embeddings"):
    embeddings = tf.get_variable(
        "embedding",
        shape=[vocabulary_size, embedding_demensions],
        initializer=tf.random_uniform_initializer(-1.0, 1.0)
    )
    # This is essentially a lookup table
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm

# Todo Distributing equally over dimensions
nce_stddev = 1.0 / math.sqrt(embedding_demensions)
nce_weights = tf.get_variable(
    "nce_weights",
    shape=[vocabulary_size, embedding_demensions],
    initializer=tf.truncated_normal_initializer(stddev=nce_stddev)
)
nce_biases = tf.get_variable(
    "nce_biases",
    shape=[vocabulary_size],
    initializer=tf.zeros_initializer()
)

loss = tf.reduce_mean(
    tf.nn.nce_loss(
        weights=nce_weights, biases=nce_biases,
        inputs=embed, labels=train_labels,
        num_sampled=negative_samples, num_classes=vocabulary_size
    )
)
tf.summary.scalar('Losses', loss)


global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(
    learning_rate=0.1,
    global_step=global_step,
    decay_steps=1000,
    decay_rate=0.95,
    staircase=True
)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter(LOG_DIR, graph=tf.get_default_graph())
saver = tf.train.Saver()

model_file = os.path.join(LOG_DIR, "w2v_model.ckpt")
full_log_path = os.path.join(os.getcwd(), LOG_DIR)
meta_file = os.path.join(full_log_path, 'metadata.tsv')

with open(meta_file, "w") as metadata:
    metadata.write("name\tClass\n")
    for k,v in index2word_map.items():
        metadata.write("%s\t%d\n" % (v, k))

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = embeddings.name

# Link embedding to its metadata file
embedding.metadata_path = meta_file
projector.visualize_embeddings(train_writer, config)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for step in range(1000):
        x_batch, y_batch = get_skipgram_batch(batch_size)
        feed_dict = {train_inputs: x_batch, train_labels: y_batch}
        summary, _ = sess.run([merged, train_step], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)

        if step % 100 == 0:
            saver.save(sess, model_file, step)
            loss_value = sess.run(loss, feed_dict=feed_dict)

            print("Loss at %d: %.5f" % (step, loss_value))

    normalized_embeddings_matrix = sess.run(normalized_embeddings)