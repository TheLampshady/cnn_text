import tensorflow as tf

input_x = tf.constant(1, shape=[10, 10])
shape = [3, 3, 1, 2]
weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
biases = tf.Variable(tf.constant(0.1, shape=[2]))
stride_shape = [1, 1, 1, 1]

# Shape: [1, 8, 8, 2]
conv = tf.nn.conv2d(input_x, weights, strides=stride_shape, padding="VALID")

next_dim = 10 - 3 + 1
pool_shape = [1, next_dim, 1, 1]

# Shape: [1, 8, 8, 2]
activated = tf.nn.relu(tf.nn.bias_add(conv, biases))

# Shape: [1, 1, 8, 2]
pooled = tf.nn.max_pool(activated, ksize=pool_shape, strides=stride_shape, padding='VALID')
