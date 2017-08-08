from math import exp

import tensorflow as tf


class MultilayerPerceptron(object):

    def __init__(self, layers, dropout=None):
        """
        :type layers: list
        :type dropout: double
        """
        self.layers = layers
        self.dropout = dropout

        self.lrmax = 0.003
        self.lrmin = 0.00001
        self.decay_speed = 2000.0

        # Placeholders
        self.input_tensor = tf.placeholder(tf.float32, [None, layers[0]], name="input")
        self.output_tensor = tf.placeholder(tf.float32, [None, layers[-1]], name="output")

        self.learn_rate = tf.placeholder(tf.float32)
        self.pkeep = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.logits = None

        self.weights = [
            tf.Variable(tf.truncated_normal(
                [layers[i], layers[i+1]],
                stddev=0.1,
                name="Weight" + str(i)
            ))
            for i in range(len(layers)-1)
        ]

        self.biases = [
            tf.Variable(tf.ones([layers[i]])/10, "Bias" + str(i))
            for i in range(1, len(layers))
        ]

        self.prediction_model = self._build_prediction_model()
        self.train_step = self._get_training_model()
        self.accuracy_model = self._get_accuracy_model()

    def _build_prediction_model(self):
        """
        Layer processing with activation function
        :param tensor: input data
        :return: tensor prediction
        """
        i = 0
        layer = self.input_tensor
        for i in range(len(self.layers) - 2):
            name = "activate_" + str(i)
            product = tf.matmul(layer, self.weights[i], name=name) + self.biases[i]
            layer = tf.nn.relu(product)
            if self.dropout:
                layer = tf.nn.dropout(layer, self.pkeep)

        # Skip Relu on output node for logits
        i += 1
        return tf.matmul(layer, self.weights[i], name="logits") + self.biases[i]

    def _get_training_model(self):
        # Define Loss
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=self.prediction_model,
            labels=self.output_tensor
        )
        self.loss = tf.reduce_mean(cross_entropy) * 100

        # Define optimization
        optimizer = tf.train.AdamOptimizer(self.learn_rate)
        return optimizer.minimize(self.loss)

    def _get_accuracy_model(self):
        # Test model
        correct_prediction = tf.equal(
            tf.argmax(self.prediction_model, 1),
            tf.argmax(self.output_tensor, 1)
        )
        # Calculate accuracy
        return tf.reduce_mean(tf.cast(correct_prediction, "float"))

    def get_learn_rate(self, i):
        return self.lrmin + (self.lrmax - self.lrmin) * exp(-i / self.decay_speed)

    def train(self, text_classifier, epochs=10, display_step=1):
        # Initializing the variables
        init = tf.global_variables_initializer()

        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)

            # Training cycle
            for epoch in range(epochs):
                avg_cost = 0.
                for i in range(text_classifier.batch_count):
                    batch_x, batch_y = text_classifier.get_train_batch(i)
                    learning_rate = self.get_learn_rate(i)
                    feed_dict = {
                        self.input_tensor: batch_x,
                        self.output_tensor: batch_y,
                        self.learn_rate: learning_rate,
                    }
                    if self.dropout:
                        feed_dict[self.pkeep] = self.dropout

                    # Run optimization op (backprop) and cost op (to get loss value)
                    c, _ = sess.run(
                        [self.loss, self.train_step],
                        feed_dict=feed_dict
                    )

                    # Compute average loss
                    avg_cost += c / text_classifier.batch_count

                # Display logs per epoch step
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(avg_cost))

            print("Optimization Finished!")

            batch_x_test, batch_y_test = text_classifier.get_test_batch()
            feed_data = {self.input_tensor: batch_x_test, self.output_tensor: batch_y_test}
            if self.dropout:
                feed_data[self.pkeep] = 1.0
            print("Accuracy:", self.accuracy_model.eval(feed_data))
