# http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, sequence_length, num_classes, vocab_size,
                    embedding_size, filter_sizes, num_filters):
        """
        :type sequence_length: int
        :type num_classes: int
        :type vocab_size: int
        :type embedding_size: int
        :type filter_sizes: list <int>
        :param num_filters: int
        """
        pass