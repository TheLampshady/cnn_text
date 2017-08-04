from collections import Counter
import numpy as np


class TextClassifier(object):

    def __init__(self, train_data, train_target, test_data, test_target, batch_size=10):
        self.train_data = train_data
        self.train_target = train_target
        self.test_data = test_data
        self.test_target = test_target
        self.batch_size = batch_size
        self.vocab = Counter()

        self.total_words = self.build_vocab()
        self.batch_count = int(len(train_data) / batch_size)

    def build_vocab(self):
        for text in self.train_data:
            for word in self.format_text(text):
                self.vocab[word] += 1

        for text in self.test_data:
            for word in self.format_text(text):
                self.vocab[word] += 1

        return len(self.vocab)

    @staticmethod
    def format_text(text):
        """
        Splits Text
        :param text:
        :return:
        """
        return [word.lower() for word in text.split(' ')]

    @property
    def word_index(self):
        if not hasattr(self, "_word_index"):
            self._word_index = {
                word.lower(): i
                for i, word in enumerate(self.vocab)
            }
        return self._word_index

    def get_batch(self, data, target):
        batches = []
        results = []

        for text in data:
            layer = np.zeros(self.total_words, dtype=float)
            for word in self.format_text(text):
                layer[self.word_index[word.lower()]] += 1

            batches.append(layer)

        for category in target:
            y = np.zeros((3), dtype=float)
            if category == 0:
                y[0] = 1.
            elif category == 1:
                y[1] = 1.
            else:
                y[2] = 1.
            results.append(y)

        return np.array(batches), np.array(results)

    def get_train_batch(self, i=0):
        start = i * self.batch_size
        end = start + self.batch_size

        return self.get_batch(
            self.train_data[start:end],
            self.train_target[start:end]
        )

    def get_test_batch(self):
        return self.get_batch(self.test_data, self.test_target)
