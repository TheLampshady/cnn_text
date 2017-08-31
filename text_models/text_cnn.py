from tensorflow.contrib import learn
import numpy as np


class TextClassifier(object):

    def __init__(self, data, target, sample_percent=.1):

        # Build vocabulary
        max_document_length = max([len(x.split(" ")) for x in data])
        self.vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        x = np.array(list(self.vocab_processor.fit_transform(data)))

        # Randomly shuffle data
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(target)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = target[shuffle_indices]

        # Split train/test set
        # TODO: This is very crude, should use `1cross-validation
        dev_sample_index = -1 * int(sample_percent * float(len(target)))
        self.train_data, self.test_data = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        self.train_target, self.test_target = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

        print("Vocabulary Size: {:d}".format(self.vocab_size))
        print("Train/Dev split: {:d}/{:d}".format(len(self.train_target), len(self.test_target)))

    @property
    def vocab_size(self):
        return len(self.vocab_processor.vocabulary_)

    def save(self, filename):
        self.vocab_processor.save(filename)

    @property
    def sequence_length(self):
        return self.train_data.shape[1]

    @property
    def output_size(self):
        return self.train_target.shape[1]

    def train_batch(self, batch_size=64, num_epochs=200, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = list(zip(self.train_data, self.train_target))
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