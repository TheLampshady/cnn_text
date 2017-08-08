from sklearn.datasets import fetch_20newsgroups

from tensor_models.ml_algorithm import MultilayerPerceptron
from text_models.text_manipulator import TextClassifier


def run():
    epochs = 10
    batch_size = 150
    dropout = 0.95

    categories = ["comp.graphics", "sci.space", "rec.sport.baseball"]
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

    text_classifier = TextClassifier(
        newsgroups_train.data,
        newsgroups_train.target,
        newsgroups_test.data,
        newsgroups_test.target,
        batch_size
    )

    n_input = text_classifier.total_words  # Words in vocab
    n_output = len(categories)

    # Network Parameters
    layers = [
        n_input,
        200,
        100,
        n_output
    ]

    algorithm = MultilayerPerceptron(layers, dropout=dropout)
    algorithm.train(text_classifier, epochs=epochs)

if __name__ == "__main__":
    run()