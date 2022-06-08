import pickle

from nltk.corpus import conll2002 as conll

from custom_chunker import ConsecutiveNPChunker
from features import base_line_features, test_features, base_line_and_history


training = conll.chunked_sents("ned.train")


def train_model_with_feature_map(feature_map, train_sents):
    return ConsecutiveNPChunker(feature_map, train_sents)


def pickle_model(model, pickle_path):
    with open(pickle_path, "wb") as file:
        pickle.dump(model, file)


def train(pickle_path, feature_map, train_sents=training):
    print(f"Training model with {feature_map}, saving to {pickle_path}")
    model = train_model_with_feature_map(feature_map, train_sents=train_sents)
    pickle_model(model, pickle_path)


def train_test():
    FILENAME = "./test.pickle"
    train(FILENAME, test_features)


def train_base_line():
    FILENAME = "./base_line.pickle"
    train(FILENAME, base_line_features)


def train_base_and_history():
    FILENAME = "./base_and_history.pickle"
    train(FILENAME, base_line_and_history)


if __name__ == "__main__":
    train_base_line()
    train_test()
    train_base_and_history()
