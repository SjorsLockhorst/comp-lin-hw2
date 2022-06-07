import pickle

from custom_chunker import ConsecutiveNPChunker
from features import base_line_features
from nltk.corpus import conll2002 as conll


def train_model_with_feature_map(feature_map):

    training = conll.chunked_sents("ned.train")
    return ConsecutiveNPChunker(feature_map, training)


def pickle_model(model, pickle_path="./base_line.pickle"):
    with open(pickle_path, "wb") as file:
        pickle.dump(model, file)


if __name__ == "__main__":
    print("Training model with base line settings, saving to base_line.pickle")
    model = train_model_with_feature_map(base_line_features)
    pickle_model(model)
