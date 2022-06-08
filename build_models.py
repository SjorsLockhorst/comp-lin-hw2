"""
FILE: build_models.py

File that contains code that builds all models.

Authors: Tijn, Gaby, Felix, Sjors
"""

import os
import pickle

from nltk.corpus import conll2002 as conll

from custom_chunker import ConsecutiveNPChunker
from features import base_line_features, test_features, base_line_and_history, testing_features


training = conll.chunked_sents("ned.train")

MODEL_DIR = "models"


def _get_path(filename):
    return os.path.join(MODEL_DIR, filename)


def train_model_with_feature_map(feature_map, train_sents):
    return ConsecutiveNPChunker(feature_map, train_sents)


def pickle_model(model, pickle_path):
    with open(pickle_path, "wb") as file:
        pickle.dump(model, file)


def train(pickle_path, feature_map, train_sents=training):
    print(f"Training model with {feature_map}, saving to {pickle_path}")
    model = train_model_with_feature_map(feature_map, train_sents=train_sents)
    pickle_model(model, pickle_path)


def train_all(model_mapper):
    for filename, feature_map in model_mapper.items():
        path = _get_path(filename)
        train(path, feature_map)


if __name__ == "__main__":
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    MODELS = {
        "base_test.pickle": test_features,
        "base_line.pickle": base_line_features,
        "base_and_history": base_line_and_history,
        "latest_test.pickle": testing_features
    }
    train_all(MODELS)
