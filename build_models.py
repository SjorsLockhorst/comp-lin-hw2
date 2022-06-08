"""
FILE: build_models.py

File that contains code that builds all models.

Authors: Tijn, Gaby, Felix, Sjors
"""

import sys
import os
import pickle
import re

from nltk.corpus import conll2002 as conll

from custom_chunker import ConsecutiveNPChunker
from features import base_line_features, test_features, base_line_and_history, testing_features
import features


training = conll.chunked_sents("ned.train")

MODEL_DIR = "models"


def _get_path(filename):
    return os.path.join(MODEL_DIR, filename)


def pickle_model(model, pickle_path):
    with open(pickle_path, "wb") as file:
        pickle.dump(model, file)


def train(pickle_path, feature_map, train_sents=training):
    print(f"Training model with {feature_map}, saving to {pickle_path}")
    model = ConsecutiveNPChunker(feature_map, train_sents=train_sents)
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
    # If run without CLI arguments, just build all models in MODELS
    if len(sys.argv) == 1:
        train_all(MODELS)
    # If run with arguments, use first argument as pickle filename, and second as
    # function name of feature map in feature.py. Then train only that model.
    else:
        pickle_filename = sys.argv[1]
        if not re.match(r"^.*\.pickle$", pickle_filename):
            pickle_filename = f"{pickle_filename}.pickle"
        try:
            feature_map_name = sys.argv[2]
        except IndexError:
            print("No feature map provided, so using base line as default")
            feature_map = base_line_features
        else:
            try:
                feature_map = getattr(features, feature_map_name)
            except AttributeError:
                raise ImportError(
                    f"Canot import {feature_map_name} from features.py, are you sure it exists?")
        model_map = {
            pickle_filename: feature_map
        }
        train_all(model_map)
