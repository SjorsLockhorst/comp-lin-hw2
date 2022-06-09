"""
FILE: build_models.py

File that contains code that builds all models.

Authors: Tijn, Gaby, Felix, Sjors
"""

import sys
import os
import pickle
import re
from typing import Tuple, List, Dict

from nltk.corpus import conll2002 as conll

from custom_chunker import ConsecutiveNPChunker
from custom_types import FeatureMap, TaggedWord
import features as ft


training = conll.chunked_sents("ned.train")

MODEL_DIR = "models"


# TODO: Expand docstrings
# TODO: maybe print training time?

def pickle_model(model: ConsecutiveNPChunker, path: str):
    """Pickle a model to a path."""
    with open(path, "wb") as file:
        pickle.dump(model, file)


def train(
        pickle_path: str,
        feature_map: FeatureMap,
        train_sents: List[Tuple[TaggedWord, str]] = training,
        algorithm: str = "NaiveBayes",
        verbose: int = 0
):
    """
    Train a model with a given feature map and train sentences
    and save it to a pickle path.
    """
    print(f"Training model with {feature_map}, saving to {pickle_path}")
    model = ConsecutiveNPChunker(
        feature_map,
        train_sents=train_sents,
        algorithm=algorithm,
        verbose=verbose
    )
    path = os.path.join(MODEL_DIR, pickle_path)
    pickle_model(model, path)


def train_all(model_mapper: Dict[str, FeatureMap]):
    """
    Train all models with standard settings, pickle them.
    """
    for filename, feature_map in model_mapper.items():
        train(filename, feature_map)


if __name__ == "__main__":
    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)
    MODEL_MAP = {
        "base_line.pickle":  ft.base_line_features,
        "abstract_features.pickle": ft.abstract_features,
        "abstract_features_plus.pickle": ft.abscract_features_plus,
        "abscract_features_plus_history.pickle": ft.abscract_features_and_history,
    }
    # If run without CLI arguments, just build all models in MODELS
    if len(sys.argv) == 1:
        train_all(MODEL_MAP)
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
            feature_map = ft.base_line_features
        else:
            try:
                feature_map = getattr(ft, feature_map_name)
            except AttributeError:
                raise ImportError(
                    f"Canot import {feature_map_name} from features.py, are you sure it exists?")
        model_map = {
            pickle_filename: feature_map
        }
        train_all(model_map)
