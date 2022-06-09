"""
FILE: evaluate_models.py

File that evaluates all models.

Authors: Felix, Tijn, Gaby and Sjors
"""

import os
import re
import pickle
from typing import List, Tuple

from nltk.corpus import conll2002 as conll

from build_models import pickle_model
from custom_chunker import ConsecutiveNPChunker
from custom_types import TaggedWord
from nltk.chunk.util import ChunkScore


MODEL_DIR = "models"

test_sents = conll.chunked_sents("ned.testa")

# TODO write better docstrings


def load_model(pickle_filename: str) -> ConsecutiveNPChunker:
    """Load model from pickle file."""
    path = os.path.join(MODEL_DIR, pickle_filename)
    with open(path, "rb") as file:
        return pickle.load(file)


def evaluate_model(pickle_path, test_sents: List[Tuple[TaggedWord, str]]) -> ChunkScore:
    """Load model from pickle path and evaluate it on test sentences."""
    print(f"Evaluating base model, with file name {pickle_path}")
    model = load_model(pickle_path)
    model.show_most_informative_features(100)
    return model.accuracy(test_sents)


def eval_all_models(test_sents: List[Tuple[TaggedWord, str]]):
    """Evaluate all models in models directory, print their results."""
    model_files = os.listdir(MODEL_DIR)  # List all files in model directory
    pickle_files = [model_file for model_file in model_files if re.search(
        r"^.*\.pickle$", model_file)]  # Select only *.pickle files

    # Init best variables to use
    best_f_measure = 0
    best_model = None
    best_file = None

    # Load each model and evaluate it, if its better than best so far, update best.
    for model_filename in pickle_files:
        model = load_model(model_filename)
        print(f"Evaluating {model_filename}")
        accuracy = model.accuracy(test_sents)
        print(accuracy)
        f_measure = accuracy.f_measure()
        if f_measure > best_f_measure:
            print(f"Found new best model: {model_filename}")
            best_f_measure = f_measure
            best_model = model
            best_file = model_filename

    # Finally, write best to ./best.pickle
    print(f"Writing {best_file} to ./best.pickle")
    if best_model:
        pickle_model(best_model, "./best.pickle")
    else:
        raise Exception("Didn't find any models")


if __name__ == "__main__":
    eval_all_models(test_sents)
