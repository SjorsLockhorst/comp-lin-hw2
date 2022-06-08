"""
FILE: evaluate_models.py

File that evaluates all models.

Authors: Felix, Tijn, Gaby and Sjors
"""

import os
import re
import pickle
from nltk.corpus import conll2002 as conll

from build_models import pickle_model


MODEL_DIR = "models"


def load_model(pickle_filename):
    path = os.path.join(MODEL_DIR, pickle_filename)
    return pickle.load(open(path, "rb"))


def evaluate_model(pickle_path, test_sents):
    print(f"Evaluating base model, with file name {pickle_path}")
    model = load_model(pickle_path)
    return model.accuracy(test_sents)


def eval_all_models(test_sents):
    model_files = os.listdir(MODEL_DIR)
    pickle_files = [model_file for model_file in model_files if re.search(
        r"^.*\.pickle$", model_file)]
    best_f_measure = 0
    best_model = None
    best_file = None
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

    print(f"Writing {best_file} to ./best.pickle")
    pickle_model(best_model, "./best.pickle")


if __name__ == "__main__":
    test_sents = conll.chunked_sents("ned.testa")
    eval_all_models(test_sents)
