import pickle
from nltk.corpus import conll2002 as conll


def load_model(pickle_path):
    return pickle.load(open(pickle_path, "rb"))


def evaluate_model(pickle_path, test_sents, n_most_informative=10):
    print(f"Evaluating base model, with file name {pickle_path}")
    model = load_model(pickle_path)
    model.show_most_informative_features(n_most_informative)
    print(model.accuracy(test_sents))


def eval_base_model(test_sents):
    FILENAME = "base_line.pickle"
    evaluate_model(FILENAME, test_sents)


def eval_test_model(test_sents):
    FILENAME = "test.pickle"
    evaluate_model(FILENAME, test_sents)


def eval_base_and_history(test_sents):
    FILENAME = "base_and_history.pickle"
    evaluate_model(FILENAME, test_sents)


if __name__ == "__main__":
    test_sents = conll.chunked_sents("ned.testa")

    eval_base_model(test_sents)
    # eval_test_model(test_sents)
    eval_base_and_history(test_sents)
