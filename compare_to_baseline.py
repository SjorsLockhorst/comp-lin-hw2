import sys

from nltk.corpus import conll2002 as conll

from evaluate_models import evaluate_model

if __name__ == "__main__":
    try:
        filename = sys.argv[1]
    except IndexError:
        raise ValueError("Must provide 1 command line argument: pickle filename to compare")

    BASE_LINE_FILE = "base_line.pickle"

    train_sents = conll.chunked_sents("ned.train")
    test_sents = conll.chunked_sents("ned.testa")

    # base_line_accuracy = evaluate_model(BASE_LINE_FILE, test_sents)
    # print(base_line_accuracy)
    tested_accuracy = evaluate_model(filename, test_sents)
    print(tested_accuracy)
