from nltk.corpus import conll2002 as conll


from evaluate_models import evaluate_model
from build_models import train
from features import base_line_and_dutch_names, base_line_features

PATH = "./test.pickle"
BASE_LINE = "./base_line.pickle"

train_sents = conll.chunked_sents("ned.train")
test_sents = conll.chunked_sents("ned.testa")

N_MOST_INFORMATIVE = 10

train(PATH, base_line_and_dutch_names, train_sents=train_sents)
train(BASE_LINE, base_line_features, train_sents=train_sents)
evaluate_model(PATH, test_sents, n_most_informative=N_MOST_INFORMATIVE)
evaluate_model(BASE_LINE, test_sents, n_most_informative=N_MOST_INFORMATIVE)
