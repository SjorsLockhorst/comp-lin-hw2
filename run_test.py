from nltk.corpus import conll2002 as conll


from evaluate_models import evaluate_model
from build_models import train
from features import base_line_and_next_word

PATH = "./test.pickle"
train_sents = conll.chunked_sents("ned.train")
test_sents = conll.chunked_sents("ned.testa")[1000:1500]

train(PATH, base_line_and_next_word, train_sents=train_sents)
evaluate_model(PATH, test_sents, n_most_informative=100)
