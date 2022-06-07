from custom_chunker import ConsecutiveNPChunker
from features import base_line_feature, test_features
from nltk.corpus import conll2002 as conll

training = conll.chunked_sents("ned.train")
testing = conll.chunked_sents("ned.testa")

my_recognizer = ConsecutiveNPChunker(base_line_feature, training)

my_recognizer.show_most_informative_features()

print(my_recognizer.accuracy(testing))
