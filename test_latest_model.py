import unittest

from features import best_features
from build_models import train, training
from evaluate_models import evaluate_model, test_sents


class TestLatestModel(unittest.TestCase):

    def setUp(self):
        self.PICKLE_TEST_FILE = "latest_greatest.pickle"

    def test_accuracy(self):
        train(self.PICKLE_TEST_FILE, best_features, training)
        print(evaluate_model(self.PICKLE_TEST_FILE, test_sents))
