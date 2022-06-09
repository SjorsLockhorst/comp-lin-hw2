import unittest

from features import abstract_features
from build_models import train, training
from evaluate_models import evaluate_model, test_sents


class TestLatestModel(unittest.TestCase):

    def setUp(self):
        self.PICKLE_TEST_FILE = "latest_test.pickle"

    def test_accuracy(self):
        train(self.PICKLE_TEST_FILE, abstract_features, training)
        print(evaluate_model(self.PICKLE_TEST_FILE, test_sents))
