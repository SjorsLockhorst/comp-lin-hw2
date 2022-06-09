import unittest

from util import (
    get_prev_iob_in_chunk,
    get_word_is_dutch_name,
    get_word_contains_percentage,
    get_word_stem,
    get_word_suffix
)


class TestPrevIob(unittest.TestCase):

    def test_get_prev_iob_correct(self):
        self.assertEqual(get_prev_iob_in_chunk([], 0, ["O", "O", "B-MISC"]), "B-MISC")

    def test_get_prev_iob_correct_false(self):
        self.assertEqual(get_prev_iob_in_chunk([], 0, ["O"]), "O")

    def test_get_prev_iob_correct_when_no_history(self):
        self.assertEqual(get_prev_iob_in_chunk([], 0, []), "<START>")


class TestIsDutchName(unittest.TestCase):

    def test_if_recognizes_names(self):
        self.assertTrue(get_word_is_dutch_name([("Peter", "N")], 0, []))
        self.assertTrue(get_word_is_dutch_name([("Sjors", "N")], 0, []))
        self.assertTrue(get_word_is_dutch_name([("Tijn", "N")], 0, []))

    def test_if_does_not_recognize_non_names(self):
        self.assertFalse(get_word_is_dutch_name([("Tafel", "N")], 0, []))
        self.assertFalse(get_word_is_dutch_name([("Ventieldopje", "N")], 0, []))


class TestWordIsPercentage(unittest.TestCase):

    def test_true_positives(self):
        self.assertTrue(get_word_contains_percentage([("19.19%", "")], 0, []))
        self.assertTrue(get_word_contains_percentage([("19,19%", "")], 0, []))
        self.assertTrue(get_word_contains_percentage([("19%", "")], 0, []))
        self.assertTrue(get_word_contains_percentage([("%", "")], 0, []))

    def test_true_negatives(self):
        self.assertFalse(get_word_contains_percentage([("19.19.%", "")], 0, []))
        self.assertFalse(get_word_contains_percentage([("Sjors", "")], 0, []))
        self.assertFalse(get_word_contains_percentage([("19-11-1912", "")], 0, []))


class TestStemming(unittest.TestCase):

    def test_correct_stemming(self):
        self.assertEqual(get_word_stem([("Lopen", "")], 0, []), "lop")
        self.assertEqual(get_word_stem([("Nederlandse", "")], 0, []), "nederland")


class TestSuffix(unittest.TestCase):

    def test_correct_suffix(self):
        self.assertEqual(get_word_suffix([("Lopen", "")], 0, []), "en")
