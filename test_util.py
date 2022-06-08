import unittest

from util import (
    get_prev_iob_in_chunk,
    get_word_is_dutch_name,
    get_word_contains_percentage,
    get_word_stem
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
        self.assertTrue(get_word_is_dutch_name([("Tijn", "N")], 0, None))
        self.assertTrue(get_word_is_dutch_name([("Sjors", "N")], 0, None))

    def test_if_does_not_recognize_non_names(self):
        self.assertFalse(get_word_is_dutch_name([("Tafel", "N")], 0, None))
        self.assertFalse(get_word_is_dutch_name([("Ventieldopje", "N")], 0, None))


class TestWordIsPercentage(unittest.TestCase):

    def test_true_positives(self):
        self.assertTrue(get_word_contains_percentage([("19.19%", "")], 0, None))
        self.assertTrue(get_word_contains_percentage([("19,19%", "")], 0, None))
        self.assertTrue(get_word_contains_percentage([("19%", "")], 0, None))
        self.assertTrue(get_word_contains_percentage([("procent", "")], 0, None))
        self.assertTrue(get_word_contains_percentage([("percentage", "")], 0, None))
        self.assertTrue(get_word_contains_percentage([("percentage", "")], 0, None))
        self.assertTrue(get_word_contains_percentage([("%", "")], 0, None))

    def test_true_negatives(self):
        self.assertFalse(get_word_contains_percentage([("19.19.%", ""), None], 0, None))
        self.assertFalse(get_word_contains_percentage([("Sjors", ""), None], 0, None))
        self.assertFalse(get_word_contains_percentage([("19-11-1912", ""), None], 0, None))


class TestStemming(unittest.TestCase):

    def test_correct_stemming(self):
        self.assertEqual(get_word_stem([("Lopen", "")], 0, None), "lop")
        self.assertEqual(get_word_stem([("Nederlandse", "")], 0, None), "nederland")
