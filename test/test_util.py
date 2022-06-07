import unittest

from ..util import get_prev_iob_in_chunk


class UtilTestSuite(unittest.TestCase):

    def test_get_prev_iob_correct_b(self):
        self.assertTrue(get_prev_iob_in_chunk([], 0, ["O", "O", "B-MISC"]))

    def test_get_prev_iob_correct_i(self):
        self.assertTrue(get_prev_iob_in_chunk([], 0, ["O", "O", "I"]))

    def test_get_prev_iob_correct_false(self):
        self.assertFalse(get_prev_iob_in_chunk([], 0, ["O"]))

    def test_get_prev_iob_correct_when_no_history(self):
        self.assertFalse(get_prev_iob_in_chunk([], 0, []))
