from util import (
    get_current_word_starts_capital,
    get_prev_word,
    get_prev_word_starts_capital
)


def test_features(sentence, i, history):
    """Chunker features designed to test the Chunker class for correctness
        - the POS tag of the word
        - the entire history of IOB tags so far
            formatted as a tuple because it needs to be hashable
    """
    _, pos = sentence[i]
    return {
        "pos": pos,
        "whole history": tuple(history)
    }


def base_line_features(sentence, i, history):
    """
    Extracts base line features from each word in a sentence.
    """
    args = (sentence, i, history)

    word, pos = sentence[i]
    word_capital = get_current_word_starts_capital(*args)
    prev_word, prev_pos = get_prev_word(*args)
    prev_starts_capital = get_prev_word_starts_capital(*args)

    return {
        "pos": pos,
        "word": word.lower(),
        "word capital": word_capital,
        "prev word capital": prev_starts_capital,
        "prevpos": prev_pos,
        "prev word": prev_word.lower()
    }
