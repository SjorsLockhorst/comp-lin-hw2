from util import (
    get_current_word_starts_capital,
    get_prev_word,
    get_prev_word_starts_capital,
    get_word,
    get_prev_iob_in_chunk,
    # get_word_length,
    get_next_word,
    # get_next_word_starts_capital,
    get_word_is_dutch_name,
    get_word_contains_percentage,
    get_word_is_alpha,
    get_word_is_numeric,
    get_prev_word_is_numeric,
    get_next_word_is_numeric
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

    word, pos = get_word(*args)
    word_capital = get_current_word_starts_capital(*args)
    prev_word, prev_pos = get_prev_word(*args)
    prev_starts_capital = get_prev_word_starts_capital(*args)

    return {
        "pos": pos,
        "word": word,
        "word capital": word_capital,
        "prev word capital": prev_starts_capital,
        "prevpos": prev_pos,
        "prev word": prev_word
    }


def base_line_and_history(sentence, i, history):
    """
    Extracts base line feature, and adds last part of history.
    """
    args = (sentence, i, history)
    features = base_line_features(*args)
    features["prev iob"] = get_prev_iob_in_chunk(*args)
    return features


def base_line_and_next_word(sentence, i, history):
    """
    Extracts base line features, and adds next word characteristics.
    """
    args = (sentence, i, history)
    features = base_line_features(*args)
    next_word, next_tag = get_next_word(*args)
    features["next word"] = next_word
    features["next pos"] = next_tag
    features["is dutch name"] = get_word_is_dutch_name(*args)
    return features


def testing_features(sentence, i, history):
    """Chunker features designed to test the Chunker class for correctness
        - the POS tag of the word
        - the entire history of IOB tags so far
            formatted as a tuple because it needs to be hashable
    """
    args = (sentence, i, history)
    features = base_line_and_next_word(*args)
    features["is alpha"] = get_word_is_alpha(*args)
    features["is num"] = get_word_is_numeric(*args)
    features["is dutch name"] = get_word_is_dutch_name(*args)
    features["word contains percentage"] = get_word_contains_percentage(*args)
    features["prev word is numeric"] = get_prev_word_is_numeric(*args)
    features["next word is numeric"] = get_next_word_is_numeric(*args)
    return features
