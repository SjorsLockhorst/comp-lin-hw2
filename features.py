"""
FILE: features.py

Authors: Tijn, Felix, Gaby and Sjors.

File that contains all functions that extract feature sets from each word in a corpus.
Each function takes sentence, i and history as arguments.
"""

import util


# TODO: write better docstrings
# TODO: Create meaningful feature sets for comparison
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


def base_line_features(*args, **kwargs):
    """
    Extracts base line features from each word in a sentence.
    """
    word, pos = util.get_word(*args, naive=False)
    word_capital = util.get_current_word_starts_capital(*args)
    prev_word, prev_pos = util.get_prev_word(*args, naive=False)
    prev_starts_capital = util.get_prev_word_starts_capital(*args, naive=False)

    return {
        "pos": pos,
        "word": word,
        "word capital": word_capital,
        "prev word capital": prev_starts_capital,
        "prevpos": prev_pos,
        "prev word": prev_word
    }


def abstract_features(*args, **kwargs):
    """Chunker features designed to test the Chunker class for correctness
        - the POS tag of the word
        - the entire history of IOB tags so far
            formatted as a tuple because it needs to be hashable
    """
    next_word, next_tag = util.get_next_word(*args, naive=False)
    # two_words_next, two_words_next_tag = util.get_two_words_next(*args)

    # features["prev word + word"] = (features["prev word"], features["word"])
    # features["word + nextword"] = (features["word"], features["next word"])
    # features["2 back + prev word"] = (two_words_back, features["prev word"])
    # features["2 back pos + prev word pos"] = (two_words_back_tag, features["prevpos"])
    # features["word suffix"] = util.get_word_suffix(*args)

    features = base_line_features(*args)
    features["next word"] = next_word
    features["next pos"] = next_tag
    features["is alpha"] = util.get_word_is_alpha(*args)
    features["is num"] = util.get_word_is_numeric(*args)
    features["word stem"] = util.get_word_stem(*args)
    features["capital count"] = util.get_count_capital(*args)
    features["prefix"] = util.get_prefix(*args)
    features["is acronym"] = util.get_acronym(*args)
    features["is abbreviation"] = util.get_abbreviation(*args)
    features["word contains percentage"] = util.get_word_contains_percentage(*args)
    return features


def abstract_plus_specific(*args, **kwargs):
    features = abstract_features(*args)
    features["is money"] = util.get_money(*args)
    features["is dutch name"] = util.get_word_is_dutch_name(*args)
    features["word contains percentage"] = util.get_word_contains_percentage(*args)
    features["is year"] = util.get_word_is_year(*args)
    features["is date format"] = util.get_word_is_date_format(*args)
    features["is acronym"] = util.get_acronym(*args)
    features["is abbreviation"] = util.get_abbreviation(*args)
    return features
