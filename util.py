"""
FILE: util.py

File that contains all individual features, and some helper functions.

Authors: Gaby, Felix, Tijn, Sjors
"""

import pickle
import re
import os
from typing import List

from custom_types import TaggedWord

from nltk.stem.snowball import DutchStemmer


MISC_DATA_DIR = "misc_data"
path = os.path.join(os.path.dirname(__file__), MISC_DATA_DIR, "dutch_names.pickle")
with open(path, "rb") as file:
    dutch_names = pickle.load(file)

# TODO: Add more features


def _get_word_starts_capital(word: str) -> bool:
    """
    Check if a word starts with a capital letter.

    Parameters
    ----------
    word : str

    Returns
    -------
    bool
    """
    if word:
        return word[0].isupper()
    else:  # If word is empty string or None, assume it does not have a capital letter
        return False


def _get_word_is_alpha(word: str) -> bool:
    """
    Check if a word contains only letter characters.

    Parameters
    ----------
    word : str

    Returns
    -------
    bool
    """
    return word.isalpha()


def _get_word_is_numeric(word: str) -> bool:
    """
    Check if a word contains only numeric characters.

    Parameters
    ----------
    word : str

    Returns
    -------
    bool
    """
    return word.isnumeric()


def get_word(sentence: List[TaggedWord], i: int, history: List[str], naive=True) -> TaggedWord:
    """
    Get tagged word in a sequence of tagged words.

    Assumes the current position is given by `i`. Tries to return word in this position.
    Naive toggles wheter or not to lower case the word if it happens to be the first word.
    This is meant to preserve only capital letters for words that are always written with
    a capital letter (for example names, countries etc.).

    Parameters
    ----------
    sentence : List[TaggedWords]
        List of TaggedWords, the sentence in which to look.
    i : int
        The position of the current word in the sentence.
    history : List[str]
        The IOB tags we have assigned to the whole corpus so far.
    naive : bool
        The naive approach is just to return the previous word as it is found in the
        text. The non naive aproach is to make a word lower case if it happens to be the
        first word. In this way, only words that contain a 'natural' upper case are
        preserved.

    Returns
    -------
    TaggedWord
        The current tagged word in the sentence.
    """
    tagged_word = ("", "<START>")

    if i >= 0:
        if i < len(sentence):
            tagged_word = sentence[i]
        else:
            tagged_word = ("", "<END>")
    if naive:
        return tagged_word
    else:
        if i == 0:
            word, tag = tagged_word
            return (word.lower(), tag)

    return tagged_word


def get_next_word(sentence: List[TaggedWord], i: int, history: List[str]):
    """
    Get next tagged word in a sequence of tagged words.

    Parameters
    ----------
    sentence : List[TaggedWords]
        List of TaggedWords, the sentence in which to look.
    i : int
        The position of the current word in the sentence.
    history : List[str]
        The IOB tags we have assigned to the whole corpus so far.
    naive : bool
        The naive approach is just to return the previous word as it is found in the
        text. The non naive aproach is to make a word lower case if it happens to be the
        first word. In this way, only words that contain a 'natural' upper case are
        preserved.

    Returns
    -------
    TaggedWord
        The next tagged word in the sentence.
    """
    return get_word(sentence, i + 1, history)


def get_prev_word(sentence: List[TaggedWord], i: int, history: List[str]):
    """
    Get previous tagged word in a sequence of tagged words.

    Parameters
    ----------
    sentence : List[TaggedWords]
        List of TaggedWords, the sentence in which to look.
    i : int
        The position of the current word in the sentence.
    history : List[str]
        The IOB tags we have assigned to the whole corpus so far.
    naive : bool
        The naive approach is just to return the previous word as it is found in the
        text. The non naive aproach is to make a word lower case if it happens to be the
        first word. In this way, only words that contain a 'natural' upper case are
        preserved.

    Returns
    -------
    TaggedWord
        The previous tagged word in the sentence.
    """
    return get_word(sentence, i - 1, history)


def get_next_word_starts_capital(
    sentence: List[TaggedWord],
    i: int,
    history: List[str]
) -> bool:
    """
    Check if next word in sentence starts with a capital letter.

    Parameters
    ----------
    sentence : List[TaggedWords]
        List of TaggedWords, the sentence in which to look.
    i : int
        The position of the current word in the sentence.
    history : List[str]
        The IOB tags we have assigned to the whole corpus so far.

    Returns
    -------
    bool
        Wheter the next word in the sentence is capitalized.
    """
    word, _ = get_next_word(sentence, i, history)
    return _get_word_starts_capital(word)


def get_current_word_starts_capital(
    sentence: List[TaggedWord],
    i: int,
    history: List[str],
    naive: bool = True
) -> bool:
    """
    Check if current word in sentence starts with a capital letter.

    Parameters
    ----------
    sentence : List[TaggedWords]
        List of TaggedWords, the sentence in which to look.
    i : int
        The position of the current word in the sentence.
    history : List[str]
        The IOB tags we have assigned to the whole corpus so far.
    naive : bool
        The naive approach is just to return the previous word as it is found in the
        text. The non naive aproach is to make a word lower case if it happens to be the
        first word. In this way, only words that contain a 'natural' upper case are
        preserved.

    Returns
    -------
    bool
        Wheter the current word in the sentence is capitalized.
    """
    word, _ = sentence[i]
    if naive:
        return _get_word_starts_capital(word)
    if i == 0:
        return False
    else:
        return _get_word_starts_capital(word)


def get_prev_word_starts_capital(
        sentence: List[TaggedWord],
        i: int,
        history: List[str],
        naive: bool = True
) -> bool:
    """
    Check if previous word in sentence starts with a capital letter.

    Parameters
    ----------
    sentence : List[TaggedWords]
        List of TaggedWords, the sentence in which to look.
    i : int
        The position of the current word in the sentence.
    history : List[str]
        The IOB tags we have assigned to the whole corpus so far.
    naive : bool
        The naive approach is just to return the previous word as it is found in the
        text. The non naive aproach is to make a word lower case if it happens to be the
        first word. In this way, only words that contain a 'natural' upper case are
        preserved.

    Returns
    -------
    bool
        Wheter the previous word in the sentence is capitalized.
    """
    word, _ = get_word(sentence, i - 1, history, naive=naive)
    if naive:
        return _get_word_starts_capital(word)
    else:
        if i != 1:
            return _get_word_starts_capital(word)
        else:
            return False


def get_prev_iob_in_chunk(sentence: List[TaggedWord], i: int, history: List[str]) -> str:
    """
    Extract the previous IOB tag from the history.

    Parameters
    ----------
    sentence : List[TaggedWords]
        List of TaggedWords, the sentence in which to look.
    i : int
        The position of the current word in the sentence.
    history : List[str]
        The IOB tags we have assigned to the whole corpus so far.

    Returns
    -------
    str
        Last IOB tag from history, or start tag if history is empty.
    """
    if history:  # Check so we can also do this for the very first word in the corpus.
        return history[-1]
    else:
        return "<START>"


def get_word_length(sentence: List[TaggedWord], i: int, history: List[str]) -> int:
    """
    Extract the word length of the current word.

    Parameters
    ----------
    sentence : List[TaggedWords]
        List of TaggedWords, the sentence in which to look.
    i : int
        The position of the current word in the sentence.
    history : List[str]
        The IOB tags we have assigned to the whole corpus so far.

    Returns
    -------
    int
        Length of current word.
    """
    return len(sentence[i][0])


def get_word_is_dutch_name(
        sentence: List[TaggedWord],
        i: int,
        history: List[str],
        names=dutch_names
) -> bool:
    """
    Check if word is in a set of Dutch names.

    Extract the word length of the current word.

    Parameters
    ----------
    sentence : List[TaggedWords]
        List of TaggedWords, the sentence in which to look.
    i : int
        The position of the current word in the sentence.
    history : List[str]
        The IOB tags we have assigned to the whole corpus so far.
    names : Set[str] : default = dutch_names
        Set of names to check against word. Default is a pickled set of Dutch names
        from one of the earlier Notebooks from the course.
    """
    word, _ = get_word(sentence, i, history)
    return word in names


def get_word_contains_percentage(
    sentence: List[TaggedWord],
    i: int,
    history: List[str]
) -> bool:
    """
    Check if a word is a percentage.

    This is done by seeing of a regex matches. This regex is designed to match any
    written percentage as digits, or the word 'percentage' or 'procent'.
    This is meant to find the named entity percent.

    Parameters
    ----------
    sentence : List[TaggedWords]
        List of TaggedWords, the sentence in which to look.
    i : int
        The position of the current word in the sentence.
    history : List[str]
        The IOB tags we have assigned to the whole corpus so far.

    Returns
    -------
    bool
        Wheter or not the word matches the regex.
    """
    word, _ = get_word(sentence, i, history)
    return bool(re.match(r"([0-9]*[.,]?[0-9]*%)|[Pp]ercentage|[Pp]rocent", word))


def get_word_is_alpha(sentence: List[TaggedWord], i: int, history: List[str]) -> bool:
    """
    Check if a word has only alphabetic characters.

    Parameters
    ----------
    sentence : List[TaggedWords]
        List of TaggedWords, the sentence in which to look.
    i : int
        The position of the current word in the sentence.
    history : List[str]
        The IOB tags we have assigned to the whole corpus so far.

    Returns
    -------
    bool
    """
    word, _ = get_word(sentence, i, history)
    return _get_word_is_alpha(word)


def get_word_is_numeric(sentence: List[TaggedWord], i: int, history: List[str]) -> bool:
    """
    Check if a word has only numeric characters.

    Parameters
    ----------
    sentence : List[TaggedWords]
        List of TaggedWords, the sentence in which to look.
    i : int
        The position of the current word in the sentence.
    history : List[str]
        The IOB tags we have assigned to the whole corpus so far.

    Returns
    -------
    bool
    """
    word, _ = get_word(sentence, i, history)
    return _get_word_is_alpha(word)


def get_prev_word_is_numeric(sentence: List[TaggedWord], i: int, history: List[str]) -> bool:
    """
    Check if a prev word has only numeric characters.

    Parameters
    ----------
    sentence : List[TaggedWords]
        List of TaggedWords, the sentence in which to look.
    i : int
        The position of the current word in the sentence.
    history : List[str]
        The IOB tags we have assigned to the whole corpus so far.

    Returns
    -------
    bool
    """
    word, _ = get_prev_word(sentence, i, history)
    return _get_word_is_numeric(word)


def get_next_word_is_numeric(sentence: List[TaggedWord], i: int, history: List[str]) -> bool:
    """
    Check if a prev word has only numeric characters.

    Parameters
    ----------
    sentence : List[TaggedWords]
        List of TaggedWords, the sentence in which to look.
    i : int
        The position of the current word in the sentence.
    history : List[str]
        The IOB tags we have assigned to the whole corpus so far.

    Returns
    -------
    bool
    """
    word, _ = get_next_word(sentence, i, history)
    return _get_word_is_numeric(word)


def get_prev_word_is_alpha(sentence: List[TaggedWord], i: int, history: List[str]) -> bool:
    """
    Check if a prev word has only alphabetic characters.

    Parameters
    ----------
    sentence : List[TaggedWords]
        List of TaggedWords, the sentence in which to look.
    i : int
        The position of the current word in the sentence.
    history : List[str]
        The IOB tags we have assigned to the whole corpus so far.

    Returns
    -------
    bool
    """
    word, _ = get_prev_word(sentence, i, history)
    return _get_word_is_alpha(word)


def get_word_stem(
    sentence: List[TaggedWord],
    i: int,
    history: List[str],
    stemmer=DutchStemmer()
) -> str:
    """
    Use nltk.stem.snowball.DutchStemmer to stem current word.

    Parameters
    ----------
    sentence : List[TaggedWords]
        List of TaggedWords, the sentence in which to look.
    i : int
        The position of the current word in the sentence.
    history : List[str]
        The IOB tags we have assigned to the whole corpus so far.

    Returns
    -------
    str
        The stem of the current word
    """
    word, _ = get_word(sentence, i, history)
    return stemmer.stem(word)


# TODO write docstrings for these
def get_word_suffix(sentence: List[TaggedWord], i: int, history: List[str]) -> str:
    word, _ = get_word(sentence, i, history)
    stem = get_word_stem(sentence, i, history)
    return re.sub(rf"{re.escape(stem)}", "", word.lower())


def get_word_is_year(sentence: List[TaggedWord], i: int, history: List[str]) -> bool:
    word, _ = sentence[i]
    if _get_word_is_numeric(word):
        return len(word) == 4
    return False


def get_word_is_date_format(sentence: List[TaggedWord], i: int, history: List[str]) -> bool:
    word, _ = get_word(sentence, i, history)
    return bool(re.match(r"([0-9]{2}(-|/)[0-9]{2}(-|/)[0-9]{4})", word))
