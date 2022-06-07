from util import get_word_starts_capital


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


def get_prev_word(sentence, i, history):
    if i - 1 > 0:
        return sentence[i - 1]
    else:
        return ("", "<START>")


def get_current_word_starts_capital(sentence, i, history, naive=True):
    word, _ = sentence[i]
    if naive:
        return get_word_starts_capital(word)
    if i == 0:
        return False
    else:
        return get_word_starts_capital(word)


def get_prev_word_starts_capital(sentence, i, history, naive=True):
    word, _ = get_prev_word(sentence, i, history)
    if naive:
        return get_word_starts_capital(word)
    else:
        if i != 1:
            return get_word_starts_capital(word)
        else:
            return False


def base_line_feature(sentence, i, history):
    """
    Feature that extracts info about both the current,
    previous and next word.
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
