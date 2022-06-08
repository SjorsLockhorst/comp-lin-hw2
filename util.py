import re


def _get_word_starts_capital(word):
    if word:
        return word[0].isupper()
    else:
        return False


def get_prev_word(sentence, i, history, naive=True):
    if i - 1 > 0:
        return get_word(sentence, i-1, history, naive=naive)
    else:
        return ("", "<START>")


def get_next_word(sentence, i, history):
    if i + 1 < len(sentence):
        return get_word(sentence, i + 1, history)
    else:
        return ("", "<END>")


def get_next_word_starts_capital(sentence, i, history):
    word, _ = get_next_word(sentence, i, history)
    return _get_word_starts_capital(word)


def get_current_word_starts_capital(sentence, i, history, naive=True):
    word, _ = sentence[i]
    if naive:
        return _get_word_starts_capital(word)
    if i == 0:
        return False
    else:
        return _get_word_starts_capital(word)


def get_prev_word_starts_capital(sentence, i, history, naive=True):
    word, _ = get_prev_word(sentence, i, history, naive=naive)
    if naive:
        return _get_word_starts_capital(word)
    else:
        if i != 1:
            return _get_word_starts_capital(word)
        else:
            return False


def get_word(sentence, i, history, naive=True):
    word, tag = sentence[i]
    if naive:
        return (word, tag)
    else:
        if i == 0:
            return (word.lower(), tag)
        return (word, tag)


def get_prev_iob_in_chunk(sentence, i, history):
    if history:
        return history[-1]
    else:
        return "<START>"


def get_word_length(sentence, i, history):
    return len(sentence[i][0])
