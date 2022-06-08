import pickle
import re
import os

path = os.path.join(os.path.dirname(__file__), "dutch_names.pickle")
with open(path, "rb") as file:
    dutch_names = pickle.load(file)


def _get_word_starts_capital(word):
    if word:
        return word[0].isupper()
    else:
        return False


def _get_word_is_alpha(word):
    return word.isalpha()


def _get_word_is_numeric(word):
    return word.isnumeric()


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


def get_word_is_dutch_name(sentence, i, history, names=dutch_names):
    word, _ = get_word(sentence, i, history)
    return word in names


def get_word_contains_percentage(sentence, i, history):
    word, _ = get_word(sentence, i, history)
    return bool(re.match(r"([0-9]*[.,]?[0-9]*%)|[Pp]ercentage|[Pp]rocent", word))


def get_word_is_alpha(sentence, i, history):
    word, _ = get_word(sentence, i, history)
    return _get_word_is_alpha(word)


def get_word_is_numeric(sentence, i, history):
    word, _ = get_word(sentence, i, history)
    return _get_word_is_alpha(word)


def get_prev_word_is_numeric(sentence, i, history):
    word, _ = get_prev_word(sentence, i, history)
    return _get_word_is_numeric(word)


def get_next_word_is_numeric(sentence, i, history):
    word, _ = get_next_word(sentence, i, history)
    return _get_word_is_numeric(word)


def get_prev_word_is_alpha(sentence, i, history):
    word, _ = get_prev_word(sentence, i, history)
    return _get_word_is_alpha(word)
