def _get_word_starts_capital(word):
    if word:
        return word[0].isupper()
    else:
        return False


def get_prev_word(sentence, i, history):
    if i - 1 > 0:
        return sentence[i - 1]
    else:
        return ("", "<START>")


def get_current_word_starts_capital(sentence, i, history, naive=True):
    word, _ = sentence[i]
    if naive:
        return _get_word_starts_capital(word)
    if i == 0:
        return False
    else:
        return _get_word_starts_capital(word)


def get_prev_word_starts_capital(sentence, i, history, naive=True):
    word, _ = get_prev_word(sentence, i, history)
    if naive:
        return _get_word_starts_capital(word)
    else:
        if i != 1:
            return _get_word_starts_capital(word)
        else:
            return False
