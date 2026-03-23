from collections import Counter
import numpy as np


def tokenize(text, th=1e-5):
    words = text.split()
    n_words = len(words)

    word_counts = Counter(words)
    p = {word: 1 - np.sqrt(th / (word_counts[word] / n_words)) for word in word_counts}

    tokens = [word for word in words if np.random.random() > p[word] and word_counts[word] >= 5]

    return tokens

def mapping(tokens):
    word_to_index = {}
    index_to_word = {}
    for i, word in enumerate(set(tokens)):
        word_to_index[word] = i
        index_to_word[i] = word
    return word_to_index, index_to_word

def generate_training_data(tokens, word_to_id, C=5):
    X = []
    y = []
    n_tokens = len(tokens)

    for i in range(n_tokens):
        R = np.random.randint(1, C + 1)

        left = max(0, i - R)
        right = min(n_tokens - 1, i + R)

        for j in range(left, right + 1):
            if i == j:
                continue
            X.append(word_to_id[tokens[i]])
            y.append(word_to_id[tokens[j]])

    return np.asarray(X), np.asarray(y)

def build_unigram_table(tokens, word_to_id, length = int(1e8)):
    word_counts = Counter(tokens)
    word_counts = {word: word_counts[word] ** (3 / 4) for word in word_counts}

    total = sum(word_counts.values())
    unigram = {word: word_counts[word] / total for word in word_counts}

    table = []
    for word, prob in unigram.items():
        instances = int (prob * length)
        for i in range(instances):
            table.append(word_to_id[word])

    return np.asarray(table)

def sigmoid(x):
    x = np.clip(x, -10, 10)
    return 1 / (1 + np.exp(-x))



