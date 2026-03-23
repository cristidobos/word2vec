import numpy as np


def evaluate(embeddings, word_to_id, filepath):
    with open(filepath) as f:
        correct = 0
        total = 0
        for line in f:
            line = line.strip()
            if line.startswith(":"):
                continue
            words = line.lower().split()
            if any(w not in word_to_id for w in words):
                continue

            res = embeddings[word_to_id[words[1]]] - embeddings[word_to_id[words[0]]] + embeddings[word_to_id[words[2]]]
            expected = word_to_id[words[3]]
            actual = most_similar(res, embeddings,
                                  exclude=[word_to_id[words[0]], word_to_id[words[1]], word_to_id[words[2]]])

            if expected == actual:
                correct += 1
            total += 1

        score = (correct / total) * 100
        return score



def most_similar(vec, embeddings, exclude=None):
    norms = np.linalg.norm(embeddings, axis = 1)
    scores = embeddings @ vec / (norms * np.linalg.norm(vec))
    if exclude is not None:
        for idx in exclude:
            scores[idx] = -np.inf
    return np.argmax(scores)

