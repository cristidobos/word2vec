import numpy as np
from utilities import sigmoid


def init_embeddings(vocab_size, dim=300):
    W_in = np.random.uniform(-0.5/dim, 0.5/dim, (vocab_size, dim))
    W_out = np.zeros((vocab_size, dim))

    return W_in, W_out

def train(X, y, W_in, W_out, unigram_table, init_alpha=0.025, K=10, epochs=5):
    indices = np.arange(len(X))

    total_steps = epochs * len(X)
    step = 0

    for _ in range(epochs):
        np.random.shuffle(indices)
        for idx in indices:
            alpha = max(init_alpha * (1 - step / total_steps), init_alpha * 0.0001)

            word, context = X[idx], y[idx]
            neg_samples = []
            while len(neg_samples) < K:
                sample = unigram_table[np.random.randint(len(unigram_table))]
                if sample != word and sample != context:
                    neg_samples.append(sample)

            if step % 10000 == 0:
                loss = (- np.log(sigmoid(np.dot(W_in[word], W_out[context])))
                        - np.sum([np.log(sigmoid(np.dot(-W_in[word], W_out[neg]))) for neg in neg_samples], axis=0))

                print(f"Progress: {100 * step / total_steps:.1f}% | Loss: {loss:.4f} | alpha: {alpha:.6f}")

            dw = (- (1 - sigmoid(np.dot(W_in[word], W_out[context]))) * W_out[context] +
                  np.sum([sigmoid(np.dot(W_in[word], W_out[neg])) * W_out[neg] for neg in neg_samples], axis=0))

            dc = - (1 - sigmoid(np.dot(W_in[word], W_out[context]))) * W_in[word]

            dn = []
            for sample in neg_samples:
                dn.append(sigmoid(np.dot(W_in[word], W_out[sample])) * W_in[word])

            W_in[word] -= alpha * dw
            W_out[context] -= alpha * dc
            for sample, gradient in zip(neg_samples, dn):
                W_out[sample] = W_out[sample] - alpha * gradient

            step += 1

    return W_in



