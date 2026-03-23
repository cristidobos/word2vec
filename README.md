# word2vec

Skip-gram with Negative Sampling implemented in pure NumPy.

## Overview

This is an implementation of the word2vec Skip-gram model with negative sampling, following Mikolov et al. (2013). 

Trained on the [text8](http://mattmahoney.net/dc/text8.zip) corpus (~17M tokens derived from Wikipedia).

## Method

**Skip-gram** learns word embeddings by predicting context words from a center word within a sliding window.

**Negative Sampling** replaces the expensive softmax over the entire vocabulary with a binary classification task: distinguish real (center, context) pairs from `K` randomly sampled noise words. The objective per training pair is:

$$J = \log \sigma(v_o^T v_c) + \sum_{k=1}^{K} \log \sigma(-v_{n_k}^T v_c)$$

Additional details matching the original paper:
- **Subsampling** of frequent words with discard probability `1 - sqrt(t / f(w))`, reducing the influence of uninformative high-frequency words such as "the", "or", "in".
- **Unigram table** for negative sampling, weighted by `count^(3/4)` to smooth the frequency distribution
- **Linear learning rate decay** from 0.025 to near zero over training
- **Min count** threshold of 5 to discard rare words

## Gradients

We minimize $L = -J$ via SGD:

**Input embedding** (center word):

$$\frac{\partial L}{\partial v_c} = -(1 - \sigma(v_o^T v_c)) \cdot v_o + \sum_{k=1}^{K} \sigma(v_{n_k}^T v_c) \cdot v_{n_k}$$

**Output embedding** (positive context):

$$\frac{\partial L}{\partial v_o} = -(1 - \sigma(v_o^T v_c)) \cdot v_c$$

**Output embedding** (negative sample):

$$\frac{\partial L}{\partial v_{n_k}} = \sigma(v_{n_k}^T v_c) \cdot v_c$$

## Design Decisions

- **Skip-gram over CBOW** — better representations for rare words, semantic relationships
- **Negative sampling over hierarchical softmax** — simpler to implement and scales well with vocabulary size
- **Negative sampling over full softmax** — much faster

## Architecture

| File | Description |
|------|-------------|
| `utilities.py` | Tokenization, subsampling, vocab mapping, training pair generation, unigram table |
| `train.py` | Embedding initialization, training loop|
| `evaluation.py` | Analogy evaluation on the Google word analogy dataset, cosine similarity |
| `main.py` | Pipeline entry point |
| `qualitative.ipynb` | Interactive exploration: nearest neighbors, analogies|

## Results

Evaluated on the [Google word analogy dataset](https://github.com/tmikolov/word2vec/blob/master/questions-words.txt) (19K analogies, e.g. `king - man + woman = queen`):

**14% accuracy** on text8 (5 epochs, 300-dim embeddings, K=10 negative samples)

Qualitative examples:
```
most_similar("king")   → anshan, canute, lulach
most_similar("queen") → elizabeth, regnant, princess
france - paris + rome  → papacy
bigger - big + small    → smaller
```

## How to Run

**Dependencies:**
```bash
pip install numpy
```

**Train:**
```bash
python main.py
```

Saves `embeddings.npy` and `word_to_id.json` after training.

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embedding dimension | 300 |
| Window size (max) | 5 |
| Negative samples (K) | 10 |
| Initial learning rate | 0.025 |
| Epochs | 5 |
| Min word count | 5 |
| Subsampling threshold | 1e-5 |

## References

1. Mikolov et al., [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) (2013)
2. Mikolov et al., [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) (2013)
