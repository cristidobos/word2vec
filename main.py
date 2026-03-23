from utilities import tokenize, mapping, generate_training_data, build_unigram_table
from train import init_embeddings, train
from evaluation import evaluate
import numpy as np
import json


if __name__ == '__main__':
    with open("data/text8") as f:
        text = f.read()

    tokens = tokenize(text)
    word_to_id, id_to_word = mapping(tokens)
    X, y = generate_training_data(tokens, word_to_id)
    unigram_table = build_unigram_table(tokens, word_to_id)

    W_in, W_out = init_embeddings(len(word_to_id))
    embeddings = train(X, y, W_in, W_out, unigram_table)

    # Evaluate
    final_score = evaluate(embeddings, word_to_id, filepath="data/questions-words.txt")
    print(f"Accuracy: {final_score:.2f}%")

    # Save
    np.save("embeddings.npy", embeddings)
    with open("word_to_id.json", "w") as f:
        json.dump(word_to_id, f)