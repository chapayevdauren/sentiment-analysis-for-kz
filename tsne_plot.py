import matplotlib.pyplot as plt
import numpy as np
from gensim.models import KeyedVectors
from sklearn.manifold import TSNE

# Loading the vectors
# [Warning] Takes a lot of time
en_model = KeyedVectors.load_word2vec_format('data/cc.kk.300.vec')

# Limit number of tokens to be visualized
limit = 500
vector_dim = 300

# Getting tokens and vectors
words = []
embedding = np.array([])
index = 0
for word in en_model.vocab:
    # Break the loop if limit exceeds
    if index == limit:
        break

    # Getting token
    words.append(word)

    # Appending the vectors
    embedding = np.append(embedding, en_model[word])

    index += 1

# Reshaping the embedding vector
embedding = embedding.reshape(limit, vector_dim)


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for idx, label in enumerate(labels):
        x, y = low_dim_embs[idx, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)


# Creating the tsne plot [Warning: will take time]
tsne = TSNE(perplexity=30.0, n_components=2, init='pca', n_iter=5000)

low_dim_embedding = tsne.fit_transform(embedding)

# Finally plotting and saving the fig
plot_with_labels(low_dim_embedding, words)
