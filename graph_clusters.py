
import numpy as np
from scipy import stats

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.corpus import wordnet

from sknetwork.topology import get_connected_components

from utils import get_keyphrases, write_result

# constants
STOPWORDS = set(
    stopwords.words('english')
    + [
        'time',
        'stream',
        'streaming',
        'live',
        'movie',
        'stats',
        'statistics'
    ]
)


def get_synonyms(k):
    synonyms = []
    for syn in wordnet.synsets(k):
        for l in syn.lemmas():
            synonyms.append(l.name())
    return set(synonyms)


def is_synonym(k1, k2):
    return k1 in get_synonyms(k2) or k2 in get_synonyms(k1)


def get_similarity(k1, k2):
    k1_words = k1.split(' ')
    k2_words = k2.split(' ')

    words_in_common = 0
    for k1w in k1_words:
        for k2w in k2_words:

            if (
                k1w in STOPWORDS
                or k2w in STOPWORDS
                or len(k1w) < 3
                or len(k2w) < 3
            ):
                continue
            if k1w == k2w or is_synonym(k1w, k2w):
                words_in_common += 1
    return words_in_common


def get_similarity_graph(keyphrases):
    num_keyphrases = len(keyphrases)
    similarity_graph = np.zeros((num_keyphrases, num_keyphrases))
    for i, k1 in enumerate(keyphrases):
        for j, k2 in enumerate(keyphrases):
            if k1 == k2:
                continue

            similarity = get_similarity(k1, k2)
            similarity_graph[i, j] = similarity
    return similarity_graph


def initialize(G, similarity_graph, keyphrases, keyphrase_labels, label):
    for i, kp in enumerate(keyphrases):
        if keyphrase_labels[i] == label:
            G.add_node(kp)

    for i in range(similarity_graph.shape[0]):
        for j in range(similarity_graph.shape[1]):
            if (
                keyphrase_labels[i] == label
                and keyphrase_labels[j] == label
                and similarity_graph[i, j] > 0
            ):
                G.add_edge(keyphrases[i], keyphrases[j])


def visualize_subgraph(similarity_graph, keyphrases, keyphrase_labels, label):
    label_mask = (keyphrase_labels == label).tolist()
    similarity_subgraph = similarity_graph[label_mask]
    similarity_subgraph = similarity_subgraph[:, label_mask]

    import networkx as nx
    import matplotlib.pyplot as plt
    G = nx.Graph()
    initialize(G, similarity_graph, keyphrases, keyphrase_labels, label)
    nx.draw_networkx(G)
    plt.show()


def main():
    keyphrases = get_keyphrases()
    similarity_graph = get_similarity_graph(keyphrases)
    keyphrase_labels = get_connected_components(similarity_graph)
    print('Found {} clusters'.format(np.unique(keyphrase_labels).shape[0]))
    print('Largest cluster has {} keyphrases'.format(
        stats.mode(keyphrase_labels, keepdims=False).count
    ))
    write_result(keyphrases, keyphrase_labels)
    visualize_subgraph(similarity_graph, keyphrases, keyphrase_labels, 0)


if __name__ == '__main__':
    main()
