
import numpy as np
import pandas as pd

from sknetwork.topology import get_connected_components


def get_keyphrases():
    keyphrases = pd.read_csv(
        open('keyphrases.csv', 'r'),
        header=None,
        index_col=None
    )
    keyphrases = np.reshape(keyphrases.values, (-1,))
    keyphrases = keyphrases.tolist()

    # keyphrases = keyphrases[:5]
    # print(keyphrases)

    return keyphrases

    # return ['a b', 'a c', 'd a', 'x y', 'x z', 'x w', 'p q', 'p t', 't q']


def get_similarity(k1, k2):
    k1_words = k1.split(' ')
    k2_words = k2.split(' ')

    words_in_common = 0
    for k1w in k1_words:
        for k2w in k2_words:
            if k1w == k2w:
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


def write_result(keyphrases, keyphrase_labels):
    result = pd.DataFrame({'keyphrase': keyphrases, 'label': keyphrase_labels})
    result = result.sort_values('label')
    result.to_csv(open('result.csv', 'w'), index=None)


def main():
    keyphrases = get_keyphrases()
    similarity_graph = get_similarity_graph(keyphrases)
    keyphrase_labels = get_connected_components(similarity_graph)
    print('Found {} clusters'.format(np.unique(keyphrase_labels).shape[0]))
    write_result(keyphrases, keyphrase_labels)




if __name__ == '__main__':
    main()
