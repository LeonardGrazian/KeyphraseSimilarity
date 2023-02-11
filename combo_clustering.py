
import numpy as np
import pandas as pd
from sklearn.cluster import AffinityPropagation

# from utils import get_keyphrases, write_result
from embedding_clusters import get_embedding
from graph_clusters import get_similarity_graph, get_connected_components



def main():
    keyphrases = pd.read_csv(
        open('input/keyphrases_w_description.csv', 'r'),
        index_col=None
    )
    keyphrases = keyphrases.fillna('')
    serps = keyphrases['description'].values.tolist()
    keyphrases = keyphrases['keyword'].values.tolist()
    embeddings = [
        get_embedding(kp, serp=serp, verbose=True)
        for kp, serp in zip(keyphrases, serps)
    ]

    clustering = AffinityPropagation(random_state=5).fit(embeddings)
    keyphrase_labels = clustering.labels_

    min_cluster_id = min(keyphrase_labels)
    max_cluster_id = max(keyphrase_labels)

    result = pd.DataFrame({'keyphrase': keyphrases, 'label': keyphrase_labels})
    keyphrase_sublabel = {}
    for cluster_id, cluster_df in result.groupby('label'):
        cluster_keyphrases = cluster_df['keyphrase'].values.tolist()
        similarity_graph = get_similarity_graph(cluster_keyphrases)

        if similarity_graph.sum() == 0:
            for i, kp in enumerate(cluster_keyphrases):
                keyphrase_sublabel[kp] = i
        else:
            cluster_keyphrase_sublabels = get_connected_components(similarity_graph)
            for kp, sl in zip(cluster_keyphrases, cluster_keyphrase_sublabels):
                keyphrase_sublabel[kp] = sl

    result['sublabel'] = [
        keyphrase_sublabel[kp]
        for kp in result['keyphrase'].values.tolist()
    ]
    result = result.sort_values('label')
    result.to_csv(open('output/result.csv', 'w'), index=None)


if __name__ == '__main__':
    main()
