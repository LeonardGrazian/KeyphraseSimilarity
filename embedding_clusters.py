
import os
from pathlib import Path
import pickle as pkl
from sklearn.cluster import AffinityPropagation

import openai
openai.api_key = os.environ.get('OPENAI_API_KEY')

from utils import get_keyphrases, write_result


# constants
EMBEDDING_DIR = 'embeddings/'


def get_embedding(kp, serp=None, use_cache=True, verbose=False):
    embedding_file = EMBEDDING_DIR + kp.replace(' ', '_') + '.pkl'
    if use_cache and Path(embedding_file).exists():
        if verbose:
            print('Using cached embedding for "{}"'.format(kp))
        embedding = pkl.load(open(embedding_file, 'rb'))
        return embedding

    if verbose:
        print('Fetching embedding for "{}"'.format(kp))
    if serp:
        response = openai.Embedding.create(
            input=kp + ' | ' + serp,
            model='text-embedding-ada-002'
        )
    else:
        response = openai.Embedding.create(
            input=kp,
            model='text-embedding-ada-002'
        )
    embedding = response['data'][0]['embedding']
    pkl.dump(
        embedding,
        open(embedding_file, 'wb')
    )
    return embedding


def main():
    keyphrases = get_keyphrases()
    embeddings = [get_embedding(kp, verbose=True) for kp in keyphrases]

    clustering = AffinityPropagation(random_state=5).fit(embeddings)
    keyphrase_labels = clustering.labels_

    write_result(keyphrases, keyphrase_labels)


if __name__ == '__main__':
    main()
