
import numpy as np
import pandas as pd


def get_keyphrases(keyphrase_file='input/keyphrases.csv'):
    keyphrases = pd.read_csv(
        open(keyphrase_file, 'r'),
        header=None,
        index_col=None
    )
    keyphrases = np.reshape(keyphrases.values, (-1,))
    keyphrases = keyphrases.tolist()

    return keyphrases


def write_result(keyphrases, keyphrase_labels):
    result = pd.DataFrame({'keyphrase': keyphrases, 'label': keyphrase_labels})
    result = result.sort_values('label')
    result.to_csv(open('output/result.csv', 'w'), index=None)
