from bertopic import BERTopic
import pandas as pd
import numpy as np
import pickle
import logging
import os

PATH_DIR = './'
os.chdir(PATH_DIR)

logging.basicConfig(level=logging.WARNING)

file_ending = '_original_v2'
PATH_OUT = 'Data/bert_out/'
PATH_MODEL = 'Data/bert_out/bert_fitted_model_reduced'

def load_pickle(filename):
    with open(PATH_OUT+filename+file_ending+'.pickle', 'rb') as handle:
        object = pickle.load(handle)
    print(filename+file_ending+" file loaded")
    return object

def save_to_pickle(object, filename, end):
    with open(PATH_OUT+filename+file_ending+end+'.pickle', 'wb') as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(filename+file_ending+" saved to pickle file")

if __name__ == "__main__":

    corpus = load_pickle("corpus")
    topic_model = BERTopic.load(PATH_MODEL)

    topic_distr, topic_token_distr = topic_model.approximate_distribution(corpus, use_embedding_model=True, batch_size=1000, window=4, stride=8)

    probs_df = pd.DataFrame(topic_distr)

    end = '_v2'

    save_to_pickle(probs_df, 'topic_distr', end)
    save_to_pickle(topic_token_distr, 'topic_token_distr', end)