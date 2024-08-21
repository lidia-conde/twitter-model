import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import umap
import hdbscan
from bertopic import BERTopic
import pickle
import logging
import os

PATH_DIR = './'
os.chdir(PATH_DIR)

logging.basicConfig(level=logging.WARNING)

class BERTModel:
  def __init__(self, PATH_DIR, PATH_OUT, PATH_XLSX, file_ending):
    self.PATH_DIR = PATH_DIR
    self.PATH_OUT = PATH_OUT
    self.PATH_XLSX = PATH_XLSX
    self.file_ending = file_ending
    self.corpus = None
    self.embeddings = None
    self.umap_model = None
    self.hdbscan_model = None
    self.topic_model = None
    self.fitted_model = None
    self.topics = None
    self.probabilities = None
    self.size = 10000

  def set_size(self, size):
    self.size = size

  def update_files(self):
    self.corpus = self.load_pickle("corpus")
    self.embeddings = self.load_pickle("embeddings")
    self.umap_model = self.load_pickle("umap_model")
    self.hdbscan_model = self.load_pickle("hdbscan_model")
    self.topic_model = self.load_pickle("bert_model")
    print("All files updated.")

  def save_bert_model(self, topic_model, filename):
    embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    topic_model.save(self.PATH_OUT+filename+self.file_ending+'/', serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)
    print("BERTopic model saved with safetensors")

  def save_to_pickle(self, object, filename):
    with open(self.PATH_OUT+filename+self.file_ending+'.pickle', 'wb') as handle:
        pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(filename+self.file_ending+" saved to pickle file")

  def load_pickle(self, filename):
    try:
      with open(self.PATH_OUT+filename+self.file_ending+'.pickle', 'rb') as handle:
          object = pickle.load(handle)
      print(filename+self.file_ending+" file loaded")
      return object
    except:
      return None

  def read_dataset(self):
    df_tweets = pd.read_excel(self.PATH_XLSX, engine='openpyxl')
    print("Length of the dataset:", len(df_tweets))
    print("Dataset downloaded")
    return df_tweets

  def read_reduced_dataset(self):
    df_tweets = pd.read_excel(self.PATH_XLSX, engine='openpyxl')
    print("Length of the original dataset:", len(df_tweets))
    print("Number of tweets selected:", self.size)
    df = df_tweets[0:self.size]
    print("Dataset downloaded")
    return df

  def create_corpus(self):
    corpus=[]
    a=[]
    df = self.read_dataset()
    df_aux = df['filtered_text']
    for i in range(len(df_aux)):
        a=df_aux[i]
        if (pd.isna(a) == False):
                corpus.append(a)

    print("Corpus created")

    self.corpus = corpus
    self.save_to_pickle(corpus, 'corpus')

  def create_embeddings(self):
    if (self.corpus is None):
        self.corpus = self.load_pickle('corpus')

    # Load sentence transformer model
    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Create documents embeddings
    embeddings = sentence_model.encode(self.corpus, show_progress_bar=True)

    print("Embeddings created")

    self.embeddings = embeddings
    self.save_to_pickle(embeddings, 'embeddings')

  def create_umap_model(self):
    # Define UMAP model to reduce embeddings dimension
    umap_model = umap.UMAP(n_neighbors=200,
                        n_components=10,
                        min_dist=0.0,
                        metric='cosine',
                        low_memory=False)

    print("Umap model created")
    self.umap_model = umap_model
    self.save_to_pickle(umap_model, 'umap_model')

  def create_hdbscan_model(self):
    # Define HDBSCAN model to perform documents clustering
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=100,
                                metric='euclidean',
                                cluster_selection_method='eom',
                                prediction_data=True)

    print("HDBSCAN model created")
    self.hdbscan_model = hdbscan_model
    self.save_to_pickle(hdbscan_model, 'hdbscan_model')

  def create_bert_model(self):
    if (self.umap_model is None):
        self.umap_model =  self.load_pickle('umap_model')
    if (self.hdbscan_model is None):
        self.hdbscan_model =  self.load_pickle('hdbscan_model')

    # Create BERTopic model
    topic_model = BERTopic(top_n_words=20,
                    n_gram_range=(1,2),
                    calculate_probabilities=False,
                    umap_model= self.umap_model,
                    hdbscan_model=self.hdbscan_model,
                    verbose=True)

    print("BERT model created")
    self.topic_model =  topic_model
    self.save_to_pickle(self.topic_model, 'bert_model')

  def fit_bert_model(self):
    if (self.topic_model is None):
       self.topic_model =  self.load_pickle('topic_model')

    fitted_model = self.topic_model
    # Train model, extract topics and probabilities
    if (self.embeddings is None):
        topics, probabilities = fitted_model.fit_transform(self.corpus)
    else:
       topics, probabilities = fitted_model.fit_transform(self.corpus, self.embeddings)

    self.fitted_model = fitted_model
    self.topics = topics
    self.probabilities = probabilities

    print("Topic model fitted")

    self.save_bert_model(self.fitted_model, 'bert_fitted_model')
    self.save_to_pickle(self.topics, 'bert_topics')
    self.save_to_pickle(self.probabilities, 'bert_probabilities')

if __name__ == "__main__":
    PATH = 'Data/Dataset 2022/'
    PATH_XLSX = PATH + 'filtered_tweets.xlsx'
    file_ending = '_2022_v5'
    PATH_OUT = 'Data/bert_out/'

    bert = BERTModel(PATH_DIR, PATH_OUT, PATH_XLSX, file_ending)
    # bert.set_size(75000)
    bert.update_files()
    
    bert.create_corpus()

    bert.create_umap_model()
    bert.create_hdbscan_model()

    bert.create_embeddings()

    bert.create_bert_model()
    bert.fit_bert_model()