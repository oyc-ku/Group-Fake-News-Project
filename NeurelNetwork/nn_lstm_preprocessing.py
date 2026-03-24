# Encode the data for LSTM model to use

# Encodes stemmed tokens as strings seperated by whitespace into stemmed tokens as integers in pytorch tensors.
# Encodes article string labels to binary integer labels (1 for reliable, 0 for fake) 

from gensim.models import Word2Vec
import torch
import pandas as pd
from parallel_pandas import ParallelPandas
ParallelPandas.initialize(n_cpu=8, split_factor=4)

EMBEDDING_MODEL_PATH = "./model1.model"
METADATA_PATH = "./../data/995,000_rows.csv"
STEMMED_DATA_PATH = "./../data/data_stemmed.csv"
LABELS_FAKE = {"fake", "hate", "rumor", "unreliable", "conspiracy", "bias", "junksci", "satire"}
LABELS_RELIABLE = {"reliable", "political", "clickbait"}


def convert_labels(label):
    if label in LABELS_FAKE:
        return 0
    elif label in LABELS_RELIABLE:
        return 1
    else:
        return pd.NA
        
def tokenlist_to_tokenindexes(embedding_model, tokens):
    return torch.tensor([embedding_model.wv.get_index(token) for token in tokens if token in embedding_model.wv], dtype=torch.long)

def preprocess_element(embedding_model, stemmed_token_string):
    token_strings = stemmed_token_string.split(" ")
    token_indexes = tokenlist_to_tokenindexes(embedding_model, token_strings)

def get_preprocessed_data():
    embedding_model = Word2Vec.load(EMBEDDING_MODEL_PATH)
    metadata = pd.read_csv(METADATA_PATH, usecols=["type"])
    stemmed_data_chunks = pd.read_csv(STEMMED_DATA_PATH, index_col=0, chunksize=2**16) 
    
    processed_chunks = []

    for chunk in stemmed_data_chunks:
        chunk = chunk["content"].p_apply(lambda x: preprocess_element(embedding_model, x))
        processed_chunks.append(chunk)

    data = pd.concat(processed_chunks).to_frame()
    data["type"] = metadata["type"].apply(convert_labels)
    
    data.dropna(inplace=True)

    return data
    
