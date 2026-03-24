import pandas as pd
from parallel_pandas import ParallelPandas
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')

from fakenews_functions import clean_data, tokenize_and_remove_stopwords, stemming_words

def process_data(text: str):
    cleaned = clean_data(text)
    tokenized = tokenize_and_remove_stopwords(cleaned)
    stemmed = stemming_words(tokenized)
    return stemmed


ParallelPandas.initialize(n_cpu=8, split_factor=2)

file_chunks = pd.read_csv("./../data/995,000_rows.csv", usecols=["content"], chunksize=2*16)
for chunk_number, chunk in enumerate(file_chunks):
    data: pd.Series = chunk["content"]
    data.dropna(inplace=True)
    data = data.p_apply(process_data)
    data.to_csv(f"./../data/data_stemmed/stemmed_chunk{chunk_number:02}.csv")