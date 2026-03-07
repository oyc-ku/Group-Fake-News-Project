from functools import lru_cache

import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm
tqdm.pandas()

import line_profiler

import site
site.addsitedir(r"../")
from dataprocessing.fakenews_functions import clean_data, tokenize_and_remove_stopwords


stemmer = SnowballStemmer("english")

@lru_cache(maxsize=2**14)
def cached_stem(word: str):
    return stemmer.stem(word)

def cached_stemming_words(text: list[str]):
    stemmed_words = [cached_stem(word) for word in text]
    stemmed_text =  " ".join(stemmed_words)
    stemmed_text = stemmed_text.replace("< num >", "<num>")
    stemmed_text = stemmed_text.replace("< date >", "<date>")
    stemmed_text = stemmed_text.replace("< email >", "<email>")
    stemmed_text = stemmed_text.replace("< url >", "<url>")
    return stemmed_text

def nocache_stemming_words(text: list[str]):
    stemmed_words = [(stemmer.stem(word)) for word in text]
    stemmed_text =  " ".join(stemmed_words)
    stemmed_text = stemmed_text.replace("< num >", "<num>")
    stemmed_text = stemmed_text.replace("< date >", "<date>")
    stemmed_text = stemmed_text.replace("< email >", "<email>")
    stemmed_text = stemmed_text.replace("< url >", "<url>")
    return stemmed_text

@line_profiler.profile
def main():
    content = pd.read_csv("./../data/995,000_rows.csv", usecols=["content"], nrows=1000)
    content = content["content"]
    content.dropna(inplace=True)

    cleaned = content.progress_apply(clean_data)
    tokenized = cleaned.progress_apply(tokenize_and_remove_stopwords)

    cache_stemmed = tokenized.progress_apply(cached_stemming_words)
    nocache_stemmed = tokenized.progress_apply(nocache_stemming_words)

    print(cache_stemmed.equals(nocache_stemmed))

if __name__ == "__main__":
    main()