import re
import nltk


def nltk_data(sentence):
    tokens=nltk.word_tokenize(sentence)
    print(tokens)

nltk_data("hej med dig")