from functools import lru_cache

import pandas as pd
import re2 as re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
stop_words = set(stopwords.words('english'))


pattern_number = r"\d[\d,\.]*"

MONTH = "(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|june?|july?|aug(ust)?|sep(tember)?|oct(ober)?|nov(ember)?|dec(ember)?)"
DAY_NUMBER = r"(0?[1-9]|[12][0-9]|3[01])"
MONTH_NUMBER = r"(0?[1-9]|1[0-2])"
YEAR_NUMBER = r"\d{4}"

pattern_date = (
    rf"({MONTH}(\.? ?{DAY_NUMBER}(st|nd|rd|th)?([,\./]? ?{YEAR_NUMBER}\.?)?| (of )?{YEAR_NUMBER}\.?))|"
    rf"({YEAR_NUMBER} {MONTH})|"
    rf"(\d{{1,4}}-{MONTH_NUMBER}-\d{{1,4}})|"
    rf"({DAY_NUMBER}(\.|st|nd|rd|th)? {MONTH} ({YEAR_NUMBER})?)"
)
pattern_email = r"[a-z\d\.]+@[a-z\d-]+(\.[a-z\d-]+)+"
pattern_url = r"""(https?://)?([a-z\d]+\.)+[a-z]{2,}([/\#:?][^\s<>\{\}\^"]*)?"""

options = re.Options()
options.never_capture = True

re_date = re.compile(pattern_date, options)
re_email = re.compile(pattern_email, options)
re_url = re.compile(pattern_url, options)
re_number = re.compile(pattern_number, options)
re_whitespace = re.compile(r" {2,}", options)

def clean_data(text: str):
    text = text.lower()
    text = text.replace("\t", " ")
    text = text.replace("\n", " ")
    text = re_whitespace.sub(" ", text)
    text = re_email.sub("<email>", text)
    text = re_url.sub("<url>", text)
    text = re_date.sub("<date>", text)
    text = re_number.sub("<num>", text)

    return text


def tokenize_and_remove_stopwords(text):
    tokens = nltk.word_tokenize(text)
    
    filtered_words = [word for word in tokens if word not in stop_words]
    return filtered_words


stemmer = SnowballStemmer("english")

@lru_cache(maxsize=2**14)
def cached_stem(word: str):
    return stemmer.stem(word)


def stemming_words(text: list[str]):
    stemmed_words = [cached_stem(word) for word in text]
    stemmed_text =  " ".join(stemmed_words)
    stemmed_text = stemmed_text.replace("< num >", "<num>")
    stemmed_text = stemmed_text.replace("< date >", "<date>")
    stemmed_text = stemmed_text.replace("< email >", "<email>")
    stemmed_text = stemmed_text.replace("< url >", "<url>")

    return stemmed_text


LABELS_FAKE = {"fake", "hate", "rumor", "unreliable", "conspiracy", "bias", "junksci", "satire"}
LABELS_RELIABLE = {"reliable", "political", "clickbait"}

def change_label(label):
    if label in LABELS_FAKE:
        return 0
    elif label in LABELS_RELIABLE:
        return 1
    else:
        return pd.NA


LABELS_FAKE_LIAR = {"false","half-true","pants-fire","barely-true"}
LABELS_RELIABLE_LIAR = {"true","mostly-true"}

def change_label_liar(label):
    if label in LABELS_FAKE_LIAR:
        return 0
    elif label in LABELS_RELIABLE_LIAR:
        return 1
    else:
        return pd.NA



if __name__ == "__main__":
    test_numbers = ["0", "69200", "2.1524", "200,000", "1357.15"]
    test_dates = ["Jan. 18, 2018", "January 24, 2018", "Feb. 23", "April 2017", "JAN 31.2018", "13th OCTOBER 2017",
                "JAN 31.2018", "APRIL 21/2017", "January 15", "1 Jan 2004", "Jan.12, 2018", "September of 2017",
                "2018 March", "July 5, 2016", "August 23rd 2013"]
    test_emails = ["tammyjcoffman@gmail.com", 
                   "info@treadwells-london.com", 
                   "rscdesigns@tampabay.rr.com",
                   "brent.kendall@wsj.com"
                   ]
    test_urls = ["Shutterstock.com",
                "http://www.wptv.com/", 
                "https://vimeo.com/ondemand/Awakeningof12strands", 
                "https://worldgovernmentsummit.org/api/publications/document?id=23747dc4-e97c-6578-b2f8-ff0000a7ddb6",
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "www.theguardian.com/media/2018/jan/23/never-get-high-on-your-own-supply-why-social-media-bosses-dont-use-social-media",
                "https://z5h64q92x9.net/proxy_u/ru-en.en/colonelcassad.livejournal.com/2962657.html"
                ]

    tests = [(re_number, test_numbers),
            (re_date, test_dates), 
            (re_email, test_emails), 
            (re_url, test_urls)]

    for pattern, test_cases in tests:
        for test_case in test_cases:
            result = pattern.fullmatch(test_case.lower())
            if result is None:
                raise re.error(f"Regular expression {pattern.pattern} did not match {test_case.lower()}.")
            else:
                print(repr(result.string))