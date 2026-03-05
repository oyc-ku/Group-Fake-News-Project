import re2 as re
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
stop_words = set(stopwords.words('english'))


pattern_newlines = re.compile(r"\n")
pattern_tabs = re.compile(r"\t")
pattern_multiplewhitespace = re.compile(r"\s{2,}")

pattern_number = re.compile(r"\d[\d,\.]*")

months = "january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|aug|sep|oct|nov|dec"
pattern_date = re.compile(rf"(({months})\.? ?\d{{1,2}}([,\.]?[ /]?\d{{4}}\.?)?)|"
                        rf"(({months}) \d{{4}}\.?)|"
                        r"(\d{1,4}-\d{1,2}-\d{1,4})|"
                        rf"(\d{{1,2}}(\.|st|nd|th)? ({months}) (\d{{4}})?)|"
                        r"(\d{1,2}, \d{1,4})|"
                        rf"(({months}) of \d{{4}})|"
                        rf"(\d{{4}} ({months}))")

pattern_email = re.compile(r"\w+@([\w-]+\.)+\w+")
pattern_url = re.compile(r"(\w+://)?\w+\.\w+(.\w+)?[/\w?=-]*")

def clean_data(text):
    text = pattern_multiplewhitespace.sub(" ", text)
    text = pattern_newlines.sub("", text)
    text = pattern_tabs.sub("", text)
    text = text.lower()
    text = pattern_url.sub("<url>", text)
    text = pattern_email.sub("<email>", text)
    text = pattern_date.sub("<date>", text)
    text = pattern_number.sub("<num>", text)
    return text


def tokenize_and_remove_stopwords(text):
    tokens = nltk.word_tokenize(text)
    
    filtered_words = [word for word in tokens if word not in stop_words]
    return filtered_words
    # return " ".join(filtered_words)


stemmer = SnowballStemmer("english")
def stemming_words(text: list[str]):
    # words = text.split(" ")
    stemmed_words = [(stemmer.stem(word)) for word in text]
    stemmed_text =  " ".join(stemmed_words)
    stemmed_text = stemmed_text.replace("< num >", "<num>")
    stemmed_text = stemmed_text.replace("< date >", "<date>")
    stemmed_text = stemmed_text.replace("< email >", "<email>")
    stemmed_text = stemmed_text.replace("< url >", "<url>")

    return stemmed_text


if __name__ == "__main__":
    test_numbers = ["0", "69200", "2.1524", "200,000", "1357.15"]
    test_dates = ["Jan. 18, 2018", "January 24, 2018", "Feb. 23", "April 2017", "JAN 31.2018", "13th OCTOBER 2017",
                "JAN 31.2018", "APRIL 21/2017", "January 15", "18, 2018", "Jan.12, 2018", "September of 2017",
                "2018 March", "July 5, 2016"]
    test_emails = ["tammyjcoffman@gmail.com", "info@treadwells-london.com", "rscdesigns@tampabay.rr.com"]
    test_urls = ["Shutterstock.com", 
                "http://www.wptv.com/", 
                "https://vimeo.com/ondemand/Awakeningof12strands", 
                "https://worldgovernmentsummit.org/api/publications/document?id=23747dc4-e97c-6578-b2f8-ff0000a7ddb6",
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                ]


    pattern_newlines = re.compile(r"\n")
    pattern_tabs = re.compile(r"\t")
    pattern_multiplewhitespace = re.compile(r"\s{2,}")

    pattern_number = re.compile(r"\d[\d,\.]*")

    months = "January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Aug|Sep|Oct|Nov|Dec"
    pattern_date = re.compile(rf"(({months})\.? ?\d{{1,2}}([,\.]?[ /]?\d{{4}}\.?)?)|"
                            rf"(({months}) \d{{4}}\.?)|"
                            r"(\d{1,4}-\d{1,2}-\d{1,4})|"
                            rf"(\d{{1,2}}(\.|st|nd|th)? ({months}) (\d{{4}})?)|"
                            r"(\d{1,2}, \d{1,4})|"
                            rf"(({months}) of \d{{4}})|"
                            rf"(\d{{4}} ({months}))")

    pattern_email = re.compile(r"\w+@([\w-]+\.)+\w+")
    pattern_url = re.compile(r"(\w+://)?\w+\.\w+(.\w+)?[/\w?=-]*")


    tests = [(pattern_number, test_numbers),
            (pattern_date, test_dates), 
            (pattern_email, test_emails), 
            (pattern_url, test_urls)]

    for pattern, test_cases in tests:
        for test_case in test_cases:
            result = pattern.fullmatch(test_case)
            if result is None:
                raise re.PatternError(f"Regular expression {pattern.pattern} did not match {test_case}.")
            else:
                print(repr(result.string))