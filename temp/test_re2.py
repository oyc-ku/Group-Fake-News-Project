import re2 as re
import line_profiler
import pandas as pd


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

@line_profiler.profile
def clean_text(text: str):
    text = text.lower()
    text = text.replace("\t", " ")
    text = text.replace("\n", " ")
    text = re_whitespace.sub(" ", text)
    text = re_email.sub("<email>", text)
    text = re_url.sub("<url>", text)
    text = re_date.sub("<date>", text)
    text = re_number.sub("<num>", text)

    return text


@line_profiler.profile
def main():
    content = pd.read_csv("./../data/995,000_rows.csv", usecols=["content"], nrows=10000)
    content = content["content"]
    content.dropna(inplace=True)
    cleaned = content.apply(clean_text)

    print(cleaned)


if __name__ == "__main__":
    main()