import polars as pl
from nltk.corpus import stopwords
import string
import re
import pathlib
import argparse
import logging

STOP_WORDS = set(stopwords.words('english'))
PUNCT_TO_REMOVE = string.punctuation


def read_data(path: str) -> pl.DataFrame:
    """Read data from csv file"""
    df = pl.read_csv(path)  # try polars
    return df


def remove_urls(text: str) -> str:
    """Remove urls from a string"""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def remove_html(text: str) -> str:
    """Remove html tags from a string"""
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)


def remove_stopwords(text: str) -> str:
    """Remove stop words from a string"""
    return " ".join([word for word in str(text).split() if word not in STOP_WORDS])


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))


# Goodreads specific
def remove_spoiler_alert(text: str) -> str:
    spoiler = re.compile(r'(\(view spoiler\).*?\(hide spoiler\))')
    return spoiler.sub(r' ', text)


def preprocess(df: pl.DataFrame, text_col: str = 'text') -> pl.DataFrame:
    """Preprocess text data"""
    logging.info('Preprocessing data')
    df['text'] = df[text_col].apply(remove_urls)
    logging.info('Removed urls')

    df['text'] = df[text_col].apply(remove_html)
    logging.info('Removed html tags')

    df['text'] = df[text_col].apply(remove_stopwords)
    logging.info('Removed stop words')

    df['text'] = df[text_col].apply(remove_punctuation)
    logging.info('Removed punctuation')

    df['text'] = df[text_col].apply(remove_spoiler_alert)
    logging.info('Removed spoiler alert')

    return df


def main():
    """Create command line interface for preprocessing data"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        default='data/goodreads_train.csv')
    parser.add_argument('--output', type=str,
                        default='data/processed_goodreads_train.csv')
    parser.add_argument('--logging', type=str, default='INFO')
    args = parser.parse_args()

    # Set logging level
    logging.basicConfig(level=args.logging)

    # Create output directory
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Read data
    df = read_data(args.input)
    logging.info(f'Read data from {args.input}')

    # Preprocess data
    df = preprocess(df, text_col='review_text')

    # Save data
    df.to_csv(args.output)
    logging.info(f'Saved data to {args.output}')


if __name__ == '__main__':
    main()
