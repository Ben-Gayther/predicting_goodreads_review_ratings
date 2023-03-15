#!/usr/bin/env python
import polars as pl
from nltk.corpus import stopwords
from operator import itemgetter
import string
import re
import pathlib
import argparse
import logging

STOP_WORDS = set(stopwords.words('english'))
PUNCT_TO_REMOVE = string.punctuation


def read_data(path: str) -> pl.DataFrame:
    """Read data from csv file"""
    df = pl.read_csv(path, has_headers=True)
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
    """Remove punctuation from a string"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))


def remove_spoiler_alert(text: str) -> str:
    """Remove spoiler alert from a string (Goodreads specific)"""
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


def calc_time_diff(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate time difference between started_at and read_at"""
    # Note date format is Sun Jul 30 07:44:10 -0700 2017 [day month day time zone year]
    df = df.with_columns([
        pl.col('started_at').str.strptime(
            pl.Date, '%a %b %d %H:%M:%S %z %Y').alias('started_at'),
        pl.col('read_at').str.strptime(
            pl.Date, '%a %b %d %H:%M:%S %z %Y').alias('read_at'),
    ])

    df = df.with_column(
        (pl.col('read_at') - pl.col('started_at')
         ).cast(pl.Int32).alias('days_to_read')
    )

    return df


def calc_reviews_per_user(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate number of reviews per user"""
    return df.groupby('user_id').agg(
        pl.count('review_id').alias('reviews_per_user'))


def calc_votes_per_user(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate number of votes per user"""
    return df.groupby('user_id').agg(pl.sum('n_votes').alias('votes_per_user'))


def calc_reviews_per_book(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate number of reviews per book"""
    return df.groupby('book_id').agg(
        pl.count('review_id').alias('reviews_per_book'))


def calc_votes_per_book(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate number of votes per book"""
    return df.groupby('book_id').agg(pl.sum('n_votes').alias('votes_per_book'))


def calc_comments_per_book(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate number of comments per book"""
    return df.groupby('book_id').agg(
        pl.sum('n_comments').alias('comments_per_book'))

# Idea behind this is to perhaps use these extra features to weight the predictions of the transformer model
def add_new_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add new features to dataframe"""
    logging.info('Adding new features')

    df = calc_time_diff(df)
    logging.info('Calculated time difference')

    df = df.join(calc_reviews_per_user(df), on='user_id', how='left')
    logging.info('Calculated reviews per user')

    df = df.join(calc_votes_per_user(df), on='user_id', how='left')
    logging.info('Calculated votes per user')

    df = df.join(calc_reviews_per_book(df), on='book_id', how='left')
    logging.info('Calculated reviews per book')

    df = df.join(calc_votes_per_book(df), on='book_id', how='left')
    logging.info('Calculated votes per book')

    df = df.join(calc_comments_per_book(df), on='book_id', how='left')
    logging.info('Calculated comments per book')

    return df


def cli() -> argparse.Namespace:
    """Create command line interface for preprocessing data"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        default='data/goodreads_train.csv')
    parser.add_argument('--output', type=str,
                        default='data/processed_goodreads_train.csv')
    parser.add_argument('--logging', type=str, default='INFO')
    args = parser.parse_args()
    return args


def main(args):
    # Set logging level
    logging.basicConfig(level=args.logging)

    # Create output directory
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Read data
    df = read_data(args.input)
    logging.info(f'Read data from {args.input}')
    logging.info(f'Fields are {df.columns}')
    # Fields should be ["user_id", "book_id", "review_id", "rating", "review_text", "date_added", "date_updated", "read_at", "started_at", "n_votes", "n_comments"]

    # if logging.getLogger().isEnabledFor(logging.INFO):
    #     print(df)

    # Add new features (commented out for now)
    # df = add_new_features(df)
    # logging.info(f'Columns are now: {df.columns}')

    # Preprocess data
    df = preprocess(df, text_col='review_text')

    # Save data
    df.to_csv(args.output)
    logging.info(f'Saved data to {args.output}')


if __name__ == '__main__':
    # Parse command line arguments
    args = cli()

    main(args)
