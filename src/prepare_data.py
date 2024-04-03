#!/usr/bin/env python
import argparse
import logging
import pathlib
import re

import polars as pl


def read_data(path: str) -> pl.DataFrame:
    """Read data from csv file"""
    df = pl.read_csv(path)
    return df


def remove_urls(text: str) -> str:
    """Remove urls from a string"""
    url_pattern = re.compile(r"https?://\S+|www\.\S+")
    return url_pattern.sub(r"", text)


def remove_html(text: str) -> str:
    """Remove html tags from a string"""
    html_pattern = re.compile("<.*?>")
    return html_pattern.sub(r"", text)


def remove_spoiler_alert(text: str) -> str:
    """Remove spoiler alert from a string (Goodreads specific)"""
    spoiler = re.compile(r"(\(view spoiler\).*?\(hide spoiler\))")
    return spoiler.sub(r"", text)


def preprocess_text(df: pl.DataFrame, text_col: str) -> pl.DataFrame:
    """Preprocess text data and make new column 'text'"""
    df = df.with_columns(
        [
            pl.col(text_col)
            .map_elements(remove_urls, return_dtype=str)
            .map_elements(remove_html, return_dtype=str)
            .map_elements(remove_spoiler_alert, return_dtype=str)
            .alias("text")
        ]
    )
    return df


def cli(opt_args=None) -> argparse.Namespace:
    """Create command line interface for preprocessing data"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/goodreads_train.csv")
    parser.add_argument(
        "--output", type=str, default="data/processed_goodreads_train.csv"
    )
    parser.add_argument("--logging", type=str, default="INFO")
    if opt_args is not None:
        args = parser.parse_args(opt_args)
    else:
        args = parser.parse_args()
    return args


def main(args):
    logging.basicConfig(level=args.logging)

    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    df = read_data(args.input)
    logging.info(f"Read data from {args.input}")
    logging.info(f"Fields are {df.columns}")

    df = preprocess_text(df, text_col="review_text")
    logging.info("Preprocessed data")

    df.write_csv(args.output)
    logging.info(f"Saved data to {args.output}")


if __name__ == "__main__":
    # Parse command line arguments
    args = cli()

    main(args)
