import logging
import pathlib
import re

import config as cfg
import pandas as pd


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
    spoiler = re.compile(r"\(view\s+spoiler\)\[.*?\(hide\s+spoiler\)\]")
    return spoiler.sub(r"", text)


def preprocess_text(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess text data and make new column 'text'"""
    df["text"] = (
        df["review_text"]
        .apply(remove_urls)
        .apply(remove_html)
        .apply(remove_spoiler_alert)
    )

    return df


def process_data(input_path: str, output_path: str) -> pd.DataFrame:
    # Make sure output directory exists
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    logging.info(f"Read data from {input_path}, fields are {df.columns}")

    df = preprocess_text(df)
    logging.info("Preprocessed data")

    df.to_csv(output_path, index=False)
    logging.info(f"Saved data to {output_path}")

    return df


def main():
    logging.basicConfig(level=cfg.logging_level, format=cfg.logging_format)

    process_data(cfg.input_train_data, cfg.output_train_data)
    process_data(cfg.input_test_data, cfg.output_test_data)

    logging.info("Finished preprocessing data")


if __name__ == "__main__":
    main()
