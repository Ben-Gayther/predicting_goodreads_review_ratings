#!/usr/bin/env python
import argparse
import logging
from functools import partial

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer


# No longer have 'rating' column
def tokenizer_without_labels(
    examples: Dataset, tokenizer: AutoTokenizer, max_length: int
) -> Dataset:
    """Tokenize data"""
    examples = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    return examples


def cli(opt_args=None) -> argparse.Namespace:
    """Create command line interface for evaluating model"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, default="data/processed_goodreads_test.csv"
    )
    parser.add_argument("--model", type=str, default="models/distilbert-base-uncased/")
    # parser.add_argument('--training_args', type=str, default='models/distilbert-base-uncased/training_args.bin')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--output", type=str, default="submission.csv")
    parser.add_argument("--test_run", action="store_true")
    parser.add_argument("--logging", type=str, default="INFO")
    if opt_args is not None:
        args = parser.parse_args(opt_args)
    else:
        args = parser.parse_args()
    return args


def main(args):
    logging.basicConfig(level=args.logging)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to(device)
    logging.info("Loaded model")

    # Load training arguments (unused)
    # training_args = torch.load(args.training_args)
    # logging.info('Loaded training arguments')

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    logging.info("Loaded tokenizer")

    test_data = pd.read_csv(args.input)

    test_data["text"] = test_data["text"].astype(str)
    test_data["text"] = test_data["text"].apply(lambda x: x.replace("nan", ""))

    if args.test_run:
        test_data = test_data.sample(100)
        logging.info("Doing test run with only 100 samples")

    logging.info("Loaded test data")

    test_ds = Dataset.from_pandas(test_data)

    cols = test_ds.column_names
    test_ds = test_ds.map(
        partial(
            tokenizer_without_labels,
            tokenizer=tokenizer,
            max_length=tokenizer.model_max_length,
        ),
        batched=True,
        remove_columns=cols,
    )
    logging.info("Tokenized test data")

    test_data["rating"] = 0

    trainer = Trainer(model)
    preds = trainer.predict(test_ds).predictions
    preds = np.argmax(preds, axis=1)

    test_data.rating = preds

    # Check distribution of predictions
    logging.info(test_data.rating.value_counts())

    test_data.to_csv(args.output, index=False)

    logging.info("Saved predictions to csv file")


if __name__ == "__main__":
    # Parse command line arguments
    args = cli()

    main(args)
