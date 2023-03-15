#!/usr/bin/env python
# Given a trained model, evaluate it on a test set and predict the rating of a book review, then save the predictions to a csv file.

import argparse
import logging
import pathlib
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from datasets import Dataset
from functools import partial

def cli() -> argparse.Namespace:
    """Create command line interface for evaluating model"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        default='data/processed_goodreads_test.csv')
    parser.add_argument('--model', type=str, default='models/distilbert-base-uncased/')
    parser.add_argument('--training_args', type=str, default='models/distilbert-base-uncased/training_args.bin')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--output', type=str, default='data/predictions.csv')
    parser.add_argument('--logging', type=str, default='INFO')
    args = parser.parse_args()
    return args

def main(args):
    # Set logging level
    logging.basicConfig(level=args.logging)

    # Detect device
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to(device)
    logging.info('Loaded model')

    # Load training arguments
    training_args = torch.load(args.training_args)
    logging.info('Loaded training arguments')

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    logging.info('Loaded tokenizer')

    # Load test data
    test_data = pd.read_csv(args.input)
    logging.info('Loaded test data')

    # Tokenize test data
    test_ds = Dataset.from_pandas(test_data)

    # No longer have 'rating' column
    def tokenize_function(examples: Dataset, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
        """Tokenize data"""
        examples = tokenizer(
            examples['text'], truncation=True, padding='max_length', max_length=max_length)

        return examples

    cols = test_ds.column_names
    test_ds = test_ds.map(partial(tokenize_function, tokenizer=tokenizer, max_length=tokenizer.model_max_length), batched=True, remove_columns=cols)
    logging.info('Tokenized test data')

    test_data['rating'] = 0

    trainer = Trainer(model)
    preds = trainer.predict(test_ds).predictions
    preds = np.argmax(preds, axis=1)

    # Save predictions to csv file
    test_data.rating = preds
    # Check distribution of predictions
    logging.info(test_data.rating.value_counts())
    test_data.to_csv(args.output)
    logging.info('Saved predictions to csv file')

if __name__ == '__main__':
    # Parse command line arguments
    args = cli()

    main()