# Now use preprocessed data with a HuggingFace transformer model to classify the sentiment of a book review.
# (We want to predict the `rating` column, which is a number between 0 and 5.)
import logging
import polars as pl
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from prepare_data import read_data
import argparse
from functools import partial
import logging


def tokenize_function(examples: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    """Tokenize data"""
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(eval_pred: torch.Tensor) -> dict:
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro')
    accuracy = accuracy_score(labels, predictions)
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    """Create command line interface for training model"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        default='data/processed_goodreads_train.csv')
    parser.add_argument('--model', type=str, default='distilbert-base-uncased')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--output', type=str, default='models/')
    parser.add_argument('--logging', type=str, default='INFO')
    args = parser.parse_args()

    # Try with cardiffnlp/twitter-xlm-roberta-base-sentiment?

    # Set logging level
    logging.basicConfig(level=args.logging)

    # Read data
    df = pd.read_csv(args.input)
    logging.info(f'Read data from {args.input}')

    # Split data into train and test
    train, test = train_test_split(df, test_size=0.2)

    # Create dataset
    train_dataset = Dataset.from_pandas(train)
    test_dataset = Dataset.from_pandas(test)

    # Tokenize data
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    logging.info(f'Using {args.model} tokenizer and model')

    train_dataset = train_dataset.map(
        partial(tokenize_function, tokenizer=tokenizer), batched=True)
    test_dataset = test_dataset.map(
        partial(tokenize_function, tokenizer=tokenizer), batched=True)
    logging.info('Tokenized data')

    # Create model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=6)

    # Create training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='logs/',
        logging_steps=10,
        evaluation_strategy='steps',
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        save_strategy='steps',
        save_steps=1000,
        save_total_limit=2,
        run_name='goodreads'
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    # Train model
    # trainer.train()
    # TODO: fix this!

    logging.info('Trained model')

    # Evaluate model
    trainer.evaluate()
    logging.info('Evaluated model')

    # Save model
    trainer.save_model(args.output)
    logging.info(f'Saved model to {args.output}')


if __name__ == '__main__':
    main()
