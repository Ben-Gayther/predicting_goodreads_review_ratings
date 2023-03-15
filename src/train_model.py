#!/usr/bin/env python
# Now use preprocessed data with a HuggingFace transformer model to predict the rating (i.e. sentiment) of a book review.
# We want to predict the `rating` column, which is a number between 0 and 5. (Form of ordinal regression!)
import logging
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import argparse
from functools import partial
import logging
import pathlib
import torch


def tokenize_function(examples: Dataset, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    """Tokenize data"""
    rating = examples['rating']
    examples = tokenizer(
        examples['text'], truncation=True, padding='max_length', max_length=max_length)
    examples['label'] = rating
    return examples


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


def cli() -> argparse.Namespace:
    """Create command line interface for training model"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        default='data/processed_goodreads_train.csv')
    parser.add_argument('--model', type=str, default='distilbert-base-uncased')
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--output', type=str, default='models/')
    parser.add_argument('--test_run', action='store_true')
    parser.add_argument('--logging', type=str, default='INFO')
    args = parser.parse_args()
    return args


def main(args):
    # Set logging level
    logging.basicConfig(level=args.logging)

    # Detect device
    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Read data
    df = pd.read_csv(args.input)
    if args.test_run:
        df = df.sample(100)
        logging.info('Doing test run with only 100 samples')
    logging.info(f'Read data from {args.input}')

    # Split data into train and test
    # stratify to keep the same distribution of ratings in train and test
    train, test = train_test_split(df, test_size=0.2, stratify=df['rating'])
    logging.info('Split data into train and test (80/20)')

    # Create dataset
    train_dataset = Dataset.from_pandas(train)
    test_dataset = Dataset.from_pandas(test)

    # Tokenize data
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    logging.info(f'Using {args.model} tokenizer and model')

    train_dataset = train_dataset.map(
        partial(tokenize_function, tokenizer=tokenizer, max_length=args.max_length), batched=True, remove_columns=['rating', 'text', 'user_id', 'book_id', 'review_id'])
    test_dataset = test_dataset.map(
        partial(tokenize_function, tokenizer=tokenizer, max_length=args.max_length), batched=True, remove_columns=['rating', 'text', 'user_id', 'book_id', 'review_id'])
    logging.info('Tokenized data')

    id2label = {k: k for k in range(6)}
    label2id = {k: k for k in range(6)}

    # Create model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, id2label=id2label, label2id=label2id, num_labels=6)
    model.to(device)

    results_dir = args.output + args.model
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Create training arguments
    training_args = TrainingArguments(
        output_dir=results_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
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
    train_results = trainer.train()
    train_results_df = pd.DataFrame(train_results.metrics, index=[0])
    train_results_df.to_csv(f'{results_dir}/train_results.csv', index=False)
    logging.info(
        f'Trained model and saved results to {results_dir}/train_results.csv')

    # Evaluate model
    eval_results = trainer.evaluate()
    eval_results_df = pd.DataFrame(eval_results, index=[0])
    eval_results_df.to_csv(f'{results_dir}/eval_results.csv', index=False)
    logging.info(
        f'Evaluated model and saved results to {results_dir}/eval_results.csv')

    # Save model
    trainer.save_model(results_dir)
    tokenizer.save_pretrained(results_dir)
    logging.info(f'Saved model and tokenizer to {results_dir}')


if __name__ == '__main__':
    # Parse command line arguments
    args = cli()

    main()
