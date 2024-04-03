#!/usr/bin/env python
import argparse
import logging
import pathlib
from functools import partial

import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def tokenizer_with_labels(
    examples: Dataset, tokenizer: AutoTokenizer, max_length: int
) -> Dataset:
    """Tokenize data"""
    rating = examples["rating"]
    examples = tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=max_length
    )
    examples["label"] = rating
    return examples


def compute_metrics(eval_pred: torch.Tensor) -> dict:
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro"
    )
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}


def cli(opt_args=None) -> argparse.Namespace:
    """Create command line interface for training model"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, default="data/processed_goodreads_train.csv"
    )
    parser.add_argument("--model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--output", type=str, default="models/")
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

    df = pd.read_csv(args.input)
    df = df.dropna()
    if args.test_run:
        df = df.sample(1000)
        logging.info("Doing test run with only 1000 samples")
    logging.info(f"Read data from {args.input}")

    # stratify to keep the same distribution of ratings in train and test
    train, test = train_test_split(df, test_size=0.2, stratify=df["rating"])
    logging.info("Split data into train and test (80/20)")

    train_dataset = Dataset.from_pandas(train)
    test_dataset = Dataset.from_pandas(test)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    logging.info(f"Using {args.model} tokenizer and model")

    train_dataset = train_dataset.map(
        partial(tokenizer_with_labels, tokenizer=tokenizer, max_length=args.max_length),
        batched=True,
        remove_columns=["rating", "text", "user_id", "book_id", "review_id"],
        # input_columns=["text"],
    )
    test_dataset = test_dataset.map(
        partial(tokenizer_with_labels, tokenizer=tokenizer, max_length=args.max_length),
        batched=True,
        remove_columns=["rating", "text", "user_id", "book_id", "review_id"],
    )
    logging.info("Tokenized data")

    # 6 classes (ratings 0-5 stars)
    id2label = {
        "0": "0-star",
        "1": "1-star",
        "2": "2-star",
        "3": "3-star",
        "4": "4-star",
        "5": "5-star",
    }
    label2id = {v: k for k, v in id2label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        id2label=id2label,
        label2id=label2id,
        num_labels=len(id2label),
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    results_dir = args.output + args.model
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=results_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        weight_decay=0.01,
        learning_rate=args.learning_rate,
        fp16=True
        if torch.cuda.is_available()
        else False,  # mixed precision only available on GPU
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=results_dir,
        run_name="goodreads",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    train_results = trainer.train()

    train_results_df = pd.DataFrame(train_results.metrics, index=[0])
    train_results_df.to_csv(f"{results_dir}/train_results.csv", index=False)
    logging.info(f"Trained model and saved results to {results_dir}/train_results.csv")

    eval_results = trainer.evaluate()

    eval_results_df = pd.DataFrame(eval_results, index=[0])
    eval_results_df.to_csv(f"{results_dir}/eval_results.csv", index=False)
    logging.info(f"Evaluated model and saved results to {results_dir}/eval_results.csv")

    trainer.save_model(results_dir)
    tokenizer.save_pretrained(results_dir)
    logging.info(f"Saved transformer model and tokenizer to {results_dir}")


if __name__ == "__main__":
    args = cli()

    main(args)
