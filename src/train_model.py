import logging
import pathlib
from functools import partial
from typing import Optional

import config as cfg
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


def tokenize_and_add_labels(
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


def create_model(model_name: str) -> AutoModelForSequenceClassification:
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
        model_name,
        id2label=id2label,
        label2id=label2id,
        num_labels=len(id2label),
        ignore_mismatched_sizes=True,
    )
    return model


def prepare_dataset(
    df: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int
) -> Dataset:
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(
        partial(tokenize_and_add_labels, tokenizer=tokenizer, max_length=max_length),
        batched=True,
        remove_columns=["rating", "text", "user_id", "book_id", "review_id"],
    )
    return dataset


def setup_trainer(
    model: AutoModelForSequenceClassification,
    train_dataset: Dataset,
    results_dir: str,
    eval_dataset: Optional[Dataset] = None,
) -> Trainer:
    training_args = TrainingArguments(
        output_dir=results_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        weight_decay=0.01,
        learning_rate=cfg.learning_rate,
        fp16=True
        if torch.cuda.is_available()
        else False,  # mixed precision only available on GPU
        evaluation_strategy="no" if cfg.full_dataset else "epoch",
        do_eval=not cfg.full_dataset,
        save_strategy="epoch",
        load_best_model_at_end=False if cfg.full_dataset else True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=results_dir,
        run_name="goodreads",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    return trainer


def main():
    logging.basicConfig(level=cfg.logging_level, format=cfg.logging_format)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    df = pd.read_csv(cfg.output_train_data)

    if cfg.test_run:
        df = df.sample(1000)
        logging.info("Doing test run with only 1000 samples")

    logging.info(f"Read data from {cfg.output_train_data}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = create_model(cfg.model_name)
    model.to(device)

    logging.info(f"Using {cfg.model_name} tokenizer and model for fine-tuning")

    results_dir = cfg.output_dir + cfg.model_name
    pathlib.Path(results_dir).mkdir(parents=True, exist_ok=True)

    if not cfg.full_dataset:
        # stratify to keep the same distribution of ratings in train and test
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["rating"])
        logging.info("Splitting data into train and test (80/20)")

        train_dataset = prepare_dataset(train_df, tokenizer, cfg.max_length)
        test_dataset = prepare_dataset(test_df, tokenizer, cfg.max_length)

        logging.info("Tokenized data")

        trainer = setup_trainer(model, train_dataset, results_dir, test_dataset)

        train_results = trainer.train()

        train_results_df = pd.DataFrame(train_results.metrics, index=[0])
        train_results_df.to_csv(f"{results_dir}/train_results.csv", index=False)
        logging.info(
            f"Trained model and saved results to {results_dir}/train_results.csv"
        )

        eval_results = trainer.evaluate()

        eval_results_df = pd.DataFrame(eval_results, index=[0])
        eval_results_df.to_csv(f"{results_dir}/eval_results.csv", index=False)
        logging.info(
            f"Evaluated model and saved results to {results_dir}/eval_results.csv"
        )

    else:
        logging.info("Using full dataset for training")
        full_dataset = prepare_dataset(df, tokenizer, cfg.max_length)
        logging.info("Tokenized data")

        trainer = setup_trainer(model, full_dataset, results_dir)

        train_results = trainer.train()

        train_results_df = pd.DataFrame(train_results.metrics, index=[0])
        train_results_df.to_csv(f"{results_dir}/train_results.csv", index=False)
        logging.info(
            f"Trained model and saved results to {results_dir}/train_results.csv"
        )

    trainer.save_model(results_dir)
    tokenizer.save_pretrained(results_dir)
    logging.info(f"Saved transformer model and tokenizer to {results_dir}")


if __name__ == "__main__":
    main()
