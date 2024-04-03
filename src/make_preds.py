import logging
from functools import partial

import config as cfg
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


def main():
    logging.basicConfig(level=cfg.logging_level, format=cfg.logging_format)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name)
    model.to(device)
    logging.info("Loaded model")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    logging.info("Loaded tokenizer")

    test_data = pd.read_csv(cfg.output_test_data)

    test_data["text"] = test_data["text"].astype(str)
    test_data["text"] = test_data["text"].apply(lambda x: x.replace("nan", ""))

    if cfg.test_run:
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

    test_data["rating"] = 0  # Placeholder for predictions

    trainer = Trainer(model)
    preds = trainer.predict(test_ds).predictions
    preds = np.argmax(preds, axis=1)

    test_data.rating = preds

    # Check distribution of predictions is sensible
    logging.info(test_data.rating.value_counts())

    test_data.to_csv(cfg.submission_name, index=False)

    logging.info("Saved predictions to csv file")


if __name__ == "__main__":
    main()
