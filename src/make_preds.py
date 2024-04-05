import logging
from functools import partial

import config as cfg
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from transformers.pipelines.pt_utils import KeyDataset


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

    model_path = cfg.output_dir + cfg.model_name
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    logging.info("Loaded model")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    logging.info("Loaded tokenizer")

    test_df = pd.read_csv(cfg.output_test_data)

    test_df["text"] = test_df["text"].astype(str)
    test_df["text"] = test_df["text"].apply(lambda x: x.replace("nan", ""))

    if cfg.test_run:
        test_df = test_df.sample(100)
        logging.info("Doing test run with only 100 samples")

    logging.info("Loaded test data")

    test_dataset = Dataset.from_pandas(test_df)
    test_dataset = test_dataset.map(
        partial(
            tokenizer_without_labels,
            tokenizer=tokenizer,
            max_length=tokenizer.model_max_length,
        ),
        batched=True,
        remove_columns=test_df.columns.drop("text").to_list(),
    )
    logging.info("Tokenized test data")

    test_df["rating"] = 0  # Placeholder for predictions

    # Use pipeline to get predictions
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        truncation=True,
        max_length=cfg.max_length,
    )

    pred_ratings = []
    for out in classifier(KeyDataset(test_dataset, "text"), batch_size=cfg.batch_size):
        pred_ratings.extend(
            [int(out["label"].split("-")[0])]
        )  # convert "label": "4-star" to 4

    test_df["rating"] = pred_ratings

    # Check distribution of predictions is sensible
    logging.info(test_df.rating.value_counts())

    test_df = test_df[["review_id", "rating"]]  # Keep only review_id and rating columns

    test_df.to_csv(cfg.submission_name, index=False)

    logging.info("Saved predictions to csv file")


if __name__ == "__main__":
    main()
