import pytest
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertForSequenceClassification,
)
from train_model import (
    create_model,
    prepare_dataset,
    compute_metrics,
)
import torch


@pytest.fixture
def dummy_data():
    # Technically, I do not use the text in dummy data for the tests
    # I only use the ratings (ground truth) and dummy predictions
    return pd.DataFrame(
        {
            "text": [
                "This book was excellent!",
                "This book was terrible",
                "Great book",
                "This book was okay",
                "This book was bad",
            ],
            "rating": [5, 1, 4, 3, 2],
            "user_id": [1000, 2000, 3000, 4000, 5000],
            "book_id": [101, 202, 303, 404, 505],
            "review_id": [1, 2, 3, 4, 5],
        }
    )


@pytest.fixture
def dummy_predictions():
    predictions = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            [0.2, 0.3, 0.4, 0.5, 0.7, 0.6],
            [0.3, 0.4, 0.5, 0.8, 0.7, 0.6],
            [0.4, 0.9, 0.6, 0.7, 0.8, 0.5],
        ]
    )
    labels = torch.tensor([5, 1, 4, 3, 2])
    return predictions, labels


@pytest.fixture
def model_name():
    return "prajjwal1/bert-small"


@pytest.fixture
def max_length():
    return 10


@pytest.fixture
def tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)


def test_prepare_dataset(dummy_data, tokenizer, max_length):
    dataset = prepare_dataset(dummy_data, tokenizer, max_length)
    assert isinstance(dataset, Dataset)
    assert "label" in dataset.column_names
    assert dataset["label"] == [5, 1, 4, 3, 2]
    assert "rating" not in dataset.column_names


def test_create_model(model_name):
    model = create_model(model_name)
    assert isinstance(
        model, (AutoModelForSequenceClassification, BertForSequenceClassification)
    )
    assert model.num_labels == 6


def test_compute_metrics(dummy_predictions):
    metrics = compute_metrics(dummy_predictions)
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert metrics["accuracy"] == 0.6
    assert metrics["f1"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
