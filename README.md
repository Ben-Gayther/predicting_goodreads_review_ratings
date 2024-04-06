# Predicting Goodreads Review Ratings

This repository contains code (see `src/`) for predicting the rating of a book review on Goodreads (as part of the [Goodreads Challenge](https://www.kaggle.com/competitions/goodreads-books-reviews-290312)). The dataset contains more than 1.3 million book reviews on roughly 25,000 books. The goal is to predict the rating of a book review based on the text of the review. This repository is used to fine-tune a BERT model for this task.

## To run fine-tuning from scratch

Install packages using poetry:

```bash
poetry install
```

Alternatively, you can install the dependencies using `pip` and the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Then run the following to download the data, train the model, and make predictions on the test set.
(Make sure Kaggle API key is set to download the data!)

```bash
poetry shell # activate the virtual environment, if using poetry
./download_data.sh # download the data from Kaggle
./run_all.sh # this will call the various scripts in the src directories
```

The configuration settings for these scripts can be modified by changing the values in `config.yaml`.

The predictions will be saved to `submission.csv`.

The kaggle notebook which executes the same code is `kaggle_submission.ipynb` in order to utilise Kaggle's GPUs.

A small notebook containg some of the inital analysis is also included in `notebooks/development.ipynb`.

## To run the tests

Run the following command from the root directory:

```bash
python -m pytest
```

This will run the tests in the `tests/` directory and will output a coverage report to the terminal.
The configuration settings for pytest are in `pyproject.toml`.
