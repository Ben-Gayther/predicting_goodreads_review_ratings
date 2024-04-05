# predicting_goodreads_review_ratings

This repository contains code (see `src/`) for predicting the rating of a book review on Goodreads (as part of the [Goodreads Challenge](https://www.kaggle.com/competitions/goodreads-books-reviews-290312)).

## To run from scratch

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

The predictions will be saved in `data/predictions.csv`.

The kaggle notebook which executes the same code is `kagglenotebook.ipynb` in order to utilise Kaggle's GPUs.

A small notebook containg some of the inital analysis is also included in `notebooks/development.ipynb`.
