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

Then run the following to download the data, train the model, and make predictions on the test set:

```bash
poetry shell # activate the virtual environment, if using poetry
./run_all.sh # this will call prepare_data.sh, train_model.sh, and eval.sh
```

The predictions will be saved in `data/predictions.csv`.

The kaggle notebook which executes the same code is `kagglenotebook.ipynb` in order to utilise Kaggle's GPUs.

A small notebook on exploratory data analysis is also included in `notebooks/EDA.ipynb`.
