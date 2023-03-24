# predicting_goodreads_review_ratings

This repository contains code (see `src/`) for predicting the rating of a book review on Goodreads (as part of the [Goodreads Challenge](https://www.kaggle.com/competitions/goodreads-books-reviews-290312)).

## To run from scratch (note will need access to Kaggle API):
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
. run_all.sh # this will call prepare_data.sh, train_model.sh, and eval.sh
```

This will download the data into `data/`, train the model, and make predictions on the test set.
The predictions will be saved in `data/predictions.csv`.

`kagglenotebook.ipynb` contains the code for the Kaggle notebook to utilise Kaggle's GPUs. (Not updated yet.)

## EDA Notebook
`notebooks/EDA.ipynb` contains the code for the simple exploratory data analysis notebook.