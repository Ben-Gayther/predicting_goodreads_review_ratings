#!/bin/bash

# Retrieve kaggle dataset if not already downloaded, unzip, and move to data folder
# Make sure you have kaggle cli installed and have your API token in ~/.kaggle/kaggle.json!
if [ ! -f "data/goodreads_train.csv" ]; then
    kaggle competitions download -c goodreads-books-reviews-290312
    unzip goodreads-books-reviews-290312.zip -d goodreads-books-reviews-290312
    mkdir -p data
    mv goodreads-books-reviews-290312/* data
    rm -rf goodreads-books-reviews-290312 goodreads-books-reviews-290312.zip
fi
