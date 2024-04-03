#!/bin/bash
# Runs all shell scripts in the main directory

# retrieve kaggle dataset if not already downloaded, unzip, and move to data folder
# Make sure you have kaggle installed and have your API token in ~/.kaggle/kaggle.json!
if [ ! -f "data/goodreads_train.csv" ]; then
    kaggle competitions download -c goodreads-books-reviews-290312
    unzip goodreads-books-reviews-290312.zip -d goodreads-books-reviews-290312
    mkdir -p data
    mv goodreads-books-reviews-290312/* data
    rm -rf goodreads-books-reviews-290312
fi

MODEL_NAME=nlptown/bert-base-multilingual-uncased-sentiment
LEARNING_RATE=2e-5
MAX_LENGTH=128
BATCH_SIZE=1
EPOCHS=1
TEST_RUN="--test_run" # remove "--test_run" to run the full training pipeline

mkdir -p models/$MODEL_NAME

# ./prepare_data.sh data/goodreads_train.csv data/processed_goodreads_train.csv
# ./prepare_data.sh data/goodreads_test.csv data/processed_goodreads_test.csv
./train.sh data/processed_goodreads_train.csv $MODEL_NAME $LEARNING_RATE $MAX_LENGTH $BATCH_SIZE $EPOCHS models/ $TEST_RUN
./make_preds.sh data/processed_goodreads_test.csv models/$MODEL_NAME $BATCH_SIZE submission.csv $TEST_RUN
