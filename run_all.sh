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

MODEL_NAME=distilbert-base-uncased
LEARNING_RATE=2e-5
MAX_LENGTH=128
BATCH_SIZE=1
EPOCHS=1
TEST_RUN=""
# TEST_RUN="--test_run" # uncomment to run a test run

mkdir -p models/$MODEL_NAME

. prepare_data.sh data/goodreads_train.csv data/processed_goodreads_train.csv
. prepare_data.sh data/goodreads_test.csv data/processed_goodreads_test.csv
. train.sh data/processed_goodreads_train.csv $MODEL_NAME $LEARNING_RATE $MAX_LENGTH $BATCH_SIZE $EPOCHS models/
. eval.sh data/processed_goodreads_test.csv models/$MODEL_NAME models/$MODEL_NAME/training_args.bin $BATCH_SIZE data/predictions.csv
