# Prepare data for training and testing
python src/prepare_data.py --input data/goodreads_train.csv \
                           --output data/processed_goodreads_train.csv

python src/prepare_data.py --input data/goodreads_test.csv \
                           --output data/processed_goodreads_test.csv
