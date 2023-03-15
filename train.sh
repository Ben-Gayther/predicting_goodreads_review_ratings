#!/bin/sh
# This script is used to train the model

python src/train_model.py --test_run # call all default parameters for a test run

# python src/train_model.py --input data/processed_goodreads_train.csv \
#                           --model distilbert-base-uncased \
#                           --learning_rate 2e-5 \
#                           --max_length 256 \
#                           --batch_size 1 \
#                           --epochs 1 \
#                           --output models/ \
