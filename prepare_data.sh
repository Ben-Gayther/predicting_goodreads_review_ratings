#!/bin/sh
# Prepare data for training and testing
INPUT=$1
OUTPUT=$2
python src/prepare_data.py --input $INPUT \
                           --output $OUTPUT