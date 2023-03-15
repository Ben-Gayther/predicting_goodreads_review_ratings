#!/bin/sh

INPUT=$1
MODEL=$2
LEARNING_RATE=$3
MAX_LENGTH=$4
BATCH_SIZE=$5
EPOCHS=$6
OUTPUT=$7

python src/train_model.py --input $INPUT \
                         --model $MODEL \
                         --learning_rate $LEARNING_RATE \
                         --max_length $MAX_LENGTH \
                         --batch_size $BATCH_SIZE \
                         --epochs $EPOCHS \
                         --output $OUTPUT

# For testing
# python src/train_model.py --test_run # call all default parameters for a test run