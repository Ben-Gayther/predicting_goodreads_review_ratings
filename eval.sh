#!/bin/sh

INPUT=$1
MODEL=$2
LEARNING_RATE=$3
BATCH_SIZE=$4
OUTPUT=$5

python src/eval_model.py --input $INPUT \
                         --model $MODEL \
                         --learning_rate $LEARNING_RATE \
                         --batch_size $BATCH_SIZE \
                         --output $OUTPUT