#!/bin/sh

INPUT=$1
MODEL=$2
BATCH_SIZE=$3
OUTPUT=$4

python src/eval_model.py --input $INPUT \
                         --model $MODEL \
                         --batch_size $BATCH_SIZE \
                         --output $OUTPUT