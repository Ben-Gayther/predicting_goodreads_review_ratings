#!/bin/sh

INPUT=$1
MODEL=$2
LEARNING_RATE=$3
MAX_LENGTH=$4
BATCH_SIZE=$5
EPOCHS=$6
OUTPUT=$7
TEST_RUN=$8

if [ "$TEST_RUN" = "--test_run" ]; then
    python src/train_model.py --input "$INPUT" \
        --model "$MODEL" \
        --learning_rate "$LEARNING_RATE" \
        --max_length "$MAX_LENGTH" \
        --batch_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --output "$OUTPUT" \
        --test_run
else
    python src/train_model.py --input "$INPUT" \
        --model "$MODEL" \
        --learning_rate "$LEARNING_RATE" \
        --max_length "$MAX_LENGTH" \
        --batch_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --output "$OUTPUT"
fi
