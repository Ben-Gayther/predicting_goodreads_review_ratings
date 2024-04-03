#!/bin/sh

INPUT=$1
MODEL=$2
BATCH_SIZE=$3
OUTPUT=$4
TEST_RUN=$5

if [ "$TEST_RUN" = "--test_run" ]; then
    python src/make_preds.py --input "$INPUT" \
        --model "$MODEL" \
        --batch_size "$BATCH_SIZE" \
        --output "$OUTPUT" \
        --test_run
else
    python src/make_preds.py --input "$INPUT" \
        --model "$MODEL" \
        --batch_size "$BATCH_SIZE" \
        --output "$OUTPUT"
fi
