#!/bin/bash

python src/prepare_data.py
python src/train_model.py
python src/make_preds.py
