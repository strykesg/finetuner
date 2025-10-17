#!/bin/bash
# This script runs a very fast test of the training pipeline.

python3 train.py \
  --model_name "unsloth/gemma-2b-it-bnb-4bit" \
  --dataset "datasets/alpaca/sample_alpaca.json" \
  --max_seq_length 128 \
  --output_dir "./test_run"
