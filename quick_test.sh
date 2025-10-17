#!/bin/bash
# This script runs a very fast test of the training pipeline.

python3 train.py \
  --model_name "unsloth/gemma-2b-it-bnb-4bit" \
  --dataset "datasets/alpaca/sample_alpaca.json" \
  --dataset_format "alpaca" \
  --output_dir "./test_run" \
  --max_seq_length 128 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --logging_steps 1