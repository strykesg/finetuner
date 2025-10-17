#!/bin/bash
# This script runs a very fast test of the training pipeline.

python3 train.py \
  # --model_name: Use a very small model for a quick test.
  --model_name "unsloth/gemma-2b-it-bnb-4bit" \

  # --dataset: Use the relative path to the tiny sample dataset.
  --dataset "datasets/alpaca/sample_alpaca.json" \
  --dataset_format "alpaca" \

  # --output_dir: Save to a temporary directory.
  --output_dir "./test_run" \

  # --max_seq_length: Use a very short sequence length for speed.
  --max_seq_length 128 \

  # --num_train_epochs: Only run for a single epoch.
  --num_train_epochs 1 \

  # --per_device_train_batch_size & --gradient_accumulation_steps: Use minimal batch sizes.
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \

  # --logging_steps: Log frequently to see progress.
  --logging_steps 1
