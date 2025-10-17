#!/bin/bash
python3 train.py \
  --dataset "/Users/bradleymutemi/Documents/finetuner/datasets/alpaca/sample_alpaca.json" \
  --dataset_format "alpaca" \
  --model_name "unsloth/Meta-Llama-3.1-8B-bnb-4bit" \
  --output_dir "./test_run" \
  --max_seq_length 128 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --logging_steps 1
