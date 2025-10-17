#!/bin/bash
# This script runs the full training process for the financial model using all available GPUs.

# Use torchrun to launch distributed training across 8 GPUs.
# --nproc_per_node=8 tells torchrun to use 8 GPUs.
torchrun --nproc_per_node=8 train.py --financial_model