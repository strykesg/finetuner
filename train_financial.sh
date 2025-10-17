#!/bin/bash
# This script runs the full training process for the financial model using all available GPUs.

# Use 'accelerate launch' for distributed training, which automatically detects and uses all available GPUs.
accelerate launch train.py --financial_model
