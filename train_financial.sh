#!/bin/bash
# Convenience wrapper for the end-to-end financial training pipeline.
# Additional arguments are forwarded to train_financial_model.py so you can
# specify paths like --llama_cpp_path or override hyperparameters.

python train_financial_model.py "$@"
