# Financial Model Training Guide

This guide covers the end-to-end pipeline implemented in `train_financial_model.py`
for fine-tuning **DSR1/DSR1-Distill-Qwen-14B** on curated financial datasets and
exporting a production-ready `Q4_K_M` GGUF.

## Overview

The financial workflow automates four stages:

1. Convert local datasets and ingest the FinLang Investopedia corpus
2. Fine-tune the base model with LoRA adapters using Unsloth
3. Merge adapters back into a full-precision Hugging Face checkpoint
4. Convert to GGUF and quantise to `Q4_K_M` for lightweight deployment

Outputs are written to:

- `financial_lora/` – LoRA adapters and tokenizer (resumable training)
- `merged_financial_fp16/` – Full FP16 Hugging Face model
- `financial_model_fp16.gguf` – FP16 GGUF export
- `financial_model_q4_k_m.gguf` – Quantised GGUF ready for inference

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline (expects llama.cpp checked out at ./llama.cpp)
python train_financial_model.py --llama_cpp_path ./llama.cpp
```

Pass `--skip_quantization` if you only need LoRA adapters or a merged HF
checkpoint.

## Model & Training Details

- **Base model**: `DSR1/DSR1-Distill-Qwen-14B`
- **Training precision**: 4-bit loading via Unsloth with LoRA (r=32, alpha=64)
- **Datasets**: Trader SFT, converted Trader DPO, FinLang Investopedia (~22.5k samples)
- **Sequence length**: 4096 tokens
- **Effective batch size**: 8 (batch 1, grad accumulation 8)
- **Learning rate**: 1e-5 (cosine schedule handled by TRL SFT trainer)
- **Quantisation**: `Q4_K_M` via `llama.cpp` after merging

Training typically requires a 24GB+ GPU and ~50GB of free disk space for all
artifacts.

## Customisation

The script exposes knobs for every stage. Common examples:

```bash
# Change output locations and quantised filename
python train_financial_model.py \
  --lora_output_dir ./outputs/lora \
  --merged_model_dir ./outputs/merged_fp16 \
  --gguf_q4_path ./outputs/finance_q4.gguf

# Adjust hyperparameters
python train_financial_model.py \
  --learning_rate 8e-6 \
  --num_train_epochs 5 \
  --max_seq_length 6144 \
  --gradient_accumulation_steps 12

# Resume raw training manually (advanced)
python train.py \
  --model_name DSR1/DSR1-Distill-Qwen-14B \
  --dataset_folder datasets/ \
  --output_dir ./financial_lora \
  --resume_from_checkpoint ./financial_lora/checkpoint-latest
```

Use `--skip_dataset_prep`, `--skip_training`, or `--skip_merge` to rerun only
specific stages.

## Using the Outputs

### 1. Continue Training with LoRA

```python
from unsloth import FastLanguageModel
from peft import PeftModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="DSR1/DSR1-Distill-Qwen-14B",
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)
model = PeftModel.from_pretrained(model, "./financial_lora")
```

### 2. Load the Merged Hugging Face Checkpoint

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "./merged_financial_fp16",
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("./merged_financial_fp16")
```

### 3. Run the Quantised GGUF with llama.cpp

```bash
./llama.cpp/build/bin/llama-cli \
  -m financial_model_q4_k_m.gguf \
  -p "Explain risk parity to an intermediate investor." \
  -n 128 --temp 0.7 -ngl 99
```

## Troubleshooting

- **CUDA OOM**: Reduce `--max_seq_length` or increase `--gradient_accumulation_steps`
- **Convert script missing**: Ensure `--llama_cpp_path` points to a llama.cpp checkout with `convert_hf_to_gguf.py`
- **llama-quantize not found**: Build llama.cpp (`cmake -B build -S . && cmake --build build`)
- **Dataset download failures**: Re-run with a stable connection; datasets are cached under `~/.cache/huggingface`

## Contribution Notes

When extending the pipeline:

- Keep defaults aligned with the scripts in this repository
- Document new arguments in this guide and `README.md`
- Provide fallbacks (`--skip_*`) so stages remain resumable

Happy quantising!
