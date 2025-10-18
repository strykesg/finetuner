# Vast.ai Quickstart for Training the Finetuner Financial Model

Use these steps to locate your active Vast.ai instance, connect to it, clone the
Finetuner repository, and run the updated financial training pipeline that
produces a Q4_K_M quantised GGUF from `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`.

## 1. Check Vast.ai for Active Instances

```bash
# Authenticate (requires Vast.ai API key in your environment)
vastai login

# List running instances with IDs, public IPs, and SSH ports
vastai show instances --status running
```

Note the instance `id`, `ssh_host`, and `ssh_port` values for the machine you
plan to use.

## 2. SSH Into the GPU Server

```bash
# Replace ID, HOST, and PORT with the values from the previous step
ssh -p PORT root@HOST
```

If Vast.ai supplies an `ssh` command via `vastai show instance <ID>`, you can
directly run that instead.

## 3. Prepare the Workspace

On the remote shell:

```bash
# (optional) ensure git is available
apt-get update && apt-get install -y git  # only if git is missing

# Clone the Finetuner repository
cd /root
rm -rf finetuner  # optional cleanup if a previous copy exists
git clone https://github.com/strykesg/finetuner.git
cd finetuner

# (optional) create/activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. (Optional) Set Up llama.cpp for Quantisation

The financial pipeline can export GGUF artefacts and quantise them to `Q4_K_M` if
`llama.cpp` is available. If you do not need quantisation, pass
`--skip_quantization` later and skip this section.

```bash
# Clone llama.cpp once per machine
cd /root
if [ ! -d llama.cpp ]; then
  git clone https://github.com/ggerganov/llama.cpp.git
  cd llama.cpp
  cmake -B build -S .
  cmake --build build -j $(nproc)
fi
```

## 5. Launch the Financial Training Pipeline

Return to the Finetuner repository and run the automated workflow. The command
below performs dataset conversion, supervised fine-tuning, adapter merging, and
GGUF export (FP16 + Q4_K_M).

```bash
cd /root/finetuner
python train_financial_model.py --llama_cpp_path /root/llama.cpp
```

Key details:
- Base model: `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`
- Outputs:
  - `financial_lora/` (LoRA adapters + tokenizer)
  - `merged_financial_fp16/` (merged FP16 HF checkpoint)
  - `financial_model_fp16.gguf`
  - `financial_model_q4_k_m.gguf`
- Dataset conversion automatically processes `random/` and downloads the
  Investopedia corpus into `datasets/`.

## 6. Customisation & Recovery

- Skip stages with `--skip_dataset_prep`, `--skip_training`, `--skip_merge`, or
  `--skip_quantization` when re-running only part of the pipeline.
- Override artefact locations, hyperparameters, or llama.cpp path, e.g.:
  ```bash
  python train_financial_model.py \
    --llama_cpp_path /root/llama.cpp \
    --learning_rate 8e-6 \
    --num_train_epochs 5 \
    --gguf_q4_path /root/models/finance_q4.gguf
  ```
- Resume interrupted SFT training directly via `train.py`:
  ```bash
  python train.py \
    --model_name deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
    --dataset_folder datasets/ \
    --output_dir ./financial_lora \
    --resume_from_checkpoint ./financial_lora/checkpoint-latest
  ```

Following these steps results in a quantised financial assistant model ready for
use with llama.cpp or other GGUF-compatible runtimes.
