#!/bin/bash
# POST_TRAINING_SUMMARY.md - Complete Guide for Finetuner Financial Model

# Overview
This summary covers the entire workflow for your fine-tuned Llama-3.1-8B model specialized for financial/trading tasks. Training was completed on Vast.ai instance ID 26919705 (RTX 5090) using Unsloth on combined datasets (~22,500 examples: SFT, DPO, Investopedia). The model is optimized for inference and ready for production or further training.

## Key Models and Files (On Vast.ai Server: /root/finetuner/)
- **LoRA Adapters**: `./lora_model/` (~1.2 GB) - Lightweight fine-tune weights. Use for re-training.
- **Merged Full Model (FP16)**: `./merged_lora_fp16/` (~16 GB) - Full merged weights for Hugging Face/Transformers.
- **GGUF FP16**: `./financial_model.gguf` (~16.1 GB) - Unquantized GGUF for max quality.
- **GGUF Q8_0 (Prod Recommended)**: `./financial_model_q8_0.gguf` (~8.1 GB) - 8-bit quantized for efficient CPU/GPU inference.
- **Logs**: `./training.log` - Training details (steps, losses).
- **Docs**: `./README.md` - Testing commands, deployment tips.

Total project size: ~40 GB. Backup via B2 or SCP (e.g., `scp -P 20511 -r root@116.127.115.18:~/finetuner/financial_model_q8_0.gguf ./models/`).

## Deploying on Production Server (No GPU: 8-Core CPU, 24GB RAM)
Use llama.cpp for CPU-only inference (no CUDA needed; runs on CPU cores). Download the Q8_0 GGUF first.

### Setup
1. **Download Model**: From local or server:
   ```
   mkdir models
   scp -P 20511 root@116.127.115.18:~/finetuner/financial_model_q8_0.gguf ./models/  # If not local yet
   ```
2. **Install llama.cpp** (one-time):
   ```
   git clone https://github.com/ggerganov/llama.cpp
   cd llama.cpp
   make -j8  # CPU build (~5 min on 8 cores)
   ```

### Quick Query (CLI)
From your prod server's terminal (e.g., Ubuntu with 8 cores/24GB RAM):
```
cd llama.cpp/build/bin
./llama-cli -m ../../models/financial_model_q8_0.gguf \
  -p "As a financial advisor, explain risk management." \
  -n 200 --temp 0.7
```
- **Expected**: Generates response (e.g., "Risk management in trading involves strategies like stop-losses...").
- **Performance**: ~5-15 tokens/sec on 8-core CPU (24GB RAM sufficient; loads ~8GB into memory).
- **Interactive Mode**: Add `-i` for chat: `./llama-cli -m ... -i`.
- **Batch Queries**: Script for multiple prompts (save as query.sh):
  ```
  #!/bin/bash
  cd llama.cpp/build/bin
  PROMPTS=("Prompt 1" "Prompt 2")
  for p in "${PROMPTS[@]}"; do
    ./llama-cli -m ../../models/financial_model_q8_0.gguf -p "$p" -n 150 --temp 0.7
  done
  ```

### Production Server Mode (API/REST)
For API access (e.g., web app integration):
```
cd llama.cpp/build/bin
./llama-server -m ../../models/financial_model_q8_0.gguf \
  -c 4096 --host 0.0.0.0 --port 8080
```
- **Query via curl** (from anywhere):
  ```
  curl http://your-server-ip:8080/completion -H "Content-Type: application/json" \
    -d '{"prompt": "Financial query", "n_predict": 150, "temperature": 0.7}'
  ```
- **RAM Usage**: ~9 GB peak (fits 24GB easily). Add `--threads 8` for all cores.
- **Security**: Bind to localhost (`--host 127.0.0.1`) for internal; use nginx reverse proxy for external.
- **Uptime**: Run in nohup: `nohup ./llama-server ... > server.log 2>&1 &`. Monitor with `ps aux | grep llama-server`.

### Optimization for CPU Server
- **Memory**: 24GB is ample; model uses ~8-10 GB. Reduce context with `-c 2048` if tight.
- **Speed**: Expect 10-20 t/s on 8 cores. For faster, quantize further to Q4_K_M (~4GB): `./llama-quantize financial_model.gguf model_q4.gguf Q4_K_M`.
- **Monitoring**: Use `htop` (CPU usage) and `free -h` (RAM). If OOM, lower threads (`--threads 4`).
- **Docker**: For containerized prod: Use official llama.cpp Docker image, mount the GGUF.

## Re-Training or Adding More Training Data
To continue fine-tuning (e.g., add new financial datasets), use Unsloth on a GPU machine (Vast.ai or local with CUDA).

### Re-Train on New Data
1. **Setup Environment** (GPU recommended):
   ```
   pip install unsloth[colab-new] @ 2024.10+
   git clone your-finetuner-repo  # If extending the repo
   cd finetuner
   ```
2. **Load Existing LoRA**:
   - Use your `./lora_model/` adapters.
   - Example script (`retrain.py`):
     ```
     from unsloth import FastLanguageModel
     from datasets import load_dataset
     from trl import SFTTrainer

     # Load base + your LoRA
     model, tokenizer = FastLanguageModel.from_pretrained(
         model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
         adapter_name = "./lora_model",  # Your adapters
         max_seq_length = 4096,
         load_in_4bit = True,
     )

     # Load new dataset (e.g., new JSONL)
     new_data = load_dataset("json", data_files="new_financial_data.jsonl", split="train")

     # Train (same as original but load LoRA)
     trainer = SFTTrainer(
         model = model,
         train_dataset = new_data,
         args = ...  # Same hyperparams: epochs=1, lr=1e-5
     )
     trainer.train()
     trainer.save_model("./new_lora_model")
     ```
3. **Add Training Data**:
   - Place new JSONL/JSON files in `./datasets/new_data/` (Alpaca/ShareGPT format).
   - Run: `python train.py --dataset_folder ./datasets/ --model_name unsloth/Meta-Llama-3.1-8B-bnb-4bit --output_dir ./new_lora_model --resume_from_checkpoint ./lora_model`.
   - Hyperparams: Reduce LR to 1e-5 for continued training to avoid overwriting.

### Full Re-Train from Scratch
- Use original `train.py --financial_model` but with updated datasets.
- Add data to `./datasets/` (e.g., `./datasets/random/new_data.jsonl` for auto-conversion).
- Run: `python train.py --financial_model --output_dir ./retrained_model`.

### Reminders
- **Backups**: Sync to B2: `b2 sync ./models/ b2://your-bucket/finetuner/`.
- **Git Repo**: Commit changes: `git add . && git commit -m "Update models" && git push`.
- **Server Cleanup**: Free space: `ssh -p 20511 root@116.127.115.18 "cd finetuner && rm -rf merged_lora_fp16 financial_model.gguf"`.
- **Troubleshooting**:
  - GPU OOM: Lower batch size/LR.
  - Model Load Error: Check paths/sizes (`ls -lh *.gguf`).
  - NaN Losses: Normal in LoRA; monitor with `--logging_steps 10`.
  - License: Unsloth is free; base Llama-3.1 under Meta's license.

Generated Oct 18, 2025 via AI assistant.
