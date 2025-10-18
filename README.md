# Finetuner: Universal LLM Fine-Tuning Utility

A comprehensive toolkit for fine-tuning Large Language Models (LLMs) with specialized focus on financial and trading domains using Unsloth's optimized framework.

## 🚀 Quick Start

### Option 1: Local Training (Recommended)
Train a financial assistant model in 5 minutes:

```bash
git clone https://github.com/strykesg/finetuner.git
cd finetuner
pip install -r requirements.txt
python train_financial_model.py
```

### Option 2: Google Colab Training
For users without local GPU access:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/strykesg/finetuner/blob/main/finetune_colab.ipynb)

**Colab Requirements:**
- Colab Pro/Pro+ subscription
- A100 GPU (recommended)
- High RAM runtime
- 8-24 hours training time

## 📋 Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Advanced Usage](#advanced-usage)
- [Model Capabilities](#model-capabilities)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## ✨ Features

- **Universal Fine-Tuning**: Support for Alpaca, ShareGPT, and custom formats
- **Multi-Dataset Training**: Combine multiple datasets seamlessly
- **Financial Specialization**: Pre-configured for financial/trading domains
- **Memory Efficient**: Unsloth optimization for consumer GPUs
- **Production Ready**: Complete training pipelines with best practices
- **Automated Dataset Conversion**: Smart format detection and conversion

## 🔧 Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with 24GB+ VRAM (recommended: RTX 3090/4090 or A100)
- **RAM**: 64GB+ system RAM
- **Storage**: 100GB+ free disk space
- **OS**: Linux (Ubuntu 20.04+ recommended)

### Software Requirements
- Python 3.8+
- CUDA 11.8+ (for GPU acceleration)
- Git

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/strykesg/finetuner.git
cd finetuner
```

### 2. Create Virtual Environment (Recommended)

```bash
python3 -m venv finetuner_env
source finetuner_env/bin/activate  # On Windows: finetuner_env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python3 -c "from unsloth import FastLanguageModel; print('Unsloth available')"
```

## 📊 Dataset Preparation

### Automatic Dataset Setup

The training script automatically handles dataset preparation:

```bash
# Convert existing datasets in random/ folder
python convert_datasets.py

# The train_financial_model.py script will automatically:
# - Convert DPO format datasets to Alpaca format
# - Download and convert FinLang Investopedia dataset
# - Organize all datasets for training
```

### Dataset Structure

```
datasets/
├── alpaca/          # Alpaca format datasets
│   ├── converted_trader_dpo_data.jsonl
│   └── sample_alpaca.json
├── sharegpt/        # ShareGPT format datasets
│   ├── processed_trader_sft_data.jsonl
│   └── sample_sharegpt.json
└── investopedia/    # Auto-downloaded Investopedia data
    ├── investopedia_train.jsonl
    └── investopedia_test.jsonl
```

### Supported Formats

- **Alpaca**: `{"instruction": "...", "input": "...", "output": "..."}`
- **ShareGPT**: `{"conversations": [{"from": "human", "value": "..."}, {"from": "assistant", "value": "..."}]}`
- **DPO**: Automatically converted to Alpaca format

## 🎯 Training

### Financial Model Training

Train and quantize the `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct` base model into a ready-to-serve
financial assistant:

```bash
python train_financial_model.py --llama_cpp_path ./llama.cpp
```

This pipeline performs the full workflow:
1. Convert and organise datasets (including Investopedia ingestion)
2. Run supervised LoRA fine-tuning for three epochs
3. Merge the adapters back into a full-precision model
4. Export an FP16 GGUF artifact
5. Quantise to `Q4_K_M` for efficient deployment

### Training Parameters

- **Base Model**: deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct (4-bit training via Unsloth)
- **Datasets**: SFT + converted DPO + Investopedia (~22.5k examples)
- **Batch Size**: 1 (effective 8 via gradient accumulation)
- **Learning Rate**: 1e-5 (tuned for 14B distilled base)
- **LoRA Rank**: 32 with alpha 64 for richer adaptation
- **Context Length**: 4096 tokens
- **Outputs**: LoRA adapters, merged FP16 model, FP16 GGUF, Q4_K_M GGUF

### Custom Training

```bash
# Override hyper-parameters or artefact locations
python train_financial_model.py \
  --learning_rate 8e-6 \
  --num_train_epochs 5 \
  --max_seq_length 6144 \
  --gguf_q4_path ./models/finance_q4.gguf

# Skip quantisation if you only need LoRA adapters
python train_financial_model.py --skip_quantization

# Resume raw training manually (advanced)
python train.py \
  --model_name deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
  --dataset_folder datasets/ \
  --output_dir ./financial_lora \
  --resume_from_checkpoint ./financial_lora/checkpoint-latest
```

### General Fine-Tuning

For other models and datasets:

```bash
# Single dataset from Hugging Face
python train.py --dataset "yahma/alpaca-cleaned"

# Multiple datasets from folder
python train.py --dataset_folder ./datasets/

# Custom configuration
python train.py \
  --model_name "unsloth/Meta-Llama-3.1-8B-bnb-4bit" \
  --dataset "your-dataset" \
  --output_dir ./custom_model \
  --learning_rate 2e-4 \
  --num_train_epochs 3
```

## 🔄 Advanced Usage

### Adding Custom Datasets

1. Place your datasets in the `random/` folder
2. Run conversion: `python convert_datasets.py`
3. Datasets will be automatically formatted and organized
4. Re-run training to include new data

### Memory Optimization

For GPUs with limited VRAM:

```bash
python train_financial_model.py \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --max_seq_length 2048
```

### Distributed Training

For multi-GPU setups:

```bash
# Train across multiple GPUs
torchrun --nproc_per_node=2 train.py \
  --model_name "unsloth/Meta-Llama-3.1-8B-bnb-4bit" \
  --dataset_folder datasets/ \
  --output_dir ./distributed_model
```

## 🧠 Model Capabilities

After training, your financial model will excel at:

### Trading & Investment
- Market trend analysis and forecasting
- Position sizing and risk management
- Entry/exit strategy development
- Portfolio optimization

### Financial Education
- Investment concept explanations
- Financial terminology clarification
- Market mechanics education
- Risk assessment guidance

### Research & Analysis
- Company fundamental analysis
- Sector trend identification
- Economic indicator interpretation
- Valuation methodology

### Professional Assistance
- Trading journal analysis
- Strategy backtesting insights
- Risk management frameworks
- Performance evaluation

## 🔧 Integration & Deployment

### Load Trained Model

```python
from unsloth import FastLanguageModel
from peft import PeftModel

# Load base + adapters for inference
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)
model = PeftModel.from_pretrained(model, "./financial_lora")
FastLanguageModel.for_inference(model)
```

### Example Usage

```python
messages = [
    {"role": "system", "content": "You are an expert financial trading assistant."},
    {"role": "user", "content": "What's the best strategy for pyramiding into a winning position?"}
]

inputs = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Production Deployment

```python
# Merge LoRA adapters for faster inference
from peft import PeftModel

# Load base model
base_model = FastLanguageModel.from_pretrained(
    model_name="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)[0]

# Merge adapters
model = PeftModel.from_pretrained(base_model, "./financial_lora")
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./merged_financial_model")
tokenizer.save_pretrained("./merged_financial_model")
```

## 🐛 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce memory usage
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
python train_financial_model.py --per_device_train_batch_size 1 --gradient_accumulation_steps 16
```

#### Import Errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Dataset Download Issues
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/datasets/
python train_financial_model.py
```

#### Model Loading Issues
```bash
# Check available memory
nvidia-smi
# Use smaller model if needed
python train.py --model_name "unsloth/Mistral-7B-Instruct-v0.3-bnb-4bit" --dataset_folder datasets/
```

#### Colab-Specific Issues
```bash
# Memory issues in Colab
# Use the Colab notebook with optimized settings
# Reduce max_seq_length to 1024-2048
# Use gradient_accumulation_steps=8 for smaller GPUs

# Time limit issues
# Save checkpoints every 50 steps
# Use resumable training
# Consider Colab Pro+ for longer sessions
```

### Performance Optimization

#### Training Speed
- Use SSD storage for datasets
- Enable mixed precision (default)
- Use gradient checkpointing (default)

#### Inference Speed
- Merge LoRA adapters
- Use 4-bit quantization
- Enable KV-cache

### Getting Help

1. Check the logs for detailed error messages
2. Verify hardware meets requirements
3. Test with smaller datasets first
4. Check GitHub issues for similar problems

## 🤝 Contributing

### Adding New Datasets

1. Fork the repository
2. Add datasets to `random/` folder
3. Update conversion logic if needed
4. Test with `python convert_datasets.py`
5. Submit a pull request

### Improving Training

1. Modify hyperparameters in `train_financial_model.py`
2. Add new model configurations
3. Enhance dataset preprocessing
4. Improve error handling

### Code Style

- Follow PEP 8 guidelines
- Add type hints to functions
- Include comprehensive docstrings
- Write tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for optimized training
- [Hugging Face](https://huggingface.co) for datasets and transformers
- [FinLang](https://huggingface.co/FinLang) for financial datasets
- [DeepSeek](https://huggingface.co/deepseek-ai) for the Coder V2 Lite Instruct base

---

## 📞 Support

For issues and questions:
- Open a GitHub issue
- Check the troubleshooting section
- Review the training logs

**Happy fine-tuning! 🚀**
