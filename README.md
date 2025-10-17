# Finetuner: Universal LLM Fine-Tuning Utility

A comprehensive toolkit for fine-tuning Large Language Models (LLMs) with specialized focus on financial and trading domains using Unsloth's optimized framework.

## 🚀 Quick Start

Train a financial assistant model in 5 minutes:

```bash
git clone https://github.com/strykesg/finetuner.git
cd finetuner
pip install -r requirements.txt
python train_financial_model.py
```

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

Train the ServiceNow/Apriel-Nemotron-15B-Thinker model on financial datasets:

```bash
python train_financial_model.py
```

This single command will:
1. Download required datasets
2. Convert and organize data
3. Configure optimal training parameters
4. Train the model for 3 epochs
5. Save the fine-tuned model

### Training Parameters

- **Model**: ServiceNow/Apriel-Nemotron-15B-Thinker (Q6_K quantized)
- **Datasets**: Combined SFT, DPO, and Investopedia datasets (~22,500 examples)
- **Batch Size**: 1 with gradient accumulation (effective batch size: 8)
- **Learning Rate**: 2e-5
- **LoRA Rank**: 32 (higher for complex financial reasoning)
- **Context Length**: 4096 tokens
- **Training Time**: 4-8 hours on modern GPU

### Custom Training

```bash
# Train with custom parameters
python train_financial_model.py \
  --output_dir ./my_finance_model \
  --extra_args \
  --learning_rate 1e-5 \
  --num_train_epochs 5 \
  --max_seq_length 8192

# Resume interrupted training
python train.py \
  --model_name ServiceNow/Apriel-Nemotron-15B-Thinker \
  --dataset_folder datasets/ \
  --output_dir ./financial_model_output \
  --resume_from_checkpoint ./financial_model_output/checkpoint-latest
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
  --extra_args \
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
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

# Load fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="./financial_model_output",
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)

# Enable faster inference
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
    model_name="ServiceNow/Apriel-Nemotron-15B-Thinker",
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)[0]

# Merge adapters
model = PeftModel.from_pretrained(base_model, "./financial_model_output")
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
python train_financial_model.py --extra_args --per_device_train_batch_size 1 --gradient_accumulation_steps 16
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
- ServiceNow for the Nemotron model

---

## 📞 Support

For issues and questions:
- Open a GitHub issue
- Check the troubleshooting section
- Review the training logs

**Happy fine-tuning! 🚀**
