# Financial Model Training Guide

This guide explains how to use the `train_financial_model.py` script to train a specialized financial assistant model.

## Overview

The financial training script provides an end-to-end solution for training the **ServiceNow/Apriel-Nemotron-15B-Thinker** model on comprehensive financial datasets, including:

- **Trader SFT Data**: Conversational trading assistant interactions
- **Trader DPO Data**: Preference learning data (converted to SFT format)
- **Investopedia Dataset**: Financial education and instruction-tuning data
- **Combined Knowledge**: Multi-domain financial expertise

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run financial model training
python train_financial_model.py
```

## Model Details

- **Base Model**: ServiceNow/Apriel-Nemotron-15B-Thinker
- **Quantization**: Q6_K GGUF (6-bit quantization for optimal performance)
- **Architecture**: Optimized for financial reasoning and analysis
- **Context Length**: 4096 tokens (extended for complex financial analysis)

## Training Configuration

### Optimized Hyperparameters

```python
# Memory-efficient configuration for large models
per_device_train_batch_size: 1
gradient_accumulation_steps: 8  # Effective batch size: 8
learning_rate: 2e-5
max_seq_length: 4096

# LoRA configuration for financial tasks
lora_r: 32  # Higher rank for complex reasoning
lora_alpha: 64  # Balanced scaling
target_modules: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

### Dataset Composition

1. **SFT Datasets** (Alpaca format):
   - Trading conversations and analysis
   - Financial Q&A pairs
   - Investment strategy discussions

2. **Investopedia Integration**:
   - Financial education content
   - Investment terminology
   - Market analysis explanations

## Advanced Usage

### Custom Output Directory

```bash
python train_financial_model.py --output_dir ./my_finance_model
```

### Custom Training Parameters

```bash
python train_financial_model.py \
  --extra_args \
  --learning_rate 1e-5 \
  --num_train_epochs 5 \
  --lora_r 64 \
  --max_seq_length 8192
```

### Resume Training

If training is interrupted, you can resume from the latest checkpoint:

```bash
python train.py \
  --model_name ServiceNow/Apriel-Nemotron-15B-Thinker \
  --dataset_folder datasets/ \
  --output_dir ./financial_model_output \
  --resume_from_checkpoint ./financial_model_output/checkpoint-latest
```

## Expected Training Time

- **Hardware**: Single GPU with 24GB+ VRAM recommended
- **Duration**: 4-8 hours depending on dataset size and hardware
- **Memory Usage**: ~16-20GB VRAM during training
- **Disk Space**: ~50GB for model and datasets

## Output Structure

```
financial_model_output/
├── adapter_model.bin          # Trained LoRA adapters
├── adapter_config.json        # LoRA configuration
├── tokenizer.json             # Updated tokenizer
├── training_args.bin          # Training configuration
├── trainer_state.json         # Training progress
└── checkpoint-*/             # Intermediate checkpoints
```

## Model Capabilities

After training, the model will be specialized for:

- **Trading Analysis**: Market trend analysis, entry/exit strategies
- **Financial Education**: Investment concepts, terminology explanation
- **Risk Assessment**: Portfolio risk evaluation, position sizing
- **Market Research**: Company analysis, sector trends
- **Investment Strategy**: Diversification advice, asset allocation

## Integration

### Load the Trained Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
model = AutoModelForCausalLM.from_pretrained(
    "ServiceNow/Apriel-Nemotron-15B-Thinker",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load fine-tuned adapters
model = PeftModel.from_pretrained(model, "./financial_model_output")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("./financial_model_output")
```

### Example Usage

```python
messages = [
    {"role": "system", "content": "You are a financial trading assistant with deep market expertise."},
    {"role": "user", "content": "What's the best strategy for pyramiding into a winning position in tech stocks?"}
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
outputs = model.generate(inputs, max_new_tokens=512)
response = tokenizer.decode(outputs[0])
```

## Troubleshooting

### Memory Issues
- Reduce `per_device_train_batch_size` to 1
- Increase `gradient_accumulation_steps`
- Use `max_seq_length` of 2048 instead of 4096

### Dataset Issues
- Run `python convert_datasets.py` to ensure proper formatting
- Check that datasets are in the correct folders
- Verify JSON format is valid

### Model Loading Issues
- Ensure sufficient VRAM (24GB+ recommended)
- Use quantization if needed
- Check model name spelling

## Performance Optimization

For production deployment:

1. **Quantization**: Use GGUF format for inference
2. **Model Merging**: Merge LoRA adapters with base model
3. **Caching**: Enable KV-cache for faster inference
4. **Batch Processing**: Implement batch inference for multiple queries

## Contributing

To add new financial datasets:

1. Place datasets in `random/` folder
2. Run `python convert_datasets.py`
3. They will be automatically formatted and organized
4. Re-run training to include new data

## License

This training setup is provided for educational and research purposes. Ensure compliance with dataset licenses and model usage terms.
