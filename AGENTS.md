# AGENTS.md - Finetuner Project

## Build/Lint/Test Commands
- **Install dependencies**: `pip install -r requirements.txt`
- **Build**: `python -m pip install -e .`
- **Test all**: `python -m pytest`
- **Test single**: `python -m pytest tests/test_file.py::TestClass::test_method -v`
- **Lint**: `flake8 .`
- **Format**: `black .`
- **Convert datasets**: `python convert_datasets.py`
- **Run fine-tuning**: `python train.py --dataset "yahma/alpaca-cleaned"`
- **Run multi-dataset training**: `python train.py --dataset_folder datasets/`
- **Train financial model**: `python train_financial_model.py`

## Architecture & Codebase Structure
- **train.py**: Main command-line utility for fine-tuning LLMs using Unsloth
- **requirements.txt**: Python dependencies for the fine-tuning pipeline
- **Data formats supported**: Alpaca (instruction-response pairs) and ShareGPT (conversational)
- **Model sources**: Hugging Face Hub repositories or local files
- **Dataset folder structure**: Organized folder system for multiple datasets (see train.py docstring)
- **Output**: LoRA adapters and tokenizer saved to specified directory

## Dataset Folder Structure
The script supports loading multiple datasets from a folder structure:

```
datasets/
├── alpaca/          # All files here are treated as alpaca format
│   ├── dataset1.json
│   └── dataset2.jsonl
├── sharegpt/        # All files here are treated as sharegpt format
│   ├── conversations.json
│   └── chat_data.jsonl
└── mixed/           # Auto-detect format per file
    ├── alpaca_style.json
    └── sharegpt_style.jsonl
```

### Format Auto-Detection
- **Alpaca format**: Files with 'instruction', 'input', 'output' fields
- **ShareGPT format**: Files with 'conversations' field containing turn lists
- **Supported file types**: .json, .jsonl

### Dataset Conversion
The `convert_datasets.py` script automatically organizes datasets from the `random/` folder:

- **DPO format** datasets → Converted to Alpaca format and moved to `datasets/alpaca/`
- **ShareGPT format** datasets → Standardized and moved to `datasets/sharegpt/`

Run `python convert_datasets.py` to process datasets in the `random/` folder.

## Financial Model Training
The `train_financial_model.py` script provides a comprehensive training setup for financial domain models:

- **Model**: ServiceNow/Apriel-Nemotron-15B-Thinker (Q6_K GGUF quantized)
- **Datasets**: Combined SFT and DPO datasets, plus FinLang Investopedia dataset
- **Optimization**: Memory-efficient training with domain-specific hyperparameters
- **Output**: Production-ready financial assistant model

### Usage Examples
```bash
# Single dataset from Hub
python train.py --dataset "yahma/alpaca-cleaned"

# Convert and organize datasets from random/ folder
python convert_datasets.py

# Multiple datasets from organized folder
python train.py --dataset_folder "./datasets/"

# Train comprehensive financial model
python train_financial_model.py

# Custom financial model training
python train_financial_model.py --output_dir ./my_finance_model --extra_args --learning_rate 1e-5
```


## Code Style Guidelines
- **Imports**: Absolute imports only, grouped by stdlib, third-party, local
- **Naming**: snake_case for functions/variables, CamelCase for classes
- **Types**: Use type hints for all function parameters and return values
- **Error handling**: Use specific exceptions, log errors with context
- **Docstrings**: Google-style docstrings for all public functions/classes
