#!/usr/bin/env python3
"""
Universal Command-Line Fine-Tuning Utility for Language Models

This script provides a zero-friction command-line interface for fine-tuning large language models
using Unsloth's optimized framework. It supports both Alpaca and ShareGPT data formats from
Hugging Face Hub repositories, local files, or organized dataset folders.

DATASET FOLDER STRUCTURE:
Place your datasets in a folder with the following structure:
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

Supported formats:
- Alpaca: JSON/JSONL with 'instruction', 'input', 'output' fields
- ShareGPT: JSON/JSONL with 'conversations' field containing turn lists

The script is designed for maximum performance on single-GPU hardware through Unsloth's
custom Triton kernels, manual backpropagation, and 4-bit quantization via QLoRA.

Example Usage Scenarios:

# Example 1: Fine-tuning Llama-3-8B on the Alpaca dataset from the Hub.
python train.py --dataset "yahma/alpaca-cleaned"

# Example 2: Fine-tuning Mistral-7B on a local ShareGPT-formatted file.
python train.py \
  --model_name "unsloth/mistral-7b-instruct-v0.3-bnb-4bit" \
  --dataset "./data/my_conversations.jsonl" \
  --dataset_format "sharegpt" \
  --output_dir "./mistral_finetuned_lora" \
  --learning_rate 2e-5 \
  --num_train_epochs 3

# Example 3: Fine-tune on multiple datasets from a folder (auto-detect formats).
python train.py \
  --dataset_folder "./datasets/" \
  --model_name "unsloth/gemma-2-9b-it-bnb-4bit" \
  --output_dir "./multi_dataset_lora" \
  --max_seq_length 4096 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --lora_r 32 \
  --lora_alpha 32

# Example 4: Comprehensive fine-tune of Gemma-9B on the UltraChat dataset.
python train.py \
  --model_name "unsloth/gemma-2-9b-it-bnb-4bit" \
  --dataset "HuggingFaceH4/ultrachat_200k" \
  --dataset_format "sharegpt" \
  --dataset_split "train_sft" \
  --output_dir "./gemma_ultrachat_lora" \
  --max_seq_length 4096 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --lora_r 32 \
  --lora_alpha 32
"""

import argparse
import os
from typing import Any, Dict

import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the fine-tuning utility.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Universal Command-Line Fine-Tuning Utility for Language Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --dataset "yahma/alpaca-cleaned"
  python train.py --model_name "unsloth/mistral-7b-instruct-v0.3-bnb-4bit" --dataset "./data.jsonl" --dataset_format "sharegpt"
        """
    )

    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        help="Base model to load from Hugging Face Hub. Strongly recommend using Unsloth's pre-quantized models (ending in -bnb-4bit) for optimal performance and accuracy."
    )

    # Dataset configuration (mutually exclusive group)
    dataset_group = parser.add_mutually_exclusive_group(required=True)

    dataset_group.add_argument(
        "--dataset",
        type=str,
        help="Dataset identifier. Can be either a Hugging Face Hub repository name (e.g., 'yahma/alpaca-cleaned') or a path to a local data file (e.g., './data/my_data.jsonl')."
    )

    dataset_group.add_argument(
        "--dataset_folder",
        type=str,
        help="Path to a folder containing multiple datasets. The folder can contain subfolders 'alpaca/' and 'sharegpt/' for format-specific datasets, or files will be auto-detected. Example: './datasets/'"
    )

    parser.add_argument(
        "--dataset_format",
        type=str,
        default="alpaca",
        choices=["alpaca", "sharegpt"],
        help="Format of the dataset (ignored when using --dataset_folder). 'alpaca' for instruction-response pairs (instruction, input, output columns). 'sharegpt' for multi-turn conversations."
    )

    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="Which split of the dataset to use for training. Important for Hub datasets with predefined splits."
    )

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./lora_model",
        help="Directory where the final trained LoRA adapters and tokenizer files will be saved."
    )

    # Model parameters
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum context length for the model during training. Unsloth enables up to 4x longer contexts than standard implementations."
    )

    # Training hyperparameters
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of times the model will see the entire training dataset. Recommended: 1-3 epochs to avoid overfitting."
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Number of training examples processed in a single forward/backward pass. For larger effective batch sizes, increase gradient_accumulation_steps instead."
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of steps to accumulate gradients before an optimizer update. Enables larger effective batch sizes without increased memory usage."
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Initial learning rate for the AdamW optimizer. For complex tasks or larger models, consider lower rates (1e-4, 5e-5, or 2e-5)."
    )

    # LoRA configuration
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="Rank of the LoRA update matrices. Controls the number of trainable parameters. 16 offers a good balance between expressiveness and efficiency."
    )

    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="Scaling factor for the LoRA weights. Commonly set equal to lora_r or 2*lora_r."
    )

    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated list of modules to apply LoRA to in the model architecture."
    )

    parser.add_argument(
        "--quantization",
        type=str,
        default="4bit",
        choices=["4bit", "8bit", "none", "Q6_K"],
        help="Quantization method for model loading. '4bit' and '8bit' use Unsloth's built-in support. 'Q6_K' (GGUF) is not natively supported—will fallback to 4bit with warning. 'none' loads full precision."
    )

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=1,
        help="How often to log training metrics (steps)."
    )

    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="How often to save checkpoints (steps)."
    )

    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=3,
        help="Maximum number of checkpoints to keep."
    )

    return parser.parse_args()


def format_alpaca_dataset(dataset: Any) -> Any:
    """
    Format an Alpaca-style dataset into training-ready text format.

    Args:
        dataset: Raw dataset with instruction, input, output columns.

    Returns:
        Dataset with formatted 'text' column.
    """
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

    def format_example(example: Dict[str, Any]) -> Dict[str, Any]:
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")

        # Handle empty input gracefully
        if not input_text:
            input_text = ""

        formatted_text = alpaca_prompt.format(instruction, input_text, output)
        return {"text": formatted_text}

    return dataset.map(format_example)


def format_sharegpt_dataset(dataset: Any, tokenizer: Any) -> Any:
    """
    Format a ShareGPT-style conversational dataset using the tokenizer's chat template.

    Args:
        dataset: Raw dataset with conversations column containing turn lists.
        tokenizer: The model's tokenizer with chat template.

    Returns:
        Dataset with formatted 'text' column.
    """
    def format_example(example: Dict[str, Any]) -> Dict[str, Any]:
        conversations = example.get("conversations", [])

        # Use tokenizer's chat template for proper formatting
        formatted_text = tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=False
        )

        return {"text": formatted_text}

    return dataset.map(format_example)


def detect_dataset_format(dataset_path: str, sample_size: int = 5) -> str:
    """
    Auto-detect the format of a dataset by examining a few sample records.

    Args:
        dataset_path: Path to the dataset file.
        sample_size: Number of records to sample for detection.

    Returns:
        Detected format: 'alpaca' or 'sharegpt'.
    """
    try:
        # Load a small sample to detect format
        file_extension = os.path.splitext(dataset_path)[1].lower()
        if file_extension in ['.json', '.jsonl']:
            data_type = 'json'
        else:
            # Default to json if extension unknown
            data_type = 'json'

        # Load just a few examples for detection
        sample_dataset = load_dataset(data_type, data_files=dataset_path, split=f"train[:{sample_size}]")

        # Check for format indicators
        sample_record = sample_dataset[0]

        # Check for ShareGPT format (has 'conversations' field)
        if 'conversations' in sample_record:
            return 'sharegpt'

        # Check for Alpaca format (has 'instruction', 'input', 'output' fields)
        if all(field in sample_record for field in ['instruction', 'input', 'output']):
            return 'alpaca'

        # Default to alpaca if unclear
        print(f"Warning: Could not clearly detect format for {dataset_path}, defaulting to alpaca")
        return 'alpaca'

    except Exception as e:
        print(f"Warning: Error detecting format for {dataset_path}: {e}, defaulting to alpaca")
        return 'alpaca'


def load_single_dataset(dataset_path: str, dataset_format: str, tokenizer: Any, is_hub_repo: bool = False, split: str = "train") -> Any:
    """
    Load and format a single dataset.

    Args:
        dataset_path: Path to dataset file or Hub repository name.
        dataset_format: Format of the dataset ('alpaca' or 'sharegpt').
        tokenizer: The model's tokenizer.
        is_hub_repo: Whether this is a Hub repository (vs local file).
        split: Dataset split to use (for Hub repos).

    Returns:
        Formatted dataset.
    """
    if is_hub_repo:
        dataset = load_dataset(dataset_path, split=split)
    else:
        # Local file
        file_extension = os.path.splitext(dataset_path)[1].lower()
        if file_extension in ['.json', '.jsonl']:
            data_type = 'json'
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Supported: .json, .jsonl")

        dataset = load_dataset(data_type, data_files=dataset_path)

    # Format according to specified format
    if dataset_format == "alpaca":
        formatted_dataset = format_alpaca_dataset(dataset)
    elif dataset_format == "sharegpt":
        formatted_dataset = format_sharegpt_dataset(dataset, tokenizer)
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_format}")

    return formatted_dataset


def load_and_format_dataset(args: argparse.Namespace, tokenizer: Any) -> Any:
    """
    Load and format dataset(s) from Hugging Face Hub, local file, or dataset folder.

    Args:
        args: Parsed command-line arguments.
        tokenizer: The model's tokenizer.

    Returns:
        Formatted dataset ready for training.
    """
    if args.dataset_folder:
        # Load multiple datasets from folder
        print(f"Loading datasets from folder: {args.dataset_folder}...")

        if not os.path.exists(args.dataset_folder):
            raise ValueError(f"Dataset folder does not exist: {args.dataset_folder}")

        all_datasets = []

        # Scan for dataset files
        for root, dirs, files in os.walk(args.dataset_folder):
            for file in files:
                if file.endswith(('.json', '.jsonl')):
                    file_path = os.path.join(root, file)

                    # Determine format based on folder structure or auto-detection
                    folder_name = os.path.basename(root)

                    if folder_name.lower() == 'alpaca':
                        dataset_format = 'alpaca'
                    elif folder_name.lower() == 'sharegpt':
                        dataset_format = 'sharegpt'
                    else:
                        # Auto-detect format
                        dataset_format = detect_dataset_format(file_path)

                    print(f"Loading {file_path} as {dataset_format} format...")
                    formatted_dataset = load_single_dataset(file_path, dataset_format, tokenizer)
                    all_datasets.append(formatted_dataset)

        if not all_datasets:
            raise ValueError(f"No dataset files found in {args.dataset_folder}")

        # Combine all datasets
        from datasets import concatenate_datasets
        combined_dataset = concatenate_datasets(all_datasets)

        print(f"Combined {len(all_datasets)} datasets with {len(combined_dataset)} total examples")
        return combined_dataset

    else:
        # Single dataset mode
        print(f"Loading and formatting dataset: {args.dataset}...")

        # Determine if dataset is local file or Hub repository
        is_hub_repo = not os.path.exists(args.dataset)

        return load_single_dataset(args.dataset, args.dataset_format, tokenizer, is_hub_repo, args.dataset_split)


def main() -> None:
    """
    Main execution function orchestrating the fine-tuning pipeline.
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Load model and tokenizer
    print(f"Loading model: {args.model_name}...")

    load_in_4bit = False
    load_in_8bit = False
    if args.quantization == "4bit":
        load_in_4bit = True
    elif args.quantization == "8bit":
        load_in_8bit = True
    elif args.quantization == "Q6_K":
        print("⚠️  Warning: Q6_K (GGUF) quantization is not natively supported in Unsloth. Falling back to 4-bit quantization. For full GGUF support, consider using llama.cpp separately.")
        load_in_4bit = True
    # 'none' uses full precision (no quantization)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,  # Auto-detect optimal dtype
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )

    # Set chat template for Llama models if missing
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<|start_header_id|>user<|end_header_id|>\n\n' + content + '<|eot_id|>' }}{% elif message['role'] == 'assistant' %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + content + '<|eot_id|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

    # Load and format dataset
    train_dataset = load_and_format_dataset(args, tokenizer)

    # Apply PEFT (LoRA) adapters
    print("Configuring PEFT adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=args.lora_target_modules.split(","),  # Parse comma-separated string
        lora_alpha=args.lora_alpha,
        lora_dropout=0,  # Standard default
        bias="none",  # Standard for LoRA
        use_gradient_checkpointing="unsloth",  # Use Unsloth's implementation
        random_state=3407,  # For reproducibility
        use_rslora=False,  # Use standard LoRA
        loftq_config=None,  # Not using LoftQ
    )

    # Configure training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=5,  # Small default for stable training
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=not is_bfloat16_supported(),  # Use fp16 if bfloat16 not supported
        bf16=is_bfloat16_supported(),  # Use bfloat16 if supported
        logging_steps=args.logging_steps,  # Use parsed arg
        save_steps=args.save_steps,  # Use parsed arg
        save_total_limit=args.save_total_limit,  # Use parsed arg
        optim="adamw_8bit",  # Memory-efficient optimizer
        weight_decay=0.01,  # Standard weight decay
        lr_scheduler_type="linear",  # Linear learning rate schedule
        seed=3407,  # For reproducibility
        output_dir=args.output_dir,
    )

    # Initialize SFTTrainer
    print("Configuring SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",  # Standardized column name
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,  # Parallel processing for data loading
        packing=False,  # False for broader compatibility (can be set to True for better throughput on short examples)
        args=training_args,
    )

    # Execute training
    print("Starting training...")
    trainer.train()

    # Save trained model and tokenizer
    print(f"Training complete. Saving LoRA adapters to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("Fine-tuning completed successfully!")

    # Provide manual quantization command
    print("\n🔄 To quantize the trained model to GGUF format, run:")
    print(f"python -m llama.cpp.convert --model {args.output_dir} --outtype q8_0 --outfile {args.output_dir}/model-q8_0.gguf")
    print("Note: Requires llama.cpp to be installed for GGUF conversion.")


if __name__ == "__main__":
    main()
