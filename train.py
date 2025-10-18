#!/usr/bin/env python3
"""
Universal Command-Line Fine-Tuning Utility for Language Models

This script provides a zero-friction command-line interface for fine-tuning large language models
using Unsloth's optimized framework. It supports both Alpaca and ShareGPT data formats from
Hugging Face Hub repositories, local files, or organized dataset folders.
"""

# 1. Unsloth import must be first
from unsloth import FastLanguageModel, is_bfloat16_supported

import argparse
import os
from typing import Any, Dict, Optional, Sequence

from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer, SFTConfig

def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Universal Command-Line Fine-Tuning Utility for Language Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode configuration
    parser.add_argument(
        "--financial_model",
        action="store_true",
        help="Run in financial model training mode. Overrides other settings."
    )

    # Model configuration
    parser.add_argument("--model_name", type=str, help="Base model from Hugging Face Hub.")
    
    # Dataset configuration
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument("--dataset", type=str, help="Hugging Face Hub dataset or local file.")
    dataset_group.add_argument("--dataset_folder", type=str, help="Path to a folder with datasets.")

    parser.add_argument("--dataset_format", type=str, default="alpaca", choices=["alpaca", "sharegpt"])
    parser.add_argument("--dataset_split", type=str, default="train")
    
    # Output configuration
    parser.add_argument("--output_dir", type=str, default="./lora_model")

    # Model & Training parameters
    parser.add_argument("--max_seq_length", type=int)
    parser.add_argument("--num_train_epochs", type=int)
    parser.add_argument("--per_device_train_batch_size", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--lora_r", type=int)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--lora_target_modules", type=str)
    parser.add_argument("--quantization", type=str, choices=["4bit", "8bit", "none", "Q6_K"])
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)

    return parser.parse_args(argv)

def format_alpaca_dataset(dataset: Any) -> Any:
    """Formats an Alpaca-style dataset."""
    alpaca_prompt = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:\n{}"
    
    def format_example(example: Dict[str, Any]) -> Dict[str, Any]:
        formatted_text = alpaca_prompt.format(
            example.get("instruction", ""),
            example.get("input", ""),
            example.get("output", "")
        )
        return {"text": formatted_text}
    
    return dataset.map(format_example)

def format_sharegpt_dataset(dataset: Any, tokenizer: Any) -> Any:
    """Formats a ShareGPT-style dataset."""
    def format_example(example: Dict[str, Any]) -> Dict[str, Any]:
        formatted_text = tokenizer.apply_chat_template(
            example.get("conversations", []),
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": formatted_text}
    
    return dataset.map(format_example)

def detect_dataset_format(dataset_path: str) -> str:
    """Auto-detects dataset format."""
    try:
        sample_dataset = load_dataset("json", data_files=dataset_path, split="train[:5]")
        sample_record = sample_dataset[0]
        if 'conversations' in sample_record:
            return 'sharegpt'
        if all(field in sample_record for field in ['instruction', 'input', 'output']):
            return 'alpaca'
    except Exception:
        pass
    return 'alpaca'

def load_and_format_dataset(args: argparse.Namespace, tokenizer: Any) -> Any:
    """Loads and formats dataset(s)."""
    if args.dataset_folder:
        all_datasets = []
        for root, _, files in os.walk(args.dataset_folder):
            for file in files:
                if file.endswith(('.json', '.jsonl')):
                    file_path = os.path.join(root, file)
                    folder_name = os.path.basename(root).lower()
                    if folder_name in ['alpaca', 'sharegpt']:
                        dataset_format = folder_name
                    else:
                        dataset_format = detect_dataset_format(file_path)
                    
                    ds = load_dataset("json", data_files=file_path, split="train")
                    if dataset_format == "alpaca":
                        all_datasets.append(format_alpaca_dataset(ds))
                    else:
                        all_datasets.append(format_sharegpt_dataset(ds, tokenizer))
        return concatenate_datasets(all_datasets)
    else:
        is_hub_repo = not os.path.exists(args.dataset)
        ds = load_dataset(args.dataset, split=args.dataset_split) if is_hub_repo else load_dataset("json", data_files=args.dataset, split="train")
        
        if args.dataset_format == "alpaca":
            return format_alpaca_dataset(ds)
        else:
            return format_sharegpt_dataset(ds, tokenizer)

def run_training(args: argparse.Namespace) -> None:
    """Execute training with a pre-parsed argument namespace."""

    # Financial model mode overrides
    if args.financial_model:
        print("Running in financial model mode...")
        args.model_name = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
        args.dataset_folder = "datasets/"
        args.output_dir = "./financial_lora"
        args.max_seq_length = 4096
        args.num_train_epochs = 3
        args.per_device_train_batch_size = 1
        args.gradient_accumulation_steps = 8
        args.learning_rate = 1e-5
        args.lora_r = 32
        args.lora_alpha = 64
        args.quantization = "4bit"
        if not args.dataset and not args.dataset_folder:
             raise ValueError("Financial model mode requires datasets to be in the 'datasets' folder.")

    # Set default values after financial model overrides
    if args.model_name is None: args.model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
    if args.max_seq_length is None: args.max_seq_length = 2048
    if args.num_train_epochs is None: args.num_train_epochs = 1
    if args.per_device_train_batch_size is None: args.per_device_train_batch_size = 2
    if args.gradient_accumulation_steps is None: args.gradient_accumulation_steps = 4
    if args.learning_rate is None: args.learning_rate = 2e-4
    if args.lora_r is None: args.lora_r = 16
    if args.lora_alpha is None: args.lora_alpha = 16
    if args.quantization is None: args.quantization = "4bit"
    if args.lora_target_modules is None: args.lora_target_modules = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    
    if not args.dataset and not args.dataset_folder:
        raise ValueError("Either --dataset or --dataset_folder must be provided.")

    load_in_4bit = "4bit" in args.quantization
    load_in_8bit = args.quantization == "8bit"

    print(f"Loading model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
    )

    # Set chat template for Llama models if missing
    if tokenizer.chat_template is None:
        tokenizer.chat_template = """<|begin_of_text|>{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] }}{% elif message['role'] == 'user' %}{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% elif message['role'] == 'assistant' %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"""

    print("Configuring PEFT adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=args.lora_target_modules.split(","),
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    print("Loading and formatting dataset...")
    train_dataset = load_and_format_dataset(args, tokenizer)

    print("Configuring SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=SFTConfig(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=5,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            optim="adamw_8bit",
            seed=3407,
        ),
    )

    print("Starting training...")
    trainer.train()

    print("Fine-tuning completed successfully!")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def main() -> None:
    """CLI entry point."""
    args = parse_arguments()
    run_training(args)


if __name__ == "__main__":
    main()
