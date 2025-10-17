#!/usr/bin/env python3
"""
Financial Model Training Setup Script

This script sets up and initiates comprehensive training for the ServiceNow/Apriel-Nemotron-15B-Thinker model
on financial and trading datasets. It includes:

1. Dataset preparation and conversion
2. Model configuration for optimal financial domain performance
3. Multi-stage training (SFT + optional DPO)
4. Integration of multiple financial datasets

Usage:
    python train_financial_model.py [options]

Datasets Used:
- Trader SFT data (conversational trading assistant)
- Trader DPO data (converted to SFT format)
- FinLang Investopedia instruction-tuning dataset
- Combined financial knowledge base

Model: ServiceNow/Apriel-Nemotron-15B-Thinker (Q6_K GGUF quantized)
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from convert_datasets import main as convert_datasets
from train import main as train_main


class FinancialModelTrainer:
    """Comprehensive trainer for financial domain models."""

    def __init__(self, args):
        self.args = args
        self.project_root = Path(__file__).parent
        self.datasets_dir = self.project_root / "datasets"
        self.output_dir = Path(args.output_dir) if args.output_dir else self.project_root / "financial_model_output"

        # Model configuration - Original model with GGUF support
        self.model_name = "ServiceNow/Apriel-Nemotron-15B-Thinker"
        self.quantization = "Q6_K"

        # Dataset sources
        self.dataset_sources = [
            "FinLang/investopedia-instruction-tuning-dataset",
            str(self.datasets_dir)  # Our converted datasets
        ]

    def download_investopedia_dataset(self):
        """Download and prepare the Investopedia instruction-tuning dataset."""
        print("📥 Downloading FinLang Investopedia dataset...")

        investopedia_dir = self.datasets_dir / "investopedia"
        investopedia_dir.mkdir(exist_ok=True)

        try:
            # Import here to avoid subprocess issues
            from datasets import load_dataset
            import json

            # Load the dataset
            print("Loading Investopedia dataset...")
            dataset = load_dataset("FinLang/investopedia-instruction-tuning-dataset")

            # Convert to Alpaca format for each split
            for split_name in dataset.keys():
                print(f"Converting {split_name} split...")
                split_data = dataset[split_name]

                alpaca_records = []
                for item in split_data:
                    # Create instruction from question, output from answer
                    question = item.get('Question')
                    answer = item.get('Answer')
                    context = item.get('Context')

                    instruction = (question.strip() if question else '') if isinstance(question, str) else ''
                    output = (answer.strip() if answer else '') if isinstance(answer, str) else ''
                    context = (context.strip() if context else '') if isinstance(context, str) else ''

                    if instruction and output:
                        # Add context if available
                        if context:
                            instruction = f"Context: {context}\n\nQuestion: {instruction}"

                        alpaca_records.append({
                            "instruction": instruction,
                            "input": "",
                            "output": output
                        })

                # Save as JSONL
                output_file = investopedia_dir / f"investopedia_{split_name}.jsonl"
                with open(output_file, 'w', encoding='utf-8') as f:
                    for record in alpaca_records:
                        json.dump(record, f, ensure_ascii=False)
                        f.write('\n')

                print(f"Saved {len(alpaca_records)} records to {output_file}")

            print("✅ Investopedia dataset downloaded and converted")
            return True

        except Exception as e:
            print(f"Error downloading Investopedia dataset: {e}")
            import traceback
            traceback.print_exc()
            return False

    def prepare_datasets(self):
        """Prepare all datasets for training."""
        print("🔧 Preparing datasets...")

        # Step 1: Convert existing random/ datasets
        print("Converting existing datasets from random/ folder...")
        convert_datasets()

        # Step 2: Download and convert Investopedia dataset
        if self.download_investopedia_dataset():
            print("✅ All datasets prepared")
            return True
        else:
            print("❌ Failed to prepare datasets")
            return False

    def setup_training_args(self) -> List[str]:
        """Set up training arguments optimized for financial domain."""

        # Base training arguments
        training_args = [
            "--model_name", self.model_name,
            "--dataset_folder", str(self.datasets_dir),
            "--output_dir", str(self.output_dir),

            # Financial domain optimization
            "--max_seq_length", "4096",  # Longer context for financial analysis
            "--num_train_epochs", "3",   # Multiple epochs for domain adaptation

            # Memory optimization for large model
            "--per_device_train_batch_size", "1",  # Conservative batch size
            "--gradient_accumulation_steps", "8",  # Effective batch size of 8
            "--learning_rate", "2e-5",  # Lower LR for fine-tuning

            # LoRA configuration optimized for financial tasks
            "--lora_r", "32",  # Higher rank for complex financial reasoning
            "--lora_alpha", "64",  # Higher alpha for better adaptation
            "--lora_target_modules", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",

            # Quantization and training monitoring
            "--quantization", "Q6_K",
            "--logging_steps", "10",
            "--save_steps", "500",
            "--save_total_limit", "3",

            # Logging and monitoring
            "--logging_steps", "10",
            "--save_steps", "500",
            "--save_total_limit", "3",
        ]

        # Add any additional args from command line
        if hasattr(self.args, 'extra_args') and self.args.extra_args:
            training_args.extend(self.args.extra_args)

        return training_args

    def run_training(self):
        """Execute the training process."""
        print(f"🚀 Starting training for {self.model_name}")
        print(f"Output directory: {self.output_dir}")
        print(f"Datasets: {', '.join(self.dataset_sources)}")

        # Prepare training arguments
        training_args = self.setup_training_args()

        print("Training configuration:")
        for i in range(0, len(training_args), 2):
            if i + 1 < len(training_args):
                print(f"  {training_args[i]:<30} {training_args[i+1]}")

        # Set up environment for training
        env = os.environ.copy()

        # Run training
        try:
            print("\n🎯 Starting training process...")

            # Import and run the training function directly
            # This avoids subprocess issues with imports
            from train import parse_arguments, main as train_main

            # Create a modified argument parser to accept our pre-configured args
            original_argv = sys.argv.copy()
            sys.argv = ['train.py'] + training_args

            try:
                args = parse_arguments()
                train_main()
                print("✅ Training completed successfully!")
                print(f"Model saved to: {self.output_dir}")

            finally:
                sys.argv = original_argv

        except KeyboardInterrupt:
            print("\n⏹️  Training interrupted by user")
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            return False

        return True

    def run(self):
        """Main execution flow."""
        print("🤖 Financial Model Training Setup")
        print("=" * 50)

        # Step 1: Prepare datasets
        if not self.prepare_datasets():
            return False

        # Step 2: Run training
        if not self.run_training():
            return False

        print("\n🎉 Financial model training complete!")
        print(f"Trained model available at: {self.output_dir}")

        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train ServiceNow/Apriel-Nemotron-15B-Thinker on financial datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_financial_model.py
  python train_financial_model.py --output_dir ./my_financial_model
  python train_financial_model.py --extra_args --learning_rate 1e-5 --num_train_epochs 5

Datasets included:
- Trader SFT conversational data
- Trader DPO preference data (converted to SFT)
- FinLang Investopedia instruction-tuning dataset
- Combined financial knowledge base
        """
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the trained model (default: ./financial_model_output)"
    )

    parser.add_argument(
        "--extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments to pass to the training script"
    )

    args = parser.parse_args()

    trainer = FinancialModelTrainer(args)
    success = trainer.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
