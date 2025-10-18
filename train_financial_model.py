#!/usr/bin/env python3
"""Financial fine-tuning pipeline for DSR1-Distill-Qwen-14B.

This script orchestrates the full workflow needed to produce an instruction-tuned
financial assistant model starting from the `DSR1/DSR1-Distill-Qwen-14B` base model.
It performs four high-level stages:

1. Dataset preparation (conversion + Investopedia ingestion)
2. Supervised fine-tuning with LoRA adapters via `train.py`
3. Adapter merging into a full-precision Hugging Face model
4. Conversion to GGUF and quantisation to `Q4_K_M`

Each stage can be skipped with CLI flags to make the pipeline resumable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from convert_datasets import main as convert_datasets
from train import parse_arguments as parse_train_arguments
from train import run_training as run_sft_training


DEFAULT_BASE_MODEL = "DSR1/DSR1-Distill-Qwen-14B"
DEFAULT_LORA_DIR = Path("financial_lora")
DEFAULT_MERGED_DIR = Path("merged_financial_fp16")
DEFAULT_GGUF_FP16 = Path("financial_model_fp16.gguf")
DEFAULT_GGUF_Q4 = Path("financial_model_q4_k_m.gguf")
DEFAULT_DATASET_DIR = Path("datasets")
DEFAULT_LLAMA_CPP = Path("llama.cpp")
DEFAULT_QUANTIZATION_TARGET = "Q4_K_M"


class FinancialTrainingPipeline:
    """End-to-end trainer for the financial assistant model."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.base_model: str = args.base_model or DEFAULT_BASE_MODEL
        self.dataset_dir: Path = Path(args.dataset_folder or DEFAULT_DATASET_DIR)
        self.lora_output_dir: Path = Path(args.lora_output_dir or DEFAULT_LORA_DIR)
        self.merged_model_dir: Path = Path(args.merged_model_dir or DEFAULT_MERGED_DIR)
        self.gguf_fp16_path: Path = Path(args.gguf_fp16_path or DEFAULT_GGUF_FP16)
        self.gguf_q4_path: Path = Path(args.gguf_q4_path or DEFAULT_GGUF_Q4)
        self.llama_cpp_path: Optional[Path] = Path(args.llama_cpp_path).expanduser() if args.llama_cpp_path else None
        self.quantization_target: str = args.quantization_target or DEFAULT_QUANTIZATION_TARGET

    # ------------------------------------------------------------------
    # Dataset handling
    # ------------------------------------------------------------------
    def prepare_datasets(self) -> None:
        if self.args.skip_dataset_prep:
            print("Skipping dataset preparation (--skip_dataset_prep).")
            return

        print("\n[1/4] Preparing datasets...")
        convert_datasets()
        self._download_investopedia_dataset()

    def _download_investopedia_dataset(self) -> None:
        """Download and standardise the FinLang Investopedia dataset."""
        output_dir = self.dataset_dir / "investopedia"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            print("Downloading FinLang/investopedia-instruction-tuning-dataset...")
            dataset = load_dataset("FinLang/investopedia-instruction-tuning-dataset")
        except Exception as exc:  # pragma: no cover - network/runtime guard
            print(f"⚠️  Failed to download Investopedia dataset: {exc}")
            return

        total_records = 0
        for split, split_dataset in dataset.items():
            records: List[dict[str, str]] = []
            for item in split_dataset:
                instruction = (item.get("Question") or "").strip()
                answer = (item.get("Answer") or "").strip()
                context = (item.get("Context") or "").strip()

                if not instruction or not answer:
                    continue

                if context:
                    instruction = f"Context: {context}\n\nQuestion: {instruction}"

                records.append({
                    "instruction": instruction,
                    "input": "",
                    "output": answer,
                })

            if not records:
                continue

            output_file = output_dir / f"investopedia_{split}.jsonl"
            with output_file.open("w", encoding="utf-8") as handle:
                for record in records:
                    json.dump(record, handle, ensure_ascii=False)
                    handle.write("\n")

            print(f"Saved {len(records)} records to {output_file}")
            total_records += len(records)

        if total_records == 0:
            print("⚠️  No Investopedia records were saved. Check dataset availability.")
        else:
            print(f"Investopedia ingestion complete. {total_records} records written.")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def run_training(self) -> None:
        if self.args.skip_training:
            print("Skipping supervised fine-tuning (--skip_training).")
            return

        print("\n[2/4] Starting supervised fine-tuning...")
        train_cli_args = self._build_training_arguments()

        print("Invoking train.py with arguments:")
        for i in range(0, len(train_cli_args), 2):
            key = train_cli_args[i]
            value = train_cli_args[i + 1] if i + 1 < len(train_cli_args) else ""
            print(f"  {key:<32} {value}")

        train_args = parse_train_arguments(train_cli_args)
        run_sft_training(train_args)

    def _build_training_arguments(self) -> List[str]:
        args: List[str] = [
            "--model_name", self.base_model,
            "--dataset_folder", str(self.dataset_dir),
            "--output_dir", str(self.lora_output_dir),
            "--max_seq_length", str(self.args.max_seq_length),
            "--num_train_epochs", str(self.args.num_train_epochs),
            "--per_device_train_batch_size", str(self.args.per_device_train_batch_size),
            "--gradient_accumulation_steps", str(self.args.gradient_accumulation_steps),
            "--learning_rate", str(self.args.learning_rate),
            "--lora_r", str(self.args.lora_r),
            "--lora_alpha", str(self.args.lora_alpha),
            "--lora_target_modules", self.args.lora_target_modules,
            "--quantization", self.args.training_quantization,
            "--logging_steps", str(self.args.logging_steps),
            "--save_steps", str(self.args.save_steps),
            "--save_total_limit", str(self.args.save_total_limit),
        ]

        if self.args.extra_train_args:
            args.extend(self.args.extra_train_args)

        return args

    # ------------------------------------------------------------------
    # Merge + quantisation
    # ------------------------------------------------------------------
    def merge_adapters(self) -> None:
        if self.args.skip_merge:
            print("Skipping LoRA merge (--skip_merge).")
            return

        print("\n[3/4] Merging LoRA adapters into full model...")
        self.merged_model_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            device_map=self.args.merge_device_map,
        )
        peft_model = PeftModel.from_pretrained(base_model, str(self.lora_output_dir))
        merged_model = peft_model.merge_and_unload()

        merged_model.save_pretrained(self.merged_model_dir)
        tokenizer.save_pretrained(self.merged_model_dir)
        print(f"Merged model written to {self.merged_model_dir}")

    def convert_to_gguf(self) -> None:
        if self.args.skip_quantization:
            print("Skipping GGUF conversion/quantisation (--skip_quantization).")
            return

        if self.llama_cpp_path is None:
            raise ValueError("Quantisation requested but --llama_cpp_path was not provided.")

        convert_script = self.llama_cpp_path / "convert_hf_to_gguf.py"
        if not convert_script.exists():
            raise FileNotFoundError(f"convert_hf_to_gguf.py not found at {convert_script}")

        self.gguf_fp16_path.parent.mkdir(parents=True, exist_ok=True)

        print("\n[4/4] Converting merged model to GGUF...")
        cmd = [
            sys.executable,
            str(convert_script),
            "--model",
            str(self.merged_model_dir),
            "--outfile",
            str(self.gguf_fp16_path),
            "--outtype",
            "f16",
        ]
        subprocess.run(cmd, check=True)
        print(f"FP16 GGUF saved to {self.gguf_fp16_path}")

        quant_binary = self.llama_cpp_path / "build" / "bin" / "llama-quantize"
        if not quant_binary.exists():
            raise FileNotFoundError(f"llama-quantize binary not found at {quant_binary}")

        cmd = [
            str(quant_binary),
            str(self.gguf_fp16_path),
            str(self.gguf_q4_path),
            self.quantization_target,
        ]
        subprocess.run(cmd, check=True)
        print(f"Quantised GGUF ({self.quantization_target}) saved to {self.gguf_q4_path}")

    # ------------------------------------------------------------------
    def run(self) -> None:
        self.prepare_datasets()
        self.run_training()
        self.merge_adapters()
        self.convert_to_gguf()
        print("\nPipeline complete!")
        if not self.args.skip_quantization:
            print(f"Quantised model available at {self.gguf_q4_path}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and quantise the financial assistant model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_financial_model.py
  python train_financial_model.py --skip_dataset_prep --skip_training --merge_adapters
  python train_financial_model.py --llama_cpp_path ~/src/llama.cpp --gguf_q4_path models/finance_q4.gguf
        """
    )

    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--dataset_folder", type=str, default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--lora_output_dir", type=str, default=str(DEFAULT_LORA_DIR))
    parser.add_argument("--merged_model_dir", type=str, default=str(DEFAULT_MERGED_DIR))
    parser.add_argument("--gguf_fp16_path", type=str, default=str(DEFAULT_GGUF_FP16))
    parser.add_argument("--gguf_q4_path", type=str, default=str(DEFAULT_GGUF_Q4))
    parser.add_argument(
        "--llama_cpp_path",
        type=str,
        default=str(DEFAULT_LLAMA_CPP),
        help="Path to local llama.cpp repository",
    )

    # Training hyper-parameters
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--training_quantization", type=str, default="4bit", choices=["4bit", "8bit", "none"])
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)

    parser.add_argument(
        "--extra_train_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to train.py",
    )

    # Pipeline toggles
    parser.add_argument("--skip_dataset_prep", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_merge", action="store_true")
    parser.add_argument("--skip_quantization", action="store_true")
    parser.add_argument("--merge_device_map", type=str, default="auto")
    parser.add_argument("--quantization_target", type=str, default=DEFAULT_QUANTIZATION_TARGET)

    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    pipeline = FinancialTrainingPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
