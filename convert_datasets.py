#!/usr/bin/env python3
"""
Dataset Converter and Organizer

This script scans the random/ folder for datasets, converts them to appropriate formats
for the fine-tuning utility, and moves them to the correct folders in datasets/.

Supported conversions:
- DPO format (prompt, chosen, rejected) → Alpaca format (instruction, input, output)
- ShareGPT format (messages array) → Remains as ShareGPT, moved to sharegpt/ folder
- Investopedia format (Question, Answer, Context) → Alpaca format

Usage:
    python convert_datasets.py

The script will:
1. Scan random/ folder for .json and .jsonl files
2. Auto-detect format of each dataset
3. Convert DPO datasets to Alpaca format for SFT training
4. Convert Investopedia datasets to Alpaca format
5. Move datasets to appropriate folders:
   - Alpaca format → datasets/alpaca/
   - ShareGPT format → datasets/sharegpt/
6. Provide detailed logging of conversions and moves

Note: The train_financial_model.py script automatically downloads and converts
the FinLang/investopedia-instruction-tuning-dataset from Hugging Face.
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any


def detect_dataset_format(file_path: str, sample_size: int = 5) -> str:
    """
    Detect the format of a dataset by examining sample records.

    Args:
        file_path: Path to the dataset file
        sample_size: Number of records to sample

    Returns:
        Format type: 'dpo', 'sharegpt', or 'unknown'
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:sample_size]

        if not lines:
            return 'unknown'

        # Try to parse first line as JSON
        first_record = json.loads(lines[0].strip())

        # Check for DPO format
        if all(key in first_record for key in ['prompt', 'chosen', 'rejected']):
            return 'dpo'

        # Check for ShareGPT format
        if 'messages' in first_record and isinstance(first_record['messages'], list):
            return 'sharegpt'

        # Check for messages array (alternative ShareGPT)
        if 'conversations' in first_record and isinstance(first_record['conversations'], list):
            return 'sharegpt'

        return 'unknown'

    except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as e:
        print(f"Warning: Error detecting format for {file_path}: {e}")
        return 'unknown'


def convert_dpo_to_alpaca(dpo_file_path: str, output_file_path: str) -> int:
    """
    Convert DPO format dataset to Alpaca format for SFT training.

    Args:
        dpo_file_path: Path to input DPO format file
        output_file_path: Path to output Alpaca format file

    Returns:
        Number of records converted
    """
    converted_count = 0

    with open(dpo_file_path, 'r', encoding='utf-8') as infile, \
         open(output_file_path, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue

            try:
                dpo_record = json.loads(line)

                # Convert DPO format to Alpaca format
                # Use the 'chosen' response as the training target for SFT
                alpaca_record = {
                    "instruction": dpo_record.get("prompt", ""),
                    "input": "",  # DPO format doesn't have separate input
                    "output": dpo_record.get("chosen", "")
                }

                # Write as JSON line
                json.dump(alpaca_record, outfile, ensure_ascii=False)
                outfile.write('\n')
                converted_count += 1

            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed line {line_num} in {dpo_file_path}: {e}")
                continue

    return converted_count


def convert_sharegpt_messages_to_standard(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Convert messages array to standard ShareGPT conversations format.

    Args:
        messages: List of message dictionaries

    Returns:
        Standardized conversations list
    """
    conversations = []

    for msg in messages:
        role = msg.get('role', '')
        content = msg.get('content', '')

        # Map role names
        if role == 'system':
            from_field = 'system'
        elif role == 'user':
            from_field = 'human'
        elif role == 'assistant':
            from_field = 'assistant'
        else:
            from_field = role

        conversations.append({
            'from': from_field,
            'value': content
        })

    return conversations


def standardize_sharegpt_file(input_file_path: str, output_file_path: str) -> int:
    """
    Standardize ShareGPT format to ensure consistent conversation structure.

    Args:
        input_file_path: Path to input file
        output_file_path: Path to output file

    Returns:
        Number of records processed
    """
    processed_count = 0

    with open(input_file_path, 'r', encoding='utf-8') as infile, \
         open(output_file_path, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)

                # Handle different ShareGPT formats
                if 'messages' in record:
                    # Convert messages format to conversations
                    conversations = convert_sharegpt_messages_to_standard(record['messages'])
                    standardized_record = {'conversations': conversations}
                elif 'conversations' in record:
                    # Already in correct format
                    standardized_record = record
                else:
                    print(f"Warning: Skipping line {line_num} in {input_file_path}: no messages or conversations field")
                    continue

                json.dump(standardized_record, outfile, ensure_ascii=False)
                outfile.write('\n')
                processed_count += 1

            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed line {line_num} in {input_file_path}: {e}")
                continue

    return processed_count


def main():
    """Main conversion and organization function."""
    # Define paths
    random_dir = Path("random")
    datasets_dir = Path("datasets")
    alpaca_dir = datasets_dir / "alpaca"
    sharegpt_dir = datasets_dir / "sharegpt"

    # Ensure output directories exist
    alpaca_dir.mkdir(parents=True, exist_ok=True)
    sharegpt_dir.mkdir(parents=True, exist_ok=True)

    # Check if random directory exists
    if not random_dir.exists():
        print(f"Error: {random_dir} directory not found!")
        return

    print("🔍 Scanning random/ directory for datasets...")

    # Find all JSON/JSONL files
    dataset_files = []
    for ext in ['*.json', '*.jsonl']:
        dataset_files.extend(random_dir.glob(ext))

    if not dataset_files:
        print("No dataset files found in random/ directory.")
        return

    print(f"Found {len(dataset_files)} dataset files:")
    for file_path in dataset_files:
        print(f"  - {file_path.name}")

    total_converted = 0
    total_moved = 0

    # Process each file
    for file_path in dataset_files:
        print(f"\n📁 Processing {file_path.name}...")

        # Detect format
        format_type = detect_dataset_format(str(file_path))
        print(f"   Detected format: {format_type}")

        if format_type == 'dpo':
            # Convert DPO to Alpaca
            output_filename = f"converted_{file_path.stem}.jsonl"
            output_path = alpaca_dir / output_filename

            print(f"   Converting DPO → Alpaca format...")
            converted_count = convert_dpo_to_alpaca(str(file_path), str(output_path))

            print(f"   ✅ Converted {converted_count} records to {output_path}")
            total_converted += converted_count

        elif format_type == 'sharegpt':
            # Standardize and move ShareGPT
            output_filename = f"processed_{file_path.name}"
            output_path = sharegpt_dir / output_filename

            print(f"   Standardizing ShareGPT format...")
            processed_count = standardize_sharegpt_file(str(file_path), str(output_path))

            print(f"   ✅ Processed {processed_count} records to {output_path}")
            total_moved += processed_count

        else:
            print(f"   ⚠️  Unknown format, skipping {file_path.name}")
            continue

    # Summary
    print(f"\n🎉 Conversion complete!")
    print(f"   Total records converted: {total_converted}")
    print(f"   Total records moved: {total_moved}")
    print(f"\n📂 Datasets are now organized in:")
    print(f"   Alpaca format: {alpaca_dir}/")
    print(f"   ShareGPT format: {sharegpt_dir}/")
    print(f"\n🚀 You can now run: python train.py --dataset_folder datasets/")


if __name__ == "__main__":
    main()
