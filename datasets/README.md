# Dataset Folder Structure

This folder contains sample datasets demonstrating the supported formats for the fine-tuning utility.

## Folder Organization

- **`alpaca/`**: Place Alpaca-format datasets here. All files in this folder will be automatically treated as Alpaca format.
- **`sharegpt/`**: Place ShareGPT-format datasets here. All files in this folder will be automatically treated as ShareGPT format.
- **`mixed/`**: Place datasets here for automatic format detection. The script will examine each file's structure to determine if it's Alpaca or ShareGPT format.

## Supported File Formats

- **JSON** (`.json`): Standard JSON array of objects
- **JSONL** (`.jsonl`): One JSON object per line

## Data Format Specifications

### Alpaca Format
Each record should contain:
```json
{
  "instruction": "The task description",
  "input": "Additional context (optional, can be empty)",
  "output": "The expected response"
}
```

### ShareGPT Format
Each record should contain:
```json
{
  "conversations": [
    {
      "from": "human",
      "value": "User message"
    },
    {
      "from": "assistant",
      "value": "Assistant response"
    }
  ]
}
```

## Usage

To train on all datasets in this folder:

```bash
python train.py --dataset_folder "./datasets/"
```

The script will:
1. Scan all subfolders for `.json` and `.jsonl` files
2. Auto-detect format based on folder name or file content
3. Load and combine all datasets
4. Train the model on the combined dataset

## Adding Your Own Datasets

1. Place your dataset files in the appropriate subfolder
2. Ensure they follow the correct JSON/JSONL format
3. Run the training command above

The script will automatically handle format detection and dataset combination.
