"""
prepare_llava_cot.py

Script to prepare LLaVA-CoT-100k dataset for fine-tuning.
This script handles downloading, format conversion, and validation of the LLaVA-CoT dataset.

The LLaVA-CoT dataset contains structured reasoning annotations that need to be
formatted to match the expected conversation structure for FinetuneDataset.

Usage:
    python fine_tune/prepare_llava_cot.py --dataset_root data
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import draccus
from tqdm import tqdm

from cobra.overwatch import initialize_overwatch
from fine_tune.download_llava_cot import download_llava_cot_dataset

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


@dataclass
class PrepareLLaVACoTConfig:
    # fmt: off
    dataset_root: Path = Path("data")                    # Root directory for datasets
    download_dataset: bool = True                        # Whether to download the dataset
    validate_format: bool = True                         # Whether to validate JSON format
    convert_format: bool = False                         # Whether to convert format (if needed)
    # fmt: on


def load_jsonl(file_path: Path) -> List[Dict]:
    """Load JSONL file (one JSON object per line)."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))
    return data


def validate_llava_cot_format(data: List[Dict]) -> bool:
    """
    Validate that the LLaVA-CoT dataset follows the expected format.
    
    Expected format:
    {
        "id": str,
        "image": str,  # Path to image
        "conversations": [
            {"from": "human", "value": str},  # Question with <image> token
            {"from": "gpt", "value": str}     # Reasoning chain + Answer
        ]
    }
    """
    required_keys = {"id", "conversations"}
    valid = True
    
    for i, example in enumerate(data):
        # Check required keys
        if not all(key in example for key in required_keys):
            overwatch.warning(f"Example {i} missing required keys: {required_keys - set(example.keys())}")
            valid = False
            continue
        
        # Check conversations format
        conversations = example["conversations"]
        if not isinstance(conversations, list) or len(conversations) < 2:
            overwatch.warning(f"Example {i} has invalid conversations format")
            valid = False
            continue
        
        # Check that we have human and assistant turns
        has_human = any(turn.get("from") in ("human", "user") for turn in conversations)
        has_assistant = any(turn.get("from") in ("gpt", "assistant") for turn in conversations)
        
        if not (has_human and has_assistant):
            overwatch.warning(f"Example {i} missing human or assistant turn")
            valid = False
            continue
        
        # Check for image reference in human turn
        if "image" in example:
            human_turn = next((t for t in conversations if t.get("from") in ("human", "user")), None)
            if human_turn and "<image>" not in human_turn.get("value", ""):
                overwatch.warning(f"Example {i} has image but <image> token not found in human turn")
    
    return valid


def convert_to_llava_format(data: List[Dict], output_path: Path) -> None:
    """
    Convert LLaVA-CoT dataset to standard LLaVA format if needed.
    
    This function handles format conversions if the source dataset uses a different structure.
    """
    overwatch.info("Converting dataset to LLaVA format...")
    
    converted_data = []
    for example in tqdm(data, desc="Converting examples"):
        # If already in correct format, use as-is
        if "conversations" in example and isinstance(example["conversations"], list):
            converted_data.append(example)
        else:
            # Handle alternative formats here if needed
            # For now, assume format is already correct
            converted_data.append(example)
    
    with open(output_path, "w") as f:
        json.dump(converted_data, f, indent=2)
    
    overwatch.info(f"Converted dataset saved to {output_path}")


@draccus.wrap()
def prepare_llava_cot(cfg: PrepareLLaVACoTConfig) -> None:
    """Main function to prepare LLaVA-CoT-100k dataset."""
    overwatch.info("Preparing LLaVA-CoT-100k Dataset")
    
    dataset_dir = cfg.dataset_root / "download" / "llava-cot-100k"
    # The dataset comes as train.jsonl (JSONL format - one JSON object per line)
    dataset_json = dataset_dir / "train.jsonl"
    if not dataset_json.exists():
        # Fallback to renamed version if it exists
        alt_json = dataset_dir / "llava_cot_100k.json"
        if alt_json.exists():
            overwatch.info(f"Found {alt_json.name}, will use it as dataset file")
            dataset_json = alt_json
    
    # Step 1: Download dataset if requested
    if cfg.download_dataset:
        overwatch.info("Downloading LLaVA-CoT-100k dataset...")
        overwatch.info("This will download:")
        overwatch.info("  1. train.jsonl (conversation data)")
        overwatch.info("  2. image.zip.part-aa through image.zip.part-ap (split image archive)")
        overwatch.info("  3. Merge and extract the image archive")
        try:
            download_llava_cot_dataset(cfg.dataset_root)
            overwatch.info("✓ Dataset and images downloaded successfully")
        except Exception as e:
            overwatch.error(f"Failed to download dataset: {e}")
            overwatch.info("You may need to manually download the dataset and place it in:")
            overwatch.info(f"  {dataset_json}")
            overwatch.info("Dataset sources:")
            overwatch.info("  - GitHub: https://github.com/PKU-YuanGroup/LLaVA-CoT")
            overwatch.info("  - HuggingFace: https://huggingface.co/datasets/Xkev/LLaVA-CoT-100k")
            overwatch.info("Note: Images come as split zip files that need to be merged manually:")
            overwatch.info("  cat image.zip.part-* > image.zip && unzip image.zip")
            overwatch.info("See FINETUNING_IMPLEMENTATION.md for more details.")
            return
    
    # Step 2: Validate format
    if not dataset_json.exists():
        overwatch.error(f"Dataset JSON not found at {dataset_json}")
        overwatch.info("Please download the dataset manually or check the download URL.")
        return
    
    if cfg.validate_format:
        overwatch.info("Validating dataset format...")
        # Handle both JSONL and JSON formats
        if dataset_json.suffix == ".jsonl" or dataset_json.name.endswith(".jsonl"):
            data = load_jsonl(dataset_json)
            overwatch.info(f"Loaded {len(data)} examples from JSONL file")
        else:
            with open(dataset_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                overwatch.error("Dataset JSON should contain a list of examples")
                return
            overwatch.info(f"Loaded {len(data)} examples from JSON file")
        
        if validate_llava_cot_format(data):
            overwatch.info("✓ Dataset format validation passed")
        else:
            overwatch.warning("Dataset format validation found issues (see warnings above)")
    
    # Step 3: Convert format if needed
    if cfg.convert_format:
        output_path = dataset_dir / "llava_cot_100k_converted.json"
        # Load data (handle both JSONL and JSON)
        if dataset_json.suffix == ".jsonl" or dataset_json.name.endswith(".jsonl"):
            data = load_jsonl(dataset_json)
        else:
            with open(dataset_json, "r", encoding="utf-8") as f:
                data = json.load(f)
        convert_to_llava_format(data, output_path)
        overwatch.info(f"Converted dataset saved to {output_path}")
    
    # Step 4: Check image paths
    overwatch.info("Checking image paths...")
    # Load data (handle both JSONL and JSON)
    if dataset_json.suffix == ".jsonl" or dataset_json.name.endswith(".jsonl"):
        data = load_jsonl(dataset_json)
    else:
        with open(dataset_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    
    image_dir = dataset_dir
    missing_images = 0
    sample_checked = min(100, len(data))  # Check first 100 examples
    
    for example in tqdm(data[:sample_checked], desc="Checking image paths"):
        if "image" in example:
            image_path = image_dir / example["image"]
            if not image_path.exists():
                # Try alternative paths
                alt_paths = [
                    dataset_dir / "images" / example["image"],
                    cfg.dataset_root / "download" / "llava-v1.5-instruct" / example["image"],
                ]
                found = False
                for alt_path in alt_paths:
                    if alt_path.exists():
                        found = True
                        break
                if not found:
                    missing_images += 1
    
    if missing_images > 0:
        overwatch.warning(f"Found {missing_images} missing images in sample (out of {sample_checked})")
        overwatch.info("You may need to download images separately or adjust image paths")
    else:
        overwatch.info("✓ Image paths validated")
    
    overwatch.info("Dataset preparation complete!")
    overwatch.info(f"Dataset ready at: {dataset_json}")
    overwatch.info("You can now use this dataset for fine-tuning with:")
    overwatch.info("  --dataset.type llava-cot-100k")


if __name__ == "__main__":
    prepare_llava_cot()

