"""
llava_cot_dataset.py

Custom dataset loader for LLaVA-CoT-100k that handles JSONL format.
This is a wrapper around the existing FinetuneDataset that adds JSONL support
without modifying the core cobra codebase.
"""
import json
from pathlib import Path
from typing import Dict, List, Type

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from cobra.models.backbones.llm.prompting import PromptBuilder
from cobra.models.backbones.vision import ImageTransform
from cobra.preprocessing.datasets import FinetuneDataset

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


def load_jsonl(file_path: Path) -> List[Dict]:
    """Load JSONL file (one JSON object per line)."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))
    return data


def convert_jsonl_to_json(jsonl_path: Path, json_path: Path) -> None:
    """Convert JSONL file to JSON array format for compatibility."""
    data = load_jsonl(jsonl_path)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


class LLaVACoTDataset(FinetuneDataset):
    """
    Dataset loader for LLaVA-CoT-100k that handles JSONL format.
    
    This extends FinetuneDataset to support JSONL files by converting them
    to JSON format on-the-fly, or by loading them directly if the base class
    is updated to support JSONL.
    """
    
    def __init__(
        self,
        instruct_json: Path,
        image_dir: Path,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
        prompt_builder_fn: Type[PromptBuilder],
        convert_to_json: bool = True,
    ) -> None:
        """
        Initialize LLaVA-CoT dataset loader.
        
        :param instruct_json: Path to train.jsonl or JSON file
        :param image_dir: Directory containing images
        :param image_transform: Image transformation pipeline
        :param tokenizer: Text tokenizer
        :param prompt_builder_fn: Prompt builder class
        :param convert_to_json: If True, convert JSONL to JSON for compatibility
        """
        self.instruct_json = Path(instruct_json)
        self.convert_to_json = convert_to_json
        
        # If JSONL format, convert to JSON temporarily for base class compatibility
        if (self.instruct_json.suffix == ".jsonl" or 
            self.instruct_json.name.endswith(".jsonl")):
            if convert_to_json:
                # Create temporary JSON file
                json_path = self.instruct_json.with_suffix(".json")
                if not json_path.exists():
                    convert_jsonl_to_json(self.instruct_json, json_path)
                # Use JSON file for base class
                super().__init__(
                    json_path,
                    image_dir,
                    image_transform,
                    tokenizer,
                    prompt_builder_fn,
                )
                self._temp_json = json_path
            else:
                # Load JSONL directly and override base class loading
                super().__init__(
                    self.instruct_json,
                    image_dir,
                    image_transform,
                    tokenizer,
                    prompt_builder_fn,
                )
                # Override examples with JSONL-loaded data
                self.examples = load_jsonl(self.instruct_json)
                self._temp_json = None
        else:
            # Regular JSON file, use base class as-is
            super().__init__(
                self.instruct_json,
                image_dir,
                image_transform,
                tokenizer,
                prompt_builder_fn,
            )
            self._temp_json = None
    
    def __del__(self):
        """Clean up temporary JSON file if created."""
        if hasattr(self, "_temp_json") and self._temp_json and self._temp_json.exists():
            # Optionally remove temp file (comment out to keep for debugging)
            # self._temp_json.unlink()
            pass
    
    def extract_reasoning_trace(self, idx: int) -> str:
        """
        Extract reasoning trace from a dataset example.
        
        For LLaVA-CoT dataset, the assistant response contains structured reasoning steps
        with tags: <SUMMARY>, <CAPTION>, <REASONING>, <CONCLUSION>
        This method extracts the reasoning portion (before the final answer).
        
        :param idx: Index of the example
        :return: Reasoning trace string, or full response if not found
        """
        conversation = self.examples[idx]["conversations"]
        # Find the assistant response (typically the last turn)
        for turn in reversed(conversation):
            if turn["from"] == "gpt" or turn["from"] == "assistant":
                response = turn["value"]
                # LLaVA-CoT uses structured tags: <SUMMARY>, <CAPTION>, <REASONING>, <CONCLUSION>
                # Extract everything except the conclusion for reasoning trace
                if "<CONCLUSION>" in response:
                    # Extract all reasoning stages before conclusion
                    conclusion_start = response.find("<CONCLUSION>")
                    reasoning_trace = response[:conclusion_start].strip()
                    return reasoning_trace if reasoning_trace else response
                # If no structured tags, return full response
                return response
        return ""

