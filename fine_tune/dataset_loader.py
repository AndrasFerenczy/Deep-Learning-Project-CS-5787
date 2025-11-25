"""
dataset_loader.py

Custom dataset loader factory for LLaVA-CoT-100k that uses the custom dataset class
instead of modifying the core cobra preprocessing module.
"""
from pathlib import Path
from typing import Tuple, Type

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from cobra.models.backbones.llm.prompting import PromptBuilder
from cobra.models.backbones.vision import ImageTransform
from cobra.preprocessing.datasets import AlignDataset
from cobra.util.data_utils import PaddedCollatorForLanguageModeling

from fine_tune.llava_cot_dataset import LLaVACoTDataset


def get_llava_cot_dataset_and_collator(
    stage: str,
    dataset_root_dir: Path,
    annotation_file: Path,
    image_dir: Path,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
) -> Tuple[Dataset, PaddedCollatorForLanguageModeling]:
    """
    Get dataset and collator for LLaVA-CoT-100k fine-tuning.
    
    This function creates the appropriate dataset (custom LLaVA-CoT dataset for finetune,
    or standard dataset for align) and returns it with a collator.
    
    :param stage: Training stage ("align" or "finetune")
    :param dataset_root_dir: Root directory for datasets
    :param annotation_file: Path to annotation file (JSON or JSONL)
    :param image_dir: Directory containing images
    :param image_transform: Image transformation pipeline
    :param tokenizer: Text tokenizer
    :param prompt_builder_fn: Prompt builder class
    :param default_image_resolution: Default image resolution (H, W, C)
    :param padding_side: Padding side for tokenizer
    :return: Tuple of (dataset, collator)
    """
    collator = PaddedCollatorForLanguageModeling(
        tokenizer.model_max_length, tokenizer.pad_token_id, default_image_resolution, padding_side=padding_side
    )
    
    annotation_path = dataset_root_dir / annotation_file
    image_path = dataset_root_dir / image_dir
    
    if stage == "align":
        # Use standard align dataset
        dataset = AlignDataset(
            annotation_path,
            image_path,
            image_transform,
            tokenizer,
        )
        return dataset, collator
    
    elif stage in ("finetune", "full-finetune"):
        # Use custom LLaVA-CoT dataset that handles JSONL
        dataset = LLaVACoTDataset(
            annotation_path,
            image_path,
            image_transform,
            tokenizer,
            prompt_builder_fn=prompt_builder_fn,
            convert_to_json=True,  # Convert JSONL to JSON for compatibility
        )
        return dataset, collator
    
    else:
        raise ValueError(f"Stage `{stage}` is not supported!")

