"""
dataset_config.py

Custom dataset configuration for LLaVA-CoT-100k fine-tuning.
This provides a way to use the LLaVA-CoT dataset without modifying
the core cobra configuration files.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from cobra.conf import DatasetConfig


@dataclass
class LLaVa_CoT_100k_Config_Custom(DatasetConfig):
    """
    Custom dataset configuration for LLaVA-CoT-100k.
    
    This extends the base DatasetConfig to provide LLaVA-CoT-100k specific
    paths without modifying the core cobra configuration.
    """
    dataset_id: str = "llava-cot-100k"

    # For fine-tuning on reasoning data, we use the same align stage as LLaVA-v15
    # (projection alignment) and then fine-tune on LLaVA-CoT-100k
    align_stage_components: Tuple[Path, Path] = (
        Path("download/llava-laion-cc-sbu-558k/chat.json"),
        Path("download/llava-laion-cc-sbu-558k/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("download/llava-cot-100k/train.jsonl"),  # JSONL format from HuggingFace
        Path("download/llava-cot-100k/"),
    )
    dataset_root_dir: Path = Path("data")


def get_llava_cot_config(dataset_root: Path = Path("data")) -> LLaVa_CoT_100k_Config_Custom:
    """
    Get LLaVA-CoT-100k dataset configuration.
    
    :param dataset_root: Root directory for datasets
    :return: Dataset configuration object
    """
    config = LLaVa_CoT_100k_Config_Custom()
    config.dataset_root_dir = dataset_root
    return config

