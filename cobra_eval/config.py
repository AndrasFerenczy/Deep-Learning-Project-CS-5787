"""
Configuration and CLI argument parsing.
"""
import argparse
from typing import Dict, Any

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cobra VLM Benchmark System")
    
    # Method selection
    parser.add_argument(
        "--method",
        type=str,
        choices=["baseline", "scratchpad", "both"],
        default="baseline",
        help="Inference method to use"
    )
    
    # Dataset settings
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["coco", "mmstar"],
        default="coco",
        help="Dataset to use for evaluation"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--no_streaming",
        action="store_true",
        help="Disable streaming mode for dataset"
    )
    
    # Scratchpad settings
    parser.add_argument(
        "--scratchpad_passes",
        type=int,
        default=1,
        help="Number of reasoning passes for scratchpad method"
    )
    
    # Model settings
    parser.add_argument(
        "--model_id",
        type=str,
        default="cobra+3b",
        help="Model ID to load"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=".hf_token",
        help="Path to HuggingFace token file"
    )
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--reasoning_max_tokens", type=int, default=256)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    
    # System
    parser.add_argument(
        "--clear_cache",
        action="store_true",
        help="Clear GPU cache before loading model"
    )
    parser.add_argument(
        "--min_free_gb",
        type=float,
        default=8.0,
        help="Minimum free GPU memory required in GB"
    )
    
    # Input/Resume
    parser.add_argument(
        "--load_results",
        type=str,
        default=None,
        help="Path to existing results JSON file to load and resume/skip generation from"
    )
    
    return parser.parse_args()

