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
        choices=["baseline", "scratchpad", "scratchpad_compare", "llava_cot", "both", "all"],
        default="baseline",
        help="Inference method to use (both=baseline+scratchpad, all=baseline+scratchpad+llava_cot, scratchpad_compare=compare multiple pass counts)"
    )
    
    # Dataset settings
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
        help="Number of reasoning passes for scratchpad and llava_cot methods. For scratchpad_compare, this is the maximum number of passes to compare (default: compares 1-4 passes)"
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

    parser.add_argument(
        "--no_baseline_cache",
        action="store_true",
        help="Do not use the default cached baseline output (cobra_eval/data/baseline_caption_COCO_output.json)"
    )
    
    # Checkpointing
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=10,
        help="Save checkpoint every N samples (0 to disable checkpointing, default: 10)"
    )
    
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Automatically resume from latest checkpoint if available"
    )
    
    # Parallelization
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers for inference (default: 1, sequential processing)"
    )
    
    return parser.parse_args()

