# Fine-Tuning LLaVA-CoT-100k on Cobra VLM

This directory contains all files related to fine-tuning Cobra VLM on the LLaVA-CoT-100k dataset for enhanced reasoning capabilities.

## Files Overview

### Documentation
- **FINETUNING_IMPLEMENTATION.md** - Detailed implementation plan and architecture
- **FINETUNING_GUIDE.md** - Step-by-step usage guide with examples
- **FINETUNING_SUMMARY.md** - Quick reference summary
- **DATASET_SOURCE.md** - Information about the LLaVA-CoT-100k dataset source

### Scripts
- **prepare_llava_cot.py** - Dataset preparation, validation, and format conversion script
- **download_llava_cot.py** - Custom download script for LLaVA-CoT-100k (handles split zip files)
- **finetune_llava_cot.py** - Custom fine-tuning script that uses the custom dataset loader

### Modules
- **llava_cot_dataset.py** - Custom dataset loader with JSONL support (extends FinetuneDataset)
- **dataset_loader.py** - Dataset loader factory for fine-tuning
- **dataset_config.py** - Custom dataset configuration
- **__init__.py** - Package initialization

## Quick Start

**Yes! Just run 2 scripts from `fine_tune/`:**

1. **Prepare the dataset** (downloads train.jsonl + images):
   ```bash
   python fine_tune/prepare_llava_cot.py --dataset_root data
   ```

2. **Fine-tune the model**:
   ```bash
   torchrun --standalone --nnodes 1 --nproc-per-node <N_GPUS> \
     fine_tune/finetune_llava_cot.py \
     --model.type cobra-3b \
     --stage finetune \
     --pretrained_checkpoint <BASELINE_CHECKPOINT> \
     --dataset_root data
   ```

**That's it!** All code is self-contained in `fine_tune/` - no modifications to `cobra/` needed.

For more details, see:
- **[QUICKSTART.md](QUICKSTART.md)** - Simple step-by-step guide
- **[FINETUNING_GUIDE.md](FINETUNING_GUIDE.md)** - Detailed instructions and examples

## Dataset Source

- **GitHub Repository**: [PKU-YuanGroup/LLaVA-CoT](https://github.com/PKU-YuanGroup/LLaVA-CoT)
- **HuggingFace Dataset**: [Xkev/LLaVA-CoT-100k](https://huggingface.co/datasets/Xkev/LLaVA-CoT-100k)

For more information, see [DATASET_SOURCE.md](DATASET_SOURCE.md).

## Implementation Details

The fine-tuning implementation uses custom modules in `fine_tune/` that extend the Cobra VLM codebase without modifying core files:
- **Custom dataset loader**: `fine_tune/llava_cot_dataset.py` - Extends `FinetuneDataset` with JSONL support
- **Custom download**: `fine_tune/download_llava_cot.py` - Handles split zip files and JSONL download
- **Dataset factory**: `fine_tune/dataset_loader.py` - Provides dataset and collator for fine-tuning
- **Dataset config**: `fine_tune/dataset_config.py` - Custom configuration for LLaVA-CoT-100k

All fine-tuning code is self-contained in the `fine_tune/` directory and does not modify core `cobra/` modules.

See [FINETUNING_IMPLEMENTATION.md](FINETUNING_IMPLEMENTATION.md) for full implementation details.

## Architecture

All fine-tuning code is self-contained in `fine_tune/` and does **not** modify core `cobra/` modules:

- **Custom Dataset**: `llava_cot_dataset.py` extends `FinetuneDataset` to handle JSONL format
- **Custom Download**: `download_llava_cot.py` handles split zip files independently
- **Custom Training**: `finetune_llava_cot.py` uses custom dataset loader without modifying core training code
- **No Core Modifications**: All functionality is provided through extension/wrapper classes

## Related Files in Main Directory

- `scripts/pretrain.py` - Main training script (can be used with custom dataset config)
- `cobra/` - Core Cobra VLM modules (unchanged by fine-tuning code)

## Citation

If you use this fine-tuning implementation, please cite:

```bibtex
@InProceedings{Xu_2025_ICCV,
    author    = {Xu, Guowei and Jin, Peng and Wu, Ziang and Li, Hao and Song, Yibing and Sun, Lichao and Yuan, Li},
    title     = {LLaVA-CoT: Let Vision Language Models Reason Step-by-Step},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {2087-2098}
}
```

