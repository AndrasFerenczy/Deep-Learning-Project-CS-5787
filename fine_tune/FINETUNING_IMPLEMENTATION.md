# Fine-Tuning Implementation Plan for LLaVA-CoT-100k

## Overview

This document outlines the implementation plan for fine-tuning Cobra VLM on the LLaVA-CoT-100k dataset to enhance reasoning capabilities. The implementation follows the methodology described in the Final Project Milestone document (Section 3.1).

## Objectives

1. **Reasoning-Enhanced Variant**: Fine-tune baseline Cobra to predict both intermediate reasoning steps and final answers
2. **Synergistic Effect**: Evaluate whether fine-tuned models produce better reasoning traces when used with the two-pass scratchpad approach
3. **Comparative Analysis**: Compare baseline Cobra, scratchpad-prompted variants, and fine-tuned models

## Implementation Components

### 1. Dataset Structure

The LLaVA-CoT-100k dataset contains 100,000 visual question-answering samples with structured reasoning annotations. Each sample includes:
- **Image**: Visual input
- **Question**: User query about the image
- **Reasoning Chain**: Structured intermediate reasoning steps (visual interpretation, logical reasoning)
- **Final Answer**: Conclusion/answer

Expected JSON format (similar to LLaVA format, with structured reasoning tags):
```json
{
  "id": "unique_id",
  "image": "path/to/image.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nQuestion about the image"
    },
    {
      "from": "gpt",
      "value": "<SUMMARY>Problem summary...</SUMMARY>\n<CAPTION>Image description...</CAPTION>\n<REASONING>Step-by-step reasoning...</REASONING>\n<CONCLUSION>Final answer...</CONCLUSION>"
    }
  ]
}
```

The LLaVA-CoT format uses structured tags:
- `<SUMMARY>`: Problem understanding and approach
- `<CAPTION>`: Visual interpretation from the image
- `<REASONING>`: Step-by-step logical reasoning
- `<CONCLUSION>`: Final answer/conclusion

Reference: [PKU-YuanGroup/LLaVA-CoT](https://github.com/PKU-YuanGroup/LLaVA-CoT)

### 2. Dataset Configuration

**File**: `cobra/conf/datasets.py`

Add a new dataset configuration class:
- `LLaVa_CoT_100k_Config`: Configures paths to LLaVA-CoT-100k dataset
- Register in `DatasetRegistry` enum

### 3. Dataset Loader

**File**: `cobra/preprocessing/datasets/datasets.py`

The existing `FinetuneDataset` class can handle LLaVA-CoT format since it follows the same conversation structure. However, we may want to add:

- **Optional reasoning trace extraction**: Helper methods to parse reasoning steps from the assistant response
- **Format validation**: Ensure reasoning structure is present

### 4. Dataset Download & Preprocessing

**File**: `cobra/preprocessing/download.py`

Add LLaVA-CoT-100k to `DATASET_REGISTRY`:
- Download JSON annotation file
- Download associated images (if separate)
- Handle any format conversions needed

**File**: `scripts/preprocess.py`

Add support for `llava-cot-100k` dataset ID.

### 5. Training Configuration

**File**: `cobra/conf/models.py`

The existing model configurations can be reused. Fine-tuning will use:
- `finetune_*` hyperparameters from model config
- Stage: `"finetune"` or `"full-finetune"`

### 6. Training Script

**File**: `scripts/pretrain.py`

The existing `pretrain.py` script supports fine-tuning. Usage:
```bash
torchrun --standalone --nnodes 1 --nproc-per-node <N_GPUS> \
  scripts/pretrain.py \
  --dataset.type llava-cot-100k \
  --model.type <MODEL_ID> \
  --stage finetune \
  --pretrained_checkpoint <PATH_TO_BASELINE_CHECKPOINT>
```

## Implementation Steps

### Step 1: Dataset Configuration
1. Add `LLaVa_CoT_100k_Config` class to `cobra/conf/datasets.py`
2. Register in `DatasetRegistry` enum
3. Configure paths for annotation JSON and image directory

### Step 2: Dataset Download Support
1. Add LLaVA-CoT-100k entry to `DATASET_REGISTRY` in `download.py`
2. Add dataset ID handling in `preprocess.py`
3. Test download and extraction

### Step 3: Dataset Format Validation
1. Verify JSON format matches expected structure
2. Test dataset loading with `FinetuneDataset`
3. Ensure reasoning traces are properly formatted in conversations

### Step 4: Fine-Tuning Execution
1. Load baseline Cobra checkpoint
2. Run fine-tuning with LLaVA-CoT-100k dataset
3. Save fine-tuned checkpoint

### Step 5: Evaluation
1. Evaluate fine-tuned model on COCO Caption dataset
2. Compare with baseline and scratchpad variants
3. Measure BLEU scores and semantic similarity

## Expected Dataset Format

Based on the LLaVA-CoT paper (arXiv:2411.10440), the dataset should contain structured reasoning chains. The format should be compatible with the existing `FinetuneDataset` class, which expects:

```python
{
  "id": str,
  "image": str,  # Path relative to image directory
  "conversations": [
    {"from": "human", "value": str},  # Question with <image> token
    {"from": "gpt", "value": str}     # Reasoning + Answer
  ]
}
```

The reasoning chain in the "gpt" response should include:
- Visual interpretation steps
- Logical reasoning steps  
- Final conclusion/answer

## Training Hyperparameters

Use the existing `finetune_*` parameters from model configs:
- Learning rate: Typically lower than align stage (e.g., 2e-5)
- Batch size: As configured in model config
- Epochs: 1-3 epochs (100k samples)
- Mixed precision: bfloat16
- Gradient checkpointing: Enabled

## Evaluation Protocol

After fine-tuning, evaluate on COCO Caption dataset:
1. **Baseline Cobra**: Direct caption generation
2. **Baseline + Scratchpad**: Two-pass reasoning
3. **Fine-tuned Cobra**: Direct caption generation
4. **Fine-tuned + Scratchpad**: Two-pass reasoning with learned patterns

Metrics:
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- Embedding-based semantic similarity (cosine similarity with Sentence-BERT/CLIP)

## Notes

- The LLaVA-CoT dataset may need to be downloaded from the official repository or HuggingFace
- Image paths in the dataset may need adjustment based on actual directory structure
- Consider creating a preprocessing script to convert LLaVA-CoT format if it differs from expected structure
- Monitor training loss to ensure model is learning reasoning patterns

## References

- **LLaVA-CoT Repository**: [PKU-YuanGroup/LLaVA-CoT](https://github.com/PKU-YuanGroup/LLaVA-CoT)
- **LLaVA-CoT Dataset**: [Xkev/LLaVA-CoT-100k on HuggingFace](https://huggingface.co/datasets/Xkev/LLaVA-CoT-100k)
- **LLaVA-CoT Paper**: Xu et al. (2025). "LLaVA-CoT: Let Vision Language Models Reason Step-by-Step." ICCV 2025. arXiv:2411.10440
- **Cobra VLM**: Original implementation
- **Final Project Milestone**: Section 3.1 - Fine-tuning with Structured Reasoning Data

