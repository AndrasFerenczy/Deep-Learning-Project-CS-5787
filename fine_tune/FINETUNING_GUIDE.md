# Fine-Tuning Guide: LLaVA-CoT-100k on Cobra VLM

This guide provides step-by-step instructions for fine-tuning Cobra VLM on the LLaVA-CoT-100k dataset to enhance reasoning capabilities.

## Overview

Fine-tuning on LLaVA-CoT-100k teaches the model to generate structured reasoning traces before producing final answers. This complements the two-pass scratchpad approach by providing learned reasoning patterns.

## Prerequisites

1. **Baseline Cobra Model**: You need a trained baseline Cobra checkpoint (from align + finetune stages)
2. **LLaVA-CoT-100k Dataset**: 100,000 samples with structured reasoning annotations
3. **Hardware**: Multi-GPU setup recommended (FSDP training)

## Step 1: Prepare the Dataset

### Option A: Automatic Download (if dataset URL is available)

```bash
# Download and prepare LLaVA-CoT-100k dataset
python fine_tune/prepare_llava_cot.py --dataset_root data
```

### Option B: Manual Setup

1. Download LLaVA-CoT-100k dataset from the official repository:
   - **GitHub**: [PKU-YuanGroup/LLaVA-CoT](https://github.com/PKU-YuanGroup/LLaVA-CoT)
   - **HuggingFace Dataset**: [Xkev/LLaVA-CoT-100k](https://huggingface.co/datasets/Xkev/LLaVA-CoT-100k)

2. Place the dataset in the expected location:
   ```bash
   mkdir -p data/download/llava-cot-100k
   # Copy llava_cot_100k.json to data/download/llava-cot-100k/
   # Ensure images are accessible (may need to link/copy from other datasets)
   ```

3. Validate the format:
   ```bash
   python fine_tune/prepare_llava_cot.py --dataset_root data --download_dataset false
   ```

### Dataset Format Requirements

The dataset JSON should follow this structure (LLaVA-CoT format with structured reasoning tags):
```json
[
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
]
```

The LLaVA-CoT format uses structured tags:
- `<SUMMARY>`: Problem understanding and approach
- `<CAPTION>`: Visual interpretation from the image  
- `<REASONING>`: Step-by-step logical reasoning
- `<CONCLUSION>`: Final answer/conclusion

## Step 2: Fine-Tune the Model

### Basic Fine-Tuning Command

```bash
torchrun --standalone --nnodes 1 --nproc-per-node <N_GPUS> \
  scripts/pretrain.py \
  --dataset.type llava-cot-100k \
  --model.type <MODEL_ID> \
  --stage finetune \
  --pretrained_checkpoint <PATH_TO_BASELINE_CHECKPOINT> \
  --run_id cobra-cot-100k-finetune \
  --seed 7
```

### Example: Fine-tuning Cobra 3B

```bash
# Assuming you have a baseline checkpoint at runs/cobra-3b+stage-finetune+x7/checkpoints/checkpoint-<step>.pt
torchrun --standalone --nnodes 1 --nproc-per-node 4 \
  scripts/pretrain.py \
  --dataset.type llava-cot-100k \
  --model.type cobra-3b \
  --stage finetune \
  --pretrained_checkpoint runs/cobra-3b+stage-finetune+x7/checkpoints/checkpoint-latest.pt \
  --run_id cobra-3b-cot-100k-finetune \
  --seed 7
```

### Hyperparameter Customization

You can override model hyperparameters:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node <N_GPUS> \
  scripts/pretrain.py \
  --dataset.type llava-cot-100k \
  --model.type cobra-3b \
  --model.finetune_learning_rate 2e-5 \
  --model.finetune_epochs 3 \
  --model.finetune_global_batch_size 128 \
  --stage finetune \
  --pretrained_checkpoint <PATH_TO_CHECKPOINT>
```

### Full Fine-Tune (All Parameters)

To fine-tune all parameters including vision backbone:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node <N_GPUS> \
  scripts/pretrain.py \
  --dataset.type llava-cot-100k \
  --model.type cobra-3b \
  --stage full-finetune \
  --pretrained_checkpoint <PATH_TO_CHECKPOINT>
```

## Step 3: Monitor Training

Training logs are saved to:
- `runs/<run_id>/logs/` - JSONL logs
- `runs/<run_id>/checkpoints/` - Model checkpoints
- Weights & Biases (if configured)

Monitor training progress:
```bash
# Watch log file
tail -f runs/<run_id>/logs/train.jsonl

# Or use W&B dashboard if configured
```

## Step 4: Evaluate Fine-Tuned Model

After fine-tuning, evaluate on COCO Caption dataset:

```bash
# Evaluate baseline model
python benchmark.py \
  --checkpoint <BASELINE_CHECKPOINT> \
  --output results/baseline_bleu.txt

# Evaluate fine-tuned model
python benchmark.py \
  --checkpoint runs/<run_id>/checkpoints/checkpoint-latest.pt \
  --output results/cot_finetuned_bleu.txt

# Evaluate fine-tuned model with scratchpad
python benchmark_scratchpad.py \
  --checkpoint runs/<run_id>/checkpoints/checkpoint-latest.pt \
  --output results/cot_finetuned_scratchpad_bleu.txt
```

## Expected Results

Based on the methodology in the milestone document:

1. **Baseline Cobra**: BLEU-4 ≈ 0.13 (from preliminary results)
2. **Baseline + Scratchpad**: Expected improvement from two-pass reasoning
3. **Fine-tuned Cobra**: Expected improvement from learned reasoning patterns
4. **Fine-tuned + Scratchpad**: Best performance (synergistic effect)

## Troubleshooting

### Dataset Not Found
- Verify dataset path: `data/download/llava-cot-100k/llava_cot_100k.json`
- Check image paths in JSON match actual image locations
- Run validation: `python fine_tune/prepare_llava_cot.py --validate_format`

### Out of Memory
- Reduce `--model.finetune_per_device_batch_size`
- Increase gradient accumulation (automatically computed from global batch size)
- Enable gradient checkpointing (default: enabled)

### Training Instability
- Lower learning rate: `--model.finetune_learning_rate 1e-5`
- Reduce batch size
- Check for NaN values in logs

### Image Path Issues
- LLaVA-CoT images may reference paths from other datasets
- Create symlinks or copy images to expected locations
- Update image paths in JSON if needed

## Next Steps

After fine-tuning:

1. **Evaluate on COCO**: Measure BLEU scores and semantic similarity
2. **Compare Variants**: Baseline vs. Fine-tuned vs. Scratchpad combinations
3. **Analyze Reasoning Traces**: Examine quality of generated reasoning steps
4. **Stretch Goal**: Consider GRPO for further enhancement (see milestone Section 3.2)

## References

- **LLaVA-CoT Repository**: [PKU-YuanGroup/LLaVA-CoT](https://github.com/PKU-YuanGroup/LLaVA-CoT)
- **LLaVA-CoT Dataset**: [Xkev/LLaVA-CoT-100k on HuggingFace](https://huggingface.co/datasets/Xkev/LLaVA-CoT-100k)
- **LLaVA-CoT Paper**: Xu et al. (2025). "LLaVA-CoT: Let Vision Language Models Reason Step-by-Step." ICCV 2025. arXiv:2411.10440
- **Implementation Plan**: See `FINETUNING_IMPLEMENTATION.md`
- **Milestone Document**: `Final Project Milestone.md` Section 3.1

## Configuration Files

- Dataset Config: `cobra/conf/datasets.py` - `LLaVa_CoT_100k_Config`
- Model Config: `cobra/conf/models.py` - Use existing model configs
- Training Script: `scripts/pretrain.py`
- Dataset Loader: `cobra/preprocessing/datasets/datasets.py` - `FinetuneDataset`

