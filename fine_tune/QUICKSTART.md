# Quick Start Guide

Yes! You only need to run 2 scripts from the `fine_tune/` directory:

## Step 1: Prepare Dataset

Download and prepare the LLaVA-CoT-100k dataset (downloads train.jsonl and all image zip parts):

```bash
python fine_tune/prepare_llava_cot.py --dataset_root data
```

This will:
- Download `train.jsonl` from HuggingFace
- Download 16 split zip files (`image.zip.part-aa` through `image.zip.part-ap`)
- Merge the zip parts into `image.zip`
- Extract the images
- Validate the dataset format

## Step 2: Fine-Tune

Run the fine-tuning script:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node <N_GPUS> \
  fine_tune/finetune_llava_cot.py \
  --model.type cobra-3b \
  --stage finetune \
  --pretrained_checkpoint <PATH_TO_BASELINE_CHECKPOINT> \
  --dataset_root data
```

### Example with 4 GPUs:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 4 \
  fine_tune/finetune_llava_cot.py \
  --model.type cobra-3b \
  --stage finetune \
  --pretrained_checkpoint runs/cobra-3b+stage-finetune+x7/checkpoints/checkpoint-latest.pt \
  --dataset_root data
```

## That's It!

All the custom code is in `fine_tune/` - no modifications to `cobra/` needed.

## Optional: Customize Training

You can override hyperparameters:

```bash
torchrun --standalone --nnodes 1 --nproc-per-node 4 \
  fine_tune/finetune_llava_cot.py \
  --model.type cobra-3b \
  --model.finetune_learning_rate 2e-5 \
  --model.finetune_epochs 3 \
  --stage finetune \
  --pretrained_checkpoint <CHECKPOINT> \
  --dataset_root data
```

## Files You Need

- ✅ `fine_tune/prepare_llava_cot.py` - Run once to download dataset
- ✅ `fine_tune/finetune_llava_cot.py` - Run to start fine-tuning
- ✅ All other files in `fine_tune/` are supporting modules (automatically imported)

## What Gets Created

- Dataset: `data/download/llava-cot-100k/train.jsonl` and `data/download/llava-cot-100k/images/`
- Training logs: `runs/llava-cot-100k+<model>+stage-finetune+x<seed>/`
- Checkpoints: `runs/llava-cot-100k+<model>+stage-finetune+x<seed>/checkpoints/`

## Troubleshooting

**Dataset not found?**
- Make sure you ran `prepare_llava_cot.py` first
- Check that `data/download/llava-cot-100k/train.jsonl` exists

**Import errors?**
- Make sure you're running from the project root directory
- The `fine_tune/` directory should be importable as a package

**Out of memory?**
- Reduce `--model.finetune_per_device_batch_size` (e.g., `--model.finetune_per_device_batch_size 1`)

