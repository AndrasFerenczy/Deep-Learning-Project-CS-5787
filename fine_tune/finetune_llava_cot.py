"""
finetune_llava_cot.py

Custom fine-tuning script for LLaVA-CoT-100k that uses the custom dataset loader
without modifying the core cobra training infrastructure.
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import draccus
import torch
import torch.distributed as dist
import yaml

from cobra.conf import ModelConfig, ModelRegistry
from cobra.models import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform, get_vlm
from cobra.overwatch import initialize_overwatch
from cobra.training import Metrics, get_train_strategy
from cobra.util import set_global_seed

from fine_tune.dataset_config import get_llava_cot_config
from fine_tune.dataset_loader import get_llava_cot_dataset_and_collator

# Disable Tokenizers Parallelism to Play Nice w/ PyTorch Multiprocessing DataLoaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


@dataclass
class FinetuneLLaVACoTConfig:
    # fmt: off
    # ModelConfig (`cobra/conf/models.py`); override with --model.type `ModelRegistry.<MODEL>.model_id`
    model: ModelConfig = field(
        default_factory=ModelConfig.get_choice_class(ModelRegistry.COBRA_3B.model_id)
    )

    # Pretraining Stage
    stage: str = "finetune"                                         # Pretraining Stage (finetune or full-finetune)
    pretrained_checkpoint: Optional[Path] = None                    # Pretrained Checkpoint to Load
                                                                    #   if None =>> will match on (run_dir / `align`)

    # Run Arguments
    run_id: Optional[str] = None                                    # Run ID for logging, Weights & Biases
    run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
    seed: int = 7                                                   # Random seed (for reproducibility)
    dataset_root: Path = Path("data")                               # Root directory for datasets

    # HF Hub Credentials (for any gated models)
    hf_token: str | Path = Path(".hf_token")                        # Environment variable or Path to HF Token

    # Tracking Parameters
    trackers: Tuple[str, ...] = ("jsonl", "wandb")                  # Trackers to initialize (if W&B, add config!)
    wandb_project: str = "cobra-cot"                                # Name of W&B project
    wandb_entity: Optional[str] = None                              # Name of W&B entity (default: None)

    def __post_init__(self) -> None:
        """Set optimization parameters based on `stage`."""
        if self.stage.endswith("finetune"):
            self.epochs = self.model.finetune_epochs
            self.max_steps = self.model.finetune_max_steps
            self.global_batch_size = self.model.finetune_global_batch_size
            self.per_device_batch_size = self.model.finetune_per_device_batch_size

            self.learning_rate = self.model.finetune_learning_rate
            self.weight_decay = self.model.finetune_weight_decay
            self.max_grad_norm = self.model.finetune_max_grad_norm
            self.lr_scheduler_type = self.model.finetune_lr_scheduler_type
            self.warmup_ratio = self.model.finetune_warmup_ratio

            self.train_strategy = self.model.finetune_train_strategy
        else:
            raise ValueError(f"Stage `{self.stage}` is not supported!")

    # fmt: on


@draccus.wrap()
def finetune_llava_cot(cfg: FinetuneLLaVACoTConfig) -> None:
    """Main fine-tuning function for LLaVA-CoT-100k."""
    overwatch.info("Cobra VLM Fine-Tuning on LLaVA-CoT-100k :: Gathering Light")

    # Note => Under `torchrun` initializing `overwatch` will automatically set up `torch.distributed`
    torch.cuda.set_device(device_id := (overwatch.rank() % torch.cuda.device_count()))
    torch.cuda.empty_cache()

    # Create Unique Run Name & Save Directory
    model_id = cfg.model.model_id
    cfg.run_id = f"llava-cot-100k+{model_id}+stage-{cfg.stage}+x{cfg.seed}" if cfg.run_id is None else cfg.run_id

    # Start =>> Build Directories and Set Randomness
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    os.makedirs(run_dir := (cfg.run_root_dir / cfg.run_id), exist_ok=True)
    os.makedirs(cfg.run_root_dir / cfg.run_id / "checkpoints", exist_ok=True)
    if overwatch.is_rank_zero():
        # Additionally save a JSON version of the config
        draccus.dump(cfg, open(run_dir / "config.yaml", "w"))
        with open(run_dir / "config.yaml", "r") as f_yaml, open(run_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    # Load Vision Backbone --> on CPU, in Full Precision
    overwatch.info(f"Loading Vision Backbone [bold]{cfg.model.vision_backbone_id}[/] via TIMM ")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        cfg.model.vision_backbone_id, image_resize_strategy=cfg.model.image_resize_strategy
    )

    # Load LLM Backbone --> on CPU, in Full Precision
    overwatch.info(f"Loading Pretrained LLM [bold]{cfg.model.llm_backbone_id}[/] via HF Transformers")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        cfg.model.llm_backbone_id, llm_max_length=cfg.model.llm_max_length, hf_token=hf_token
    )

    # Create VLM => wraps `vision_backbone` and `llm`
    overwatch.info(f"Instantiating CobraVLM `{model_id}` for Training Stage = `{cfg.stage}`")
    vlm = get_vlm(
        model_id,
        cfg.model.arch_specifier,
        vision_backbone,
        llm_backbone,
        enable_mixed_precision_training=cfg.model.enable_mixed_precision_training,
    )

    # [Explicit] Call to `freeze_backbones` here for clarity
    overwatch.info(f"Invoking `VLM.freeze_backbones()` for `{model_id}` => Training Stage: `{cfg.stage}`")
    vlm.freeze_backbones(cfg.stage)

    # Load Weights from Checkpoint
    overwatch.info(f"Invoking `VLM.load_checkpoint()` for `{model_id}` => Training Stage: `{cfg.stage}`")
    vlm.load_from_checkpoint(cfg.stage, run_dir, pretrained_checkpoint=cfg.pretrained_checkpoint)

    # Get Custom Dataset for LLaVA-CoT-100k
    overwatch.info("Creating LLaVA-CoT-100k Dataset => Stage: `finetune`")
    dataset_cfg = get_llava_cot_config(cfg.dataset_root)
    
    train_dataset, collator = get_llava_cot_dataset_and_collator(
        cfg.stage,
        dataset_cfg.dataset_root_dir,
        dataset_cfg.finetune_stage_components[0],  # train.jsonl
        dataset_cfg.finetune_stage_components[1],  # image directory
        image_transform,
        tokenizer,
        prompt_builder_fn=llm_backbone.prompt_builder_fn,
        default_image_resolution=vision_backbone.default_image_resolution,
        padding_side=tokenizer.padding_side,
    )

    # Create Train Strategy
    overwatch.info(f"Initializing Train Strategy `{cfg.train_strategy}`")
    train_strategy = get_train_strategy(
        train_strategy=cfg.train_strategy,
        vlm=vlm,
        device_id=device_id,
        epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        global_batch_size=cfg.global_batch_size,
        per_device_batch_size=cfg.per_device_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        enable_gradient_checkpointing=cfg.model.enable_gradient_checkpointing,
        enable_mixed_precision_training=cfg.model.enable_mixed_precision_training,
        reduce_in_full_precision=cfg.model.reduce_in_full_precision,
        worker_init_fn=worker_init_fn,
    )
    train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(train_dataset))

    # Create Metrics =>> Handles on the fly tracking, logging to specified trackers
    overwatch.info(f"Creating Metrics with Active Trackers => `{cfg.trackers}`")
    metrics = Metrics(
        cfg.trackers,
        cfg.run_id,
        run_dir,
        draccus.encode(cfg),
        cfg.stage,
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
        grad_accumulation_steps=train_strategy.grad_accumulation_steps,
    )

    # Run Training
    overwatch.info("Starting Training Loop")
    train_strategy.run_training(train_dataset, collator, metrics, stage=cfg.stage, seed=cfg.seed)

    # Finalize
    overwatch.info("Done with Training =>> Finalizing Metrics")
    metrics.finalize()

    # And... we're done!
    overwatch.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    finetune_llava_cot()

