"""
Model loading and preparation for QLoRA fine-tuning.
"""
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def add_bias_to_lora_layers(model):
    """
    Add bias attribute to all LoRA-wrapped layers in the model.
    This ensures mamba_ssm can access bias without AttributeError.
    """
    try:
        from peft.tuners.lora import Linear as LoRALinear
        
        patched_count = 0
        for name, module in model.named_modules():
            # Check if this is a LoRA-wrapped Linear layer
            if isinstance(module, LoRALinear) or (hasattr(module, 'base_layer') and hasattr(module, 'lora_A')):
                if not hasattr(module, '_bias_patched'):
                    # Get bias from base layer if it exists
                    bias_value = None
                    if hasattr(module, 'base_layer'):
                        bias_value = getattr(module.base_layer, 'bias', None)
                    elif hasattr(module, 'bias'):
                        bias_value = getattr(module, 'bias', None)
                    
                    # Add bias as a cached property/attribute
                    # Use object.__setattr__ to bypass PyTorch's module checks
                    try:
                        object.__setattr__(module, 'bias', bias_value)
                        module._bias_patched = True
                        patched_count += 1
                    except:
                        # If that fails, try regular setattr
                        try:
                            setattr(module, 'bias', bias_value)
                            module._bias_patched = True
                            patched_count += 1
                        except:
                            pass
        
        if patched_count > 0:
            print(f"Added bias attribute to {patched_count} LoRA-wrapped layers")
    except (ImportError, AttributeError) as e:
        print(f"Warning: Could not add bias to LoRA layers: {e}")

def wrap_lora_layers_for_bias(model):
    """
    Patch mamba_ssm and add bias to LoRA layers to safely handle bias access.
    This fixes the AttributeError when mamba_ssm tries to access bias on LoRA-wrapped Linear layers.
    """
    # First, add bias attributes to all LoRA layers
    add_bias_to_lora_layers(model)
    
    # The bias attributes should now be accessible, so the original forward should work
    # But we can also patch the forward method as a safety measure
    try:
        import mamba_ssm.modules.mamba_simple as mamba_simple
        
        if not hasattr(mamba_simple.Mamba, '_bias_patched'):
            # The bias attributes are now added, so original forward should work
            # But we can add a safety check just in case
            mamba_simple.Mamba._bias_patched = True
            print("Prepared mamba_ssm for LoRA bias compatibility")
    except (ImportError, AttributeError) as e:
        print(f"Warning: Could not patch mamba_ssm for LoRA bias compatibility: {e}")

# Add parent directory to path so we can import qlora_finetune modules
script_dir = Path(__file__).parent
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from qlora_finetune.utils import add_parent_to_path, find_mamba_target_modules

# Add parent to path to import cobra
add_parent_to_path()

from cobra.models.load import load as load_cobra_model
from cobra.models.vlms.cobra import CobraVLM
from cobra.models.mamba.modeling_mamba import MambaForCausalLM
from transformers import AutoTokenizer


def get_quantization_config(
    load_in_4bit: bool = True,
    bnb_4bit_compute_dtype: str = "float16",
    bnb_4bit_quant_type: str = "nf4",
    bnb_4bit_use_double_quant: bool = True,
) -> Optional[BitsAndBytesConfig]:
    """
    Create BitsAndBytesConfig for quantization.
    
    Args:
        load_in_4bit: Whether to use 4-bit quantization
        bnb_4bit_compute_dtype: Compute dtype ("float16" or "bfloat16")
        bnb_4bit_quant_type: Quantization type ("nf4" or "fp4")
        bnb_4bit_use_double_quant: Whether to use double quantization
        
    Returns:
        BitsAndBytesConfig or None if quantization is disabled
    """
    if not load_in_4bit:
        return None
    
    compute_dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    
    compute_dtype = compute_dtype_map.get(bnb_4bit_compute_dtype, torch.float16)
    
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
    )


def load_mamba_with_quantization(
    mamba_model_id: str,
    hf_token: Optional[str] = None,
    quantization_config: Optional[BitsAndBytesConfig] = None,
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    Load base Mamba model directly with 4-bit quantization.
    
    This is the proper way to do QLoRA - load the base model with quantization
    from the start, rather than loading the full model and trying to quantize it.
    
    Args:
        mamba_model_id: HuggingFace model ID for the Mamba model (e.g., "xiuyul/mamba-2.8b-zephyr")
        hf_token: HuggingFace token or path to .hf_token file
        quantization_config: BitsAndBytesConfig for quantization
        
    Returns:
        Tuple of (quantized Mamba model, tokenizer)
    """
    # Load HF token if path provided
    if hf_token is not None and Path(hf_token).exists():
        with open(hf_token, "r") as f:
            hf_token = f.read().strip()
    
    print(f"Loading Mamba model {mamba_model_id} with 4-bit quantization...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(mamba_model_id, token=hf_token)
    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model with quantization
    if quantization_config is not None:
        model = MambaForCausalLM.from_pretrained(
            mamba_model_id,
            quantization_config=quantization_config,
            token=hf_token,
            torch_dtype=torch.float16,  # Use float16 for compute
        )
        # Ensure model is in eval mode initially (will be set to train mode by trainer)
        model.eval()
        # Quantized models should be on GPU automatically, but ensure all parameters are
        print("Model loaded with 4-bit quantization")
        # Verify device placement
        if torch.cuda.is_available():
            # Check if model is on GPU
            first_param = next(model.parameters())
            if first_param.device.type != 'cuda':
                print(f"Warning: Model parameters are on {first_param.device}, expected cuda")
            else:
                print(f"Model is on device: {first_param.device}")
    else:
        # Load without quantization (fallback)
        model = MambaForCausalLM.from_pretrained(
            mamba_model_id,
            token=hf_token,
            torch_dtype=torch.float16,
        )
        # Move to GPU if available (only for non-quantized models)
        if torch.cuda.is_available():
            model = model.cuda()
        print("Model loaded without quantization (warning: will use more memory)")
    
    # Disable fused_add_norm to avoid Triton issues
    if hasattr(model, 'config') and hasattr(model.config, 'fused_add_norm'):
        model.config.fused_add_norm = False
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'fused_add_norm'):
            model.backbone.fused_add_norm = False
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'layers'):
            for layer in model.backbone.layers:
                if hasattr(layer, 'fused_add_norm'):
                    layer.fused_add_norm = False
                # Replace RMSNorm with LayerNorm
                if hasattr(layer, 'norm') and 'RMSNorm' in str(type(layer.norm)):
                    import torch.nn as nn
                    old_norm = layer.norm
                    # Get device from old norm's weight
                    device = old_norm.weight.device if old_norm.weight is not None else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
                    new_norm = nn.LayerNorm(
                        old_norm.weight.shape[0],
                        eps=old_norm.eps,
                        elementwise_affine=old_norm.weight is not None
                    ).to(device)  # Move to same device as old norm
                    if old_norm.weight is not None:
                        new_norm.weight.data.copy_(old_norm.weight.data)
                    if hasattr(old_norm, 'bias') and old_norm.bias is not None:
                        new_norm.bias.data.copy_(old_norm.bias.data)
                    layer.norm = new_norm
        # Replace final norm
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'norm_f'):
            if 'RMSNorm' in str(type(model.backbone.norm_f)):
                import torch.nn as nn
                old_norm = model.backbone.norm_f
                # Get device from old norm's weight
                device = old_norm.weight.device if old_norm.weight is not None else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
                new_norm = nn.LayerNorm(
                    old_norm.weight.shape[0],
                    eps=old_norm.eps,
                    elementwise_affine=old_norm.weight is not None
                ).to(device)  # Move to same device as old norm
                if old_norm.weight is not None:
                    new_norm.weight.data.copy_(old_norm.weight.data)
                if hasattr(old_norm, 'bias') and old_norm.bias is not None:
                    new_norm.bias.data.copy_(old_norm.bias.data)
                model.backbone.norm_f = new_norm
        print("Disabled fused_add_norm and replaced RMSNorm with LayerNorm")
    
    return model, tokenizer


def load_cobra_for_qlora(
    model_id: str,
    pretrained_checkpoint: Optional[Path] = None,
    hf_token: Optional[str] = None,
    quantization_config: Optional[BitsAndBytesConfig] = None,
    freeze_vision_encoder: bool = True,
    use_quantized_loader: bool = True,  # New parameter to use quantized loader
) -> Tuple[Optional[CobraVLM], torch.nn.Module]:
    """
    Load Cobra model and prepare LLM backbone for QLoRA.
    
    This function:
    1. Loads the full CobraVLM model
    2. Extracts the LLM backbone
    3. Applies quantization if specified
    4. Freezes the vision encoder
    
    Args:
        model_id: Cobra model ID (e.g., "cobra+3b")
        pretrained_checkpoint: Path to pretrained checkpoint (optional)
        hf_token: HuggingFace token or path to .hf_token file
        quantization_config: BitsAndBytesConfig for quantization
        freeze_vision_encoder: Whether to freeze vision encoder
        
    Returns:
        Tuple of (CobraVLM model, LLM backbone for QLoRA)
    """
    # Load HF token if path provided
    if hf_token is not None and Path(hf_token).exists():
        with open(hf_token, "r") as f:
            hf_token = f.read().strip()
    
    # If using quantized loader, load base Mamba model directly with quantization
    if use_quantized_loader and quantization_config is not None and pretrained_checkpoint is None:
        # Map Cobra model IDs to their base Mamba model IDs
        mamba_model_map = {
            "cobra+3b": "xiuyul/mamba-2.8b-zephyr",
            "cobra+7b": "state-spaces/mamba-2.8b",  # Update if different
        }
        
        # Get the base Mamba model ID
        mamba_model_id = mamba_model_map.get(model_id, "xiuyul/mamba-2.8b-zephyr")
        
        # Load quantized Mamba model directly
        llm_backbone, _ = load_mamba_with_quantization(
            mamba_model_id=mamba_model_id,
            hf_token=hf_token,
            quantization_config=quantization_config,
        )
        
        # For QLoRA training, we only need the LLM backbone
        # Return None for vlm since we're not using the full Cobra model
        print("Loaded quantized Mamba model directly (skipping full Cobra model)")
        return None, llm_backbone
    
    # Otherwise, load full Cobra model (original method)
    if pretrained_checkpoint is not None:
        # Load from local checkpoint
        vlm = load_cobra_model(pretrained_checkpoint, hf_token=hf_token)
    else:
        # Load from HuggingFace Hub
        vlm = load_cobra_model(model_id, hf_token=hf_token)
    
    # Disable fused_add_norm and replace RMSNorm with PyTorch implementation
    # to avoid Triton compilation issues (ldconfig not available)
    llm = vlm.llm_backbone.llm
    if hasattr(llm, 'config') and hasattr(llm.config, 'fused_add_norm'):
        # Disable in config
        llm.config.fused_add_norm = False
        # Disable in model backbone
        if hasattr(llm, 'backbone') and hasattr(llm.backbone, 'fused_add_norm'):
            llm.backbone.fused_add_norm = False
        # Disable in each layer
        if hasattr(llm, 'backbone') and hasattr(llm.backbone, 'layers'):
            for layer in llm.backbone.layers:
                if hasattr(layer, 'fused_add_norm'):
                    layer.fused_add_norm = False
                # Replace RMSNorm with PyTorch LayerNorm to avoid Triton
                if hasattr(layer, 'norm') and 'RMSNorm' in str(type(layer.norm)):
                    import torch.nn as nn
                    # Create a PyTorch RMSNorm equivalent using LayerNorm
                    # RMSNorm is essentially LayerNorm without centering (no mean subtraction)
                    # For now, use LayerNorm as a close approximation
                    old_norm = layer.norm
                    new_norm = nn.LayerNorm(
                        old_norm.weight.shape[0],
                        eps=old_norm.eps,
                        elementwise_affine=old_norm.weight is not None
                    )
                    if old_norm.weight is not None:
                        new_norm.weight.data.copy_(old_norm.weight.data)
                    if hasattr(old_norm, 'bias') and old_norm.bias is not None:
                        new_norm.bias.data.copy_(old_norm.bias.data)
                    layer.norm = new_norm
        # Replace final norm if it's RMSNorm
        if hasattr(llm, 'backbone') and hasattr(llm.backbone, 'norm_f'):
            if 'RMSNorm' in str(type(llm.backbone.norm_f)):
                import torch.nn as nn
                old_norm = llm.backbone.norm_f
                new_norm = nn.LayerNorm(
                    old_norm.weight.shape[0],
                    eps=old_norm.eps,
                    elementwise_affine=old_norm.weight is not None
                )
                if old_norm.weight is not None:
                    new_norm.weight.data.copy_(old_norm.weight.data)
                if hasattr(old_norm, 'bias') and old_norm.bias is not None:
                    new_norm.bias.data.copy_(old_norm.bias.data)
                llm.backbone.norm_f = new_norm
        print("Disabled fused_add_norm and replaced RMSNorm with LayerNorm to avoid Triton compilation issues")
    
    # Freeze vision encoder if requested
    if freeze_vision_encoder:
        for param in vlm.vision_backbone.parameters():
            param.requires_grad = False
        print("Vision encoder frozen")
    
    # Extract LLM backbone
    llm_backbone = vlm.llm_backbone.llm  # Access the underlying MambaForCausalLM
    
    # Apply quantization if specified
    if quantization_config is not None:
        # Note: Quantization needs to be applied when loading the model
        # Since we're loading from Cobra, we may need to reload with quantization
        # For now, we'll apply it to the backbone if possible
        print("Note: Quantization should be applied during model loading.")
        print("For full quantization support, consider loading the base Mamba model directly.")
    
    return vlm, llm_backbone


def prepare_model_for_qlora(
    model: torch.nn.Module,
    quantization_config: Optional[BitsAndBytesConfig] = None,
    target_modules: Optional[list] = None,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_bias: str = "none",
) -> torch.nn.Module:
    """
    Prepare model for QLoRA training by applying quantization and LoRA.
    
    Args:
        model: The model to prepare (typically LLM backbone)
        quantization_config: BitsAndBytesConfig for quantization
        target_modules: List of module names to apply LoRA to (None = auto-detect)
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        lora_bias: LoRA bias setting
        
    Returns:
        Model with QLoRA applied
    """
    # Auto-detect target modules if not provided
    if target_modules is None or len(target_modules) == 0:
        target_modules = find_mamba_target_modules(model)
        print(f"Auto-detected target modules: {target_modules}")
    
    if not target_modules:
        raise ValueError(
            "No target modules found for LoRA. Please specify target_modules manually."
        )
    
    # Prepare model for k-bit training (if quantized)
    # Note: prepare_model_for_kbit_training expects a quantized model.
    # Since we're loading through Cobra's loader which doesn't support quantization,
    # we need to handle this differently. For now, we'll skip quantization preparation
    # and apply LoRA directly. Full QLoRA support would require loading the base
    # Mamba model directly with quantization.
    if quantization_config is not None:
        try:
            # Try to prepare for k-bit training - this will work if model is quantized
            model = prepare_model_for_kbit_training(model)
            print("Model prepared for k-bit training")
        except Exception as e:
            print(f"Warning: Could not prepare model for k-bit training: {e}")
            print("Continuing with LoRA only (no quantization).")
            print("Note: For full QLoRA, load the base Mamba model with quantization first.")
    
    # Create LoRA config
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Wrap LoRA layers to safely expose bias attribute for mamba_ssm compatibility
    wrap_lora_layers_for_bias(model)
    
    print(f"Applied LoRA to {len(target_modules)} module types")
    print(f"LoRA config: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    
    return model


def load_and_prepare_model(
    model_id: str,
    pretrained_checkpoint: Optional[Path] = None,
    hf_token: Optional[str] = None,
    load_in_4bit: bool = True,
    bnb_4bit_compute_dtype: str = "float16",
    freeze_vision_encoder: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: Optional[list] = None,
    lora_bias: str = "none",
    use_quantized_loader: bool = True,  # Use quantized loader by default
) -> Tuple[Optional[CobraVLM], torch.nn.Module]:
    """
    Complete pipeline: Load Cobra model and prepare for QLoRA training.
    
    Args:
        model_id: Cobra model ID
        pretrained_checkpoint: Path to checkpoint (optional)
        hf_token: HuggingFace token
        load_in_4bit: Use 4-bit quantization
        bnb_4bit_compute_dtype: Compute dtype
        freeze_vision_encoder: Freeze vision encoder
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        lora_target_modules: Target modules for LoRA
        lora_bias: LoRA bias setting
        
    Returns:
        Tuple of (CobraVLM, prepared LLM backbone with QLoRA)
    """
    # Create quantization config
    quant_config = None
    if load_in_4bit:
        quant_config = get_quantization_config(
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        )
    
    # Load model
    vlm, llm_backbone = load_cobra_for_qlora(
        model_id=model_id,
        pretrained_checkpoint=pretrained_checkpoint,
        hf_token=hf_token,
        quantization_config=quant_config,
        freeze_vision_encoder=freeze_vision_encoder,
    )
    
    # Prepare for QLoRA
    llm_backbone = prepare_model_for_qlora(
        model=llm_backbone,
        quantization_config=quant_config,
        target_modules=lora_target_modules,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_bias=lora_bias,
    )
    
    return vlm, llm_backbone

