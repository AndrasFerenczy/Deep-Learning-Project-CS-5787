"""
Main entry point for Cobra Evaluation System.
"""
import torch
from pathlib import Path
from datetime import datetime
from cobra import load

from .config import parse_args
from .registry import Registry
from .data.loader import COCODatasetLoader
from .utils.gpu import clear_gpu_memory, check_gpu_memory
from .utils.io import save_json_results, load_json_results, find_latest_result_file
from .utils.viz import create_visualization_from_results

# Import all plugins to ensure registration
from .generators import baseline, scratchpad, external
from .metrics import bleu, bert_score

def main():
    args = parse_args()

    # Use method-specific subdirectories for cleaner organization
    result_dir = Path(args.output_dir) / args.method
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. GPU Setup
    if args.clear_cache and torch.cuda.is_available():
        print("Clearing GPU cache...")
        clear_gpu_memory()
        
    if torch.cuda.is_available():
        if not check_gpu_memory(min_free_gb=args.min_free_gb):
            print("Exiting due to insufficient GPU memory.")
            return
        device = torch.device('cuda')
        print("Using GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
        
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    # 2. Load Model
    print(f"Loading model: {args.model_id}")
    try:
        hf_token_path = Path(args.hf_token)
        if hf_token_path.exists():
            hf_token = hf_token_path.read_text().strip()
        else:
            hf_token = None
            
        vlm = load(args.model_id, hf_token=hf_token)
        vlm.to(device, dtype=dtype)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Initialize Generator
    generator_cls = Registry.get_generator(args.method)
    
    # Handle different init signatures
    if args.method == "scratchpad":
        generator = generator_cls(vlm, scratchpad_passes=args.scratchpad_passes)
    else:
        generator = generator_cls(vlm)
        
    print(f"Initialized {args.method} generator")

    # 4. Load Dataset
    print(f"Loading dataset ({args.num_samples} samples)...")
    data_loader = COCODatasetLoader(
        split="val",
        streaming=not args.no_streaming,
        limit=args.num_samples
    )

    # 4b. Load Cached Results (if any)
    loaded_results_map = {}
    if args.load_results:
        load_path = None
        
        # Resolve path
        if args.load_results.lower() == "latest":
            prefix = f"results_{args.method}"
            load_path = find_latest_result_file(result_dir, prefix)
            if not load_path:
                print(f"No previous results found for method '{args.method}' in {result_dir}")
        else:
            p = Path(args.load_results)
            if p.is_dir():
                prefix = f"results_{args.method}"
                load_path = find_latest_result_file(p, prefix)
            else:
                load_path = p
                
        if load_path and load_path.exists():
            print(f"Loading results from {load_path}...")
            try:
                loaded_data = load_json_results(load_path)
                for r in loaded_data.get("results", []):
                    loaded_results_map[r["image_id"]] = r
                print(f"Loaded {len(loaded_results_map)} cached results.")
            except Exception as e:
                print(f"Error loading results file: {e}")
                print("Continuing without cache...")
        elif args.load_results:
             print(f"Warning: Results file not found: {args.load_results}")

    # 5. Run Evaluation
    results = []
    references_map = {}  # For global metric computation
    predictions_map = {}
    images_map = {}      # Keep images for visualization
    
    start_time = datetime.now()
    print(f"Starting generation at {start_time.strftime('%H:%M:%S')}...")
    for i, (image_id, image, refs) in enumerate(data_loader):
        # Store image for visualization (limit to save memory)
        if len(images_map) < 20: 
            images_map[image_id] = image.copy()

        # Clean up refs (ensure list of strings)
        if isinstance(refs, str):
            refs = [refs]

        # Check cache first
        if args.load_results and image_id in loaded_results_map:
            print(f"Using cached result for image {image_id}")
            cached = loaded_results_map[image_id]
            
            # Store for metrics
            references_map[image_id] = refs
            predictions_map[image_id] = [cached["generated_caption"]]
            
            # Append cached result directly
            # We might want to ensure 'metrics' is cleared if we plan to recompute them,
            # but usually we recompute them anyway in step 6.
            # Actually, the loop below recomputes sample-level BLEU for 'results'.
            # So we should append the dict.
            # However, the cached result might have old metrics. 
            # The code in step 6 updates `res["metrics"]`.
            
            results.append(cached)
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Processed {i + 1}/{args.num_samples} samples (Cached)...")
            continue
            
        # Generate
        gen_result = generator.generate(
            image, 
            "Please carefully observe the image and come up with a caption for the image.",
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            reasoning_max_tokens=args.reasoning_max_tokens,
            repetition_penalty=args.repetition_penalty
        )
        
        # Store for metrics
        references_map[image_id] = refs
        predictions_map[image_id] = [gen_result.caption]
        
        # Record result
        results.append({
            "image_id": image_id,
            "reference_captions": refs,
            "generated_caption": gen_result.caption,
            "reasoning_trace": gen_result.reasoning_trace,
            "metadata": gen_result.metadata,
            "metrics": {} # Populated later
        })
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Processed {i + 1}/{args.num_samples} samples...")
        if (i + 1) % 10 == 0:
            clear_gpu_memory()

    # 6. Compute Metrics
    print("Computing global metrics...")
    aggregate_metrics = {}
    
    # Instantiate all registered metrics
    # (In a real CLI we might want to select which metrics to run)
    metric_instances = []
    try:
        metric_instances.append(Registry.get_metric("bleu")())
        metric_instances.append(Registry.get_metric("bert_score")())
    except Exception as e:
        print(f"Warning: Could not initialize some metrics: {e}")

    for metric in metric_instances:
        try:
            scores = metric.compute(references_map, predictions_map)
            aggregate_metrics.update(scores)
            
            # If metric supports per-sample scores, we would need a different interface
            # Current interfaces.py BaseMetric.compute returns Dict[str, float] (aggregated)
            # To get per-sample scores, we'd need to adjust the interface or iterate.
            # For now, let's just compute per-sample BLEU for the JSON report manually or if supported.
            
            # Hack: re-compute per sample for BLEU since it's fast
            if isinstance(metric, Registry.get_metric("bleu")):
                for res in results:
                    img_id = res["image_id"]
                    single_ref = {img_id: references_map[img_id]}
                    single_pred = {img_id: predictions_map[img_id]}
                    sample_score = metric.compute(single_ref, single_pred)
                    res["metrics"].update(sample_score)
                    
        except Exception as e:
            print(f"Error computing metric {metric.__class__.__name__}: {e}")

    # 7. Save Results
    output_data = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "config": vars(args)
        },
        "aggregate_metrics": aggregate_metrics,
        "results": results
    }
    
    output_path = save_json_results(output_data, result_dir, f"results_{args.method}")
    
    # 8. Visualize
    viz_path = output_path.parent / f"viz_{output_path.stem}.png"
    print("Generating visualization...")
    create_visualization_from_results(output_data, images_map, viz_path)

    end_time = datetime.now()
    duration = end_time - start_time
    print(f"\nEvaluation completed in {duration}")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()

