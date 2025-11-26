"""
Visualization utilities.
"""
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from PIL import Image
from pathlib import Path
from textwrap import wrap

def create_visualization_from_results(
    results_data: Dict[str, Any],
    images_map: Dict[int, Image.Image],
    output_path: Path,
    max_images: int = 10
):
    """
    Create visualization table from structured results.
    
    Args:
        results_data: The full results dictionary
        images_map: Map of image_id to PIL Image (loaded separately as JSON doesn't store images)
        output_path: Path to save image
        max_images: Limit
    """
    results_list = results_data["results"]
    if not results_list:
        print("No results to visualize")
        return

    # Filter results that have images available
    valid_results = [r for r in results_list if r["image_id"] in images_map]
    valid_results = sorted(valid_results, key=lambda x: x["image_id"])[:max_images]
    
    n_images = len(valid_results)
    if n_images == 0:
        print("No matching images found for visualization")
        return

    fig = plt.figure(figsize=(24, 6 * n_images))
    
    for idx, res in enumerate(valid_results):
        img_id = res["image_id"]
        image = images_map[img_id]
        
        refs = res["reference_captions"]
        gen_cap = res["generated_caption"]
        trace = res.get("reasoning_trace")
        
        # Image subplot
        ax_img = plt.subplot(n_images, 2, idx * 2 + 1)
        ax_img.imshow(image)
        ax_img.axis('off')
        ax_img.set_title(f'Image {img_id}', fontsize=16, fontweight='bold', pad=15)
        
        # Text subplot
        ax_text = plt.subplot(n_images, 2, idx * 2 + 2)
        ax_text.axis('off')
        ax_text.set_xlim(0, 1)
        ax_text.set_ylim(0, 1)
        
        text_parts = []
        
        # References
        text_parts.append(("Reference Captions:", "bold", 13))
        for i, ref in enumerate(refs[:3]):
            text_parts.append((f"  {i+1}. {ref}", "normal", 11))
        text_parts.append(("", "normal", 0))
        
        # Metrics
        if "metrics" in res:
            text_parts.append(("Metrics:", "bold", 13))
            metrics_str = "  " + "  ".join([f"{k}: {v:.3f}" for k, v in res["metrics"].items() if isinstance(v, float)])
            text_parts.append((metrics_str, "normal", 11))
            text_parts.append(("", "normal", 0))
            
        # Trace
        if trace:
            text_parts.append(("Reasoning Trace:", "bold", 13))
            # Truncate
            trace_short = trace[:400] + "..." if len(trace) > 400 else trace
            text_parts.append((f"  {trace_short}", "italic", 10))
            text_parts.append(("", "normal", 0))
            
        # Caption
        text_parts.append(("Generated Caption:", "bold", 13))
        text_parts.append((f"  {gen_cap}", "normal", 11))
        
        # Rendering text
        y_pos = 0.98
        line_height_base = 0.045
        
        for text, style, fontsize in text_parts:
            if not text:
                y_pos -= 0.03
                continue
                
            weight = 'bold' if style == "bold" else 'normal'
            style_attr = 'italic' if style == "italic" else 'normal'
            line_height = line_height_base * (fontsize / 11)
            
            ax_text.text(0.02, y_pos, text, transform=ax_text.transAxes,
                        fontsize=fontsize, weight=weight, style=style_attr,
                        verticalalignment='top', horizontalalignment='left',
                        family='sans-serif', wrap=True,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7) if style == "bold" else None)
            
            # Rough estimation of lines
            num_lines = max(1, len(text) // 90 + 1)
            y_pos -= line_height * num_lines + 0.01

    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Visualization saved to {output_path}")
    plt.close()

def create_comparison_visualization(
    baseline_data: Dict[str, Any],
    scratchpad_data: Dict[str, Any],
    images_map: Dict[int, Image.Image],
    output_path: Path,
    max_images: int = 10
):
    """
    Create side-by-side comparison visualization.
    """
    # Map results by image_id
    base_map = {r["image_id"]: r for r in baseline_data["results"]}
    scratch_map = {r["image_id"]: r for r in scratchpad_data["results"]}
    
    # Find common images that exist in images_map
    common_ids = sorted([
        iid for iid in base_map.keys() 
        if iid in scratch_map and iid in images_map
    ])[:max_images]
    
    if not common_ids:
        print("No common images found for comparison")
        return

    n_images = len(common_ids)
    # 3 columns: Image, Baseline, Scratchpad
    fig = plt.figure(figsize=(30, 8 * n_images))
    
    for idx, img_id in enumerate(common_ids):
        base_res = base_map[img_id]
        scratch_res = scratch_map[img_id]
        image = images_map[img_id]
        
        # Col 1: Image & Refs
        ax_img = plt.subplot(n_images, 3, idx * 3 + 1)
        ax_img.imshow(image)
        ax_img.axis('off')
        ax_img.set_title(f'Image {img_id}', fontsize=16, fontweight='bold', pad=15)
        
        # Add refs below image
        ref_text = "References:\n" + "\n".join([f"- {r}" for r in base_res["reference_captions"][:3]])
        # Render near bottom
        ax_img.text(0.05, -0.05, ref_text, transform=ax_img.transAxes, fontsize=11, verticalalignment='top', wrap=True)
        
        # Helper to render text block
        def render_text_block(ax, title, res, compare_with=None):
            ax.axis('off')
            ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
            y = 0.95
            
            # Caption
            ax.text(0, y, "Caption:", fontweight='bold', fontsize=12)
            y -= 0.04
            
            cap_lines = wrap(res["generated_caption"], width=50)
            for line in cap_lines:
                ax.text(0.02, y, line, fontsize=11)
                y -= 0.03
            y -= 0.02
            
            # Metrics
            if "metrics" in res:
                ax.text(0, y, "Metrics:", fontweight='bold', fontsize=12)
                y -= 0.04
                
                for k, v in res["metrics"].items():
                    if not isinstance(v, float): continue
                    
                    text = f"{k}: {v:.3f}"
                    color = "black"
                    
                    # Comparison diff
                    if compare_with and k in compare_with["metrics"]:
                        base_v = compare_with["metrics"][k]
                        diff = v - base_v
                        # Only show diff if meaningful
                        if abs(diff) > 0.001:
                            sign = "+" if diff > 0 else ""
                            diff_text = f" ({sign}{diff:.3f})"
                            # Green for positive diff, Red for negative
                            color = "green" if diff > 0 else "red"
                            text += diff_text
                    
                    ax.text(0.02, y, text, fontsize=11, color=color, fontweight='bold' if color != 'black' else 'normal')
                    y -= 0.03
            
            y -= 0.02
            
            # Reasoning Trace (only if present)
            trace = res.get("reasoning_trace")
            if trace:
                ax.text(0, y, "Reasoning Trace:", fontweight='bold', fontsize=12)
                y -= 0.04
                trace_short = trace[:500] + "..." if len(trace) > 500 else trace
                trace_lines = wrap(trace_short, width=60)
                for line in trace_lines:
                    ax.text(0.02, y, line, fontsize=9, style='italic')
                    y -= 0.025

        # Col 2: Baseline
        ax_base = plt.subplot(n_images, 3, idx * 3 + 2)
        render_text_block(ax_base, "Baseline", base_res)
        
        # Col 3: Scratchpad (Compare with Baseline)
        ax_scratch = plt.subplot(n_images, 3, idx * 3 + 3)
        render_text_block(ax_scratch, "Scratchpad", scratch_res, compare_with=base_res)

    plt.tight_layout(pad=3.0)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Comparison visualization saved to {output_path}")
    plt.close()
