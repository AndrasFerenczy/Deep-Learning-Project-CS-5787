"""
Visualization utilities.
"""
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from PIL import Image
from pathlib import Path

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

