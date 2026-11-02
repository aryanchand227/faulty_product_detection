"""
Resolution Impact Analysis on Industrial Defect Masks.

Quantifies the effect of downsampling from native 900x900 to:
- 256 x 256
- 384 x 384
- 512 x 512

Evaluates:
- Defect pixel preservation ratio
- Connected component count preservation (detecting vanished defect fragments)
- Visual boundary fidelity across broken_small and contamination classes
"""

import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import MVTEC_BOTTLE_DIR, DOCS_DIR
from src.data.eda import scan_dataset_structure

OUTPUT_ASSETS_DIR = DOCS_DIR / "assets" / "eda"
OUTPUT_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

RESOLUTIONS = [256, 384, 512]

def evaluate_resolution_impact():
    df = scan_dataset_structure(MVTEC_BOTTLE_DIR)
    defective_df = df[df["mask_path"].notnull()]
    
    records = []
    
    for _, row in defective_df.iterrows():
        mask_path = row["mask_path"]
        orig_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if orig_mask is None:
            continue
            
        orig_defect_pixels = np.count_nonzero(orig_mask > 0)
        orig_total_pixels = orig_mask.shape[0] * orig_mask.shape[1]
        orig_defect_pct = (orig_defect_pixels / orig_total_pixels) * 100.0
        
        # Original connected components
        num_orig_labels, _ = cv2.connectedComponents((orig_mask > 0).astype(np.uint8))
        orig_components = max(0, num_orig_labels - 1)
        
        row_data = {
            "image_id": Path(row["image_path"]).name,
            "defect_type": row["defect_type"],
            "orig_pixels": orig_defect_pixels,
            "orig_pct": orig_defect_pct,
            "orig_components": orig_components
        }
        
        for res in RESOLUTIONS:
            # Nearest neighbor interpolation for discrete mask
            resized_mask = cv2.resize(orig_mask, (res, res), interpolation=cv2.INTER_NEAREST)
            res_defect_pixels = np.count_nonzero(resized_mask > 0)
            res_total_pixels = res * res
            res_defect_pct = (res_defect_pixels / res_total_pixels) * 100.0
            
            num_res_labels, _ = cv2.connectedComponents((resized_mask > 0).astype(np.uint8))
            res_components = max(0, num_res_labels - 1)
            
            # Component preservation (did any fragment disappear?)
            vanished_components = max(0, orig_components - res_components)
            
            row_data[f"pixels_{res}"] = res_defect_pixels
            row_data[f"pct_{res}"] = res_defect_pct
            row_data[f"pct_error_{res}"] = abs(res_defect_pct - orig_defect_pct)
            row_data[f"components_{res}"] = res_components
            row_data[f"vanished_components_{res}"] = vanished_components

        records.append(row_data)

    res_df = pd.DataFrame(records)
    
    # Summary by defect type
    print("=" * 80)
    print("RESOLUTION IMPACT STUDY: 256x256 vs 384x384 vs 512x512")
    print("=" * 80)
    
    for res in RESOLUTIONS:
        print(f"\n--- Resolution: {res} x {res} ---")
        summary = res_df.groupby("defect_type").agg(
            mean_pixels=(f"pixels_{res}", "mean"),
            min_pixels=(f"pixels_{res}", "min"),
            mean_pct_error=(f"pct_error_{res}", "mean"),
            max_pct_error=(f"pct_error_{res}", "max"),
            total_vanished_fragments=(f"vanished_components_{res}", "sum")
        ).reset_index()
        print(summary.to_string(index=False))

    # Generate visual comparison plot of a small defect across resolutions
    small_defect_row = res_df[res_df["defect_type"] == "broken_small"].sort_values("orig_pixels").iloc[0]
    sample_mask_path = defective_df[defective_df["image_path"].str.endswith(small_defect_row["image_id"])]["mask_path"].values[0]
    orig_mask = cv2.imread(sample_mask_path, cv2.IMREAD_GRAYSCALE)
    
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    axes[0].imshow(orig_mask, cmap="gray")
    axes[0].set_title(f"Original (900x900)\n{small_defect_row['orig_pixels']} pixels | {small_defect_row['orig_components']} components", fontsize=9)
    axes[0].axis("off")
    
    for idx, res in enumerate(RESOLUTIONS):
        resized = cv2.resize(orig_mask, (res, res), interpolation=cv2.INTER_NEAREST)
        pix = np.count_nonzero(resized > 0)
        num_c, _ = cv2.connectedComponents((resized > 0).astype(np.uint8))
        c = max(0, num_c - 1)
        axes[idx + 1].imshow(resized, cmap="gray")
        axes[idx + 1].set_title(f"Resized ({res}x{res})\n{pix} pixels | {c} components", fontsize=9)
        axes[idx + 1].axis("off")
        
    plt.suptitle(f"Defect Detail Preservation across Scales: Sample {small_defect_row['image_id']} (broken_small)", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUTPUT_ASSETS_DIR / "resolution_comparison.png", dpi=200)
    plt.close()
    
    print(f"\n[OK] Visual resolution study plot saved to: {OUTPUT_ASSETS_DIR / 'resolution_comparison.png'}")
    return res_df

if __name__ == "__main__":
    evaluate_resolution_impact()
