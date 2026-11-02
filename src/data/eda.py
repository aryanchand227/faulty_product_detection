"""
Exploratory Data Analysis (EDA) Module for MVTec AD Bottle Dataset.

Performs quantitative dataset analysis:
- Dataset sample counts across splits and defect types
- Resolution and channel depth checks
- RGB channel intensity statistics
- Defect mask analysis (pixel counts, defect area %, connected components)
- Exports visual charts to docs/assets/eda/
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import MVTEC_BOTTLE_DIR, DOCS_DIR

OUTPUT_ASSETS_DIR = DOCS_DIR / "assets" / "eda"
OUTPUT_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

def scan_dataset_structure(bottle_dir: Path = MVTEC_BOTTLE_DIR) -> pd.DataFrame:
    """
    Scans the directory structure and catalogues all images, labels, and mask paths.
    
    Returns:
        DataFrame with columns: [split, defect_type, is_anomaly, image_path, mask_path]
    """
    records = []
    
    # 1. Scan Training Set (normal only)
    train_good_dir = bottle_dir / "train" / "good"
    if train_good_dir.exists():
        for img_path in sorted(train_good_dir.glob("*.png")):
            records.append({
                "split": "train",
                "defect_type": "good",
                "is_anomaly": 0,
                "image_path": str(img_path),
                "mask_path": None
            })
            
    # 2. Scan Test Set (normal + defects)
    test_dir = bottle_dir / "test"
    gt_dir = bottle_dir / "ground_truth"
    
    if test_dir.exists():
        for category_folder in sorted(test_dir.iterdir()):
            if not category_folder.is_dir():
                continue
            cat_name = category_folder.name
            is_anomaly = 0 if cat_name == "good" else 1
            
            for img_path in sorted(category_folder.glob("*.png")):
                mask_path = None
                if is_anomaly:
                    # Corresponding mask file pattern: <name>_mask.png
                    expected_mask = gt_dir / cat_name / f"{img_path.stem}_mask.png"
                    if expected_mask.exists():
                        mask_path = str(expected_mask)
                    else:
                        print(f"[Warning] Mask missing for {img_path}")
                        
                records.append({
                    "split": "test",
                    "defect_type": cat_name,
                    "is_anomaly": is_anomaly,
                    "image_path": str(img_path),
                    "mask_path": mask_path
                })
                
    df = pd.DataFrame(records)
    return df

def analyze_image_properties(df: pd.DataFrame) -> dict:
    """
    Analyzes resolution, channels, bit-depth, and RGB intensity distributions.
    """
    resolutions = set()
    channels = set()
    means = []
    stds = []
    
    for path in df["image_path"]:
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue
        h, w, c = img_bgr.shape
        resolutions.add((w, h))
        channels.add(c)
        
        # Convert BGR to RGB and normalize to [0, 1] for statistical profiling
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) / 255.0
        means.append(img_rgb.mean(axis=(0, 1)))
        stds.append(img_rgb.std(axis=(0, 1)))
        
    means = np.array(means)
    stds = np.array(stds)
    
    return {
        "unique_resolutions": list(resolutions),
        "unique_channels": list(channels),
        "dataset_rgb_mean": means.mean(axis=0).tolist(),
        "dataset_rgb_std": stds.mean(axis=0).tolist()
    }

def analyze_mask_properties(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates pixel-level metrics for all defective test images:
    - Total defect pixel count
    - Defect area percentage
    - Number of distinct connected components (defect clusters)
    """
    mask_records = []
    defective_df = df[df["mask_path"].notnull()]
    
    for _, row in defective_df.iterrows():
        mask = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
            
        total_pixels = mask.shape[0] * mask.shape[1]
        defect_pixels = np.count_nonzero(mask > 0)
        defect_pct = (defect_pixels / total_pixels) * 100.0
        
        # Connected Component Analysis (8-connectivity)
        num_labels, labels_im = cv2.connectedComponents((mask > 0).astype(np.uint8))
        # num_labels includes background (0), so subtract 1
        num_defect_components = max(0, num_labels - 1)
        
        mask_records.append({
            "image_id": Path(row["image_path"]).name,
            "defect_type": row["defect_type"],
            "total_pixels": total_pixels,
            "defect_pixels": defect_pixels,
            "defect_pct": defect_pct,
            "num_components": num_defect_components
        })
        
    return pd.DataFrame(mask_records)

def generate_visual_artifacts(df: pd.DataFrame, mask_df: pd.DataFrame):
    """
    Generates summary charts and sample overlay inspections for documentation.
    """
    # 1. Sample Distribution Bar Chart
    plt.figure(figsize=(9, 5))
    counts = df.groupby(["split", "defect_type"]).size().unstack(fill_value=0)
    counts.plot(kind="bar", stacked=True, colormap="viridis", edgecolor="black")
    plt.title("MVTec AD Bottle: Sample Distribution across Splits and Classes", fontsize=12)
    plt.xlabel("Dataset Split", fontsize=11)
    plt.ylabel("Number of Images", fontsize=11)
    plt.xticks(rotation=0)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(OUTPUT_ASSETS_DIR / "class_distribution.png", dpi=200)
    plt.close()

    # 2. Defect Area Percentage Histogram
    plt.figure(figsize=(8, 4.5))
    for defect_type, group in mask_df.groupby("defect_type"):
        plt.hist(group["defect_pct"], bins=15, alpha=0.6, label=defect_type, edgecolor="black")
    plt.title("Defect Pixel Area Distribution (% of Total Image)", fontsize=12)
    plt.xlabel("Defect Area (%)", fontsize=11)
    plt.ylabel("Sample Count", fontsize=11)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(OUTPUT_ASSETS_DIR / "defect_area_distribution.png", dpi=200)
    plt.close()

    # 3. Defect-Mask Overlay Gallery
    defect_types = mask_df["defect_type"].unique()
    fig, axes = plt.subplots(len(defect_types), 3, figsize=(10, 3.5 * len(defect_types)))
    if len(defect_types) == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, dtype in enumerate(defect_types):
        sample_row = df[(df["defect_type"] == dtype) & (df["mask_path"].notnull())].iloc[0]
        img = cv2.cvtColor(cv2.imread(sample_row["image_path"]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(sample_row["mask_path"], cv2.IMREAD_GRAYSCALE)
        
        # Color overlay: Defect in vibrant red
        overlay = img.copy()
        overlay[mask > 0] = [255, 30, 30]
        blended = cv2.addWeighted(img, 0.65, overlay, 0.35, 0)
        
        # Col 1: Original Image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"{dtype} (Original)", fontsize=10)
        axes[i, 0].axis("off")
        
        # Col 2: Ground Truth Binary Mask
        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title(f"{dtype} (Ground Truth Mask)", fontsize=10)
        axes[i, 1].axis("off")
        
        # Col 3: Visual Overlay
        axes[i, 2].imshow(blended)
        axes[i, 2].set_title(f"{dtype} (Inspection Overlay)", fontsize=10)
        axes[i, 2].axis("off")

    plt.suptitle("MVTec AD Bottle: Defect Types vs Ground Truth Masks", fontsize=13, y=1.00)
    plt.tight_layout()
    plt.savefig(OUTPUT_ASSETS_DIR / "defect_mask_samples.png", dpi=200)
    plt.close()
    print(f"[OK] Visual artifacts saved to: {OUTPUT_ASSETS_DIR}")

def run_eda_pipeline():
    """Executes the full EDA workflow and prints the comprehensive report."""
    print("=" * 70)
    print("MVTEC AD BOTTLE: EXPLORATORY DATA ANALYSIS PIPELINE")
    print("=" * 70)
    
    df = scan_dataset_structure()
    print(f"\n[1] Total samples catalogued: {len(df)}")
    
    # Summary Table
    summary_counts = df.groupby(["split", "defect_type", "is_anomaly"]).size().reset_index(name="count")
    print("\nDataset Class & Split Distribution:")
    print(summary_counts.to_string(index=False))
    
    # Image properties
    props = analyze_image_properties(df)
    print("\n[2] Image Properties:")
    print(f"  - Unique Resolutions (W, H): {props['unique_resolutions']}")
    print(f"  - Unique Channels: {props['unique_channels']}")
    print(f"  - Dataset RGB Mean: {[round(x, 4) for x in props['dataset_rgb_mean']]}")
    print(f"  - Dataset RGB Std:  {[round(x, 4) for x in props['dataset_rgb_std']]}")
    
    # Mask properties
    mask_df = analyze_mask_properties(df)
    print("\n[3] Defect Mask Statistics (Test Anomalous Samples):")
    print(f"  - Analyzed defective images: {len(mask_df)}")
    
    stats_by_type = mask_df.groupby("defect_type").agg(
        sample_count=("defect_pct", "count"),
        min_defect_pct=("defect_pct", "min"),
        mean_defect_pct=("defect_pct", "mean"),
        max_defect_pct=("defect_pct", "max"),
        mean_components=("num_components", "mean"),
        max_components=("num_components", "max")
    ).reset_index()
    
    print("\nDefect Metrics per Defect Class:")
    print(stats_by_type.to_string(index=False))
    
    # Generate visualization artifacts
    generate_visual_artifacts(df, mask_df)
    
    return df, mask_df, props

if __name__ == "__main__":
    run_eda_pipeline()

