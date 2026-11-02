"""
MVTec AD Dataset Downloader.

Downloads the specified category of the MVTec Anomaly Detection dataset
from the Hugging Face dataset mirror (foersben/mvtec-ad) directly into
the local datasets/raw/mvtec_ad directory without requiring the entire 5GB archive.
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import RAW_DATA_DIR, MVTecConfig

def download_mvtec_category(category: str = "bottle", target_dir: Path = None) -> Path:
    """
    Download a single MVTec AD category from Hugging Face mirror.
    
    Args:
        category: Subcategory name (e.g. 'bottle', 'cable', 'capsule')
        target_dir: Destination root folder. Defaults to datasets/raw/mvtec_ad.
        
    Returns:
        Path to the downloaded category folder.
    """
    if target_dir is None:
        target_dir = RAW_DATA_DIR / "mvtec_ad"
        
    target_dir.mkdir(parents=True, exist_ok=True)
    category_dir = target_dir / category
    
    print(f"[*] Downloading MVTec AD '{category}' to: {target_dir}")
    print(f"[*] Repository: foersben/mvtec-ad (allow_patterns='{category}/*')")
    
    # Download category files directly
    snapshot_download(
        repo_id="foersben/mvtec-ad",
        repo_type="dataset",
        allow_patterns=f"{category}/*",
        local_dir=str(target_dir),
    )
    
    print(f"[OK] Download completed successfully: {category_dir}")
    return category_dir

if __name__ == "__main__":
    config = MVTecConfig()
    download_mvtec_category(config.category)

