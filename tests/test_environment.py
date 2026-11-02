"""
Environment and Installation Verification Script.

Tests that all required development dependencies are properly installed,
verifies device detection (CPU locally, device-agnostic code),
and performs a quick tensor and image processing sanity check.
"""

import sys
import platform
import pytest

def test_python_version():
    """Verify Python version is 3.11 or 3.12."""
    version = sys.version_info
    print(f"\n[Test] Python version: {version.major}.{version.minor}.{version.micro}")
    assert version.major == 3 and version.minor in (11, 12), (
        f"Expected Python 3.11 or 3.12, found {version.major}.{version.minor}"
    )

def test_torch_cpu_sanity():
    """Verify PyTorch import, CPU device detection, and basic tensor math."""
    import torch
    print(f"\n[Test] PyTorch version: {torch.__version__}")
    print(f"[Test] CUDA available: {torch.cuda.is_available()}")
    
    # We expect False locally as established in Phase 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Test] Using device: {device}")
    
    # Simple tensor operation
    x = torch.ones((2, 3), device=device)
    y = x * 2.0
    assert y.sum().item() == 12.0
    print("[Test] PyTorch tensor math: OK")

def test_torchvision():
    """Verify torchvision import and basic image transformation."""
    import torchvision
    import torchvision.transforms as T
    from PIL import Image
    import numpy as np

    print(f"\n[Test] Torchvision version: {torchvision.__version__}")
    # Create a dummy image
    dummy_img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor()
    ])
    tensor = transform(dummy_img)
    assert tensor.shape == (3, 64, 64)
    print("[Test] Torchvision transforms: OK")

def test_opencv():
    """Verify OpenCV headless import and basic matrix operations."""
    import cv2
    import numpy as np
    print(f"\n[Test] OpenCV version: {cv2.__version__}")
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    resized = cv2.resize(img, (50, 50))
    assert resized.shape == (50, 50, 3)
    print("[Test] OpenCV resize: OK")

def test_scientific_stack():
    """Verify NumPy, Pandas, Scikit-Learn."""
    import numpy as np
    import pandas as pd
    import sklearn
    print(f"\n[Test] NumPy version: {np.__version__}")
    print(f"[Test] Pandas version: {pd.__version__}")
    print(f"[Test] Scikit-Learn version: {sklearn.__version__}")
    assert np.__version__ is not None
    assert pd.__version__ is not None
    assert sklearn.__version__ is not None

def test_fastapi_and_pydantic():
    """Verify FastAPI and Pydantic for backend readiness."""
    import fastapi
    import pydantic
    print(f"\n[Test] FastAPI version: {fastapi.__version__}")
    print(f"[Test] Pydantic version: {pydantic.__version__}")
    assert fastapi.__version__ is not None
    assert pydantic.__version__ is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

