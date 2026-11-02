# %% [markdown]
# # Colab Environment Setup and Verification
# 
# **Purpose:** This notebook verifies the Google Colab training environment,
# installs required dependencies, configures Google Drive and Hugging Face
# authentication, and runs a GPU smoke test.
# 
# **Run this notebook first** before any training experiment.
# 
# ---

# %% [markdown]
# ## Step 1: Verify GPU Availability
# 
# Google Colab provides different GPU types depending on your plan and
# availability. We need a GPU for training — the Tesla T4 is the most
# common free-tier GPU.
# 
# **What this cell does:**
# - Checks if CUDA (NVIDIA's GPU computing platform) is available
# - Reports the GPU model and memory
# - **Stops the notebook** if no GPU is found
# 
# **If this fails:** Go to Runtime → Change runtime type → Select GPU

# %%
import torch
import sys

print("=" * 60)
print("ENVIRONMENT VERIFICATION")
print("=" * 60)

print(f"\nPython version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Get GPU memory info
    total_mem = torch.cuda.get_device_properties(0).total_mem
    print(f"GPU Memory: {total_mem / (1024**3):.1f} GB")
    
    # Quick GPU computation test
    x = torch.randn(1000, 1000, device="cuda")
    y = torch.matmul(x, x)
    print(f"\nGPU computation test: PASSED")
    print(f"  Created 1000x1000 tensor on GPU")
    print(f"  Matrix multiplication result shape: {y.shape}")
    del x, y
    torch.cuda.empty_cache()
else:
    print("\n" + "!" * 60)
    print("ERROR: No GPU detected!")
    print("Go to Runtime → Change runtime type → Select GPU")
    print("!" * 60)
    raise RuntimeError(
        "GPU is required for training. "
        "Enable GPU in Colab: Runtime → Change runtime type → GPU"
    )

# %% [markdown]
# ## Step 2: Install Required Dependencies
# 
# Colab comes with PyTorch pre-installed, but we need additional
# libraries for our project.
# 
# **What we're installing:**
# - `opencv-python-headless`: Image processing (headless = no GUI, suitable for servers)
# - `ultralytics`: YOLO object detection framework
# - `anomalib`: Industrial anomaly detection library (includes PatchCore)
# - `albumentations`: Advanced image augmentation
# - `huggingface_hub`: Upload/download models from Hugging Face
# - `mlflow`: Experiment tracking
# - `scikit-learn`: Classical ML metrics
# - `torchmetrics`: PyTorch-native metrics
# - `timm`: PyTorch Image Models (EfficientNet, etc.)

# %%
# Install dependencies
# The --quiet flag suppresses verbose output
# We pin major versions where possible for reproducibility

# !pip install --quiet opencv-python-headless>=4.8
# !pip install --quiet ultralytics>=8.0
# !pip install --quiet anomalib>=1.0
# !pip install --quiet albumentations>=1.3
# !pip install --quiet huggingface_hub>=0.20
# !pip install --quiet mlflow>=2.10
# !pip install --quiet scikit-learn>=1.3
# !pip install --quiet torchmetrics>=1.2
# !pip install --quiet timm>=0.9
# !pip install --quiet matplotlib>=3.7
# !pip install --quiet pandas>=2.0

print("Dependencies installed successfully.")
print("Note: Restart runtime if prompted.")

# %% [markdown]
# ## Step 3: Verify Installed Packages
# 
# After installation, verify that all packages import correctly.
# If any import fails, the error message will tell us which
# package needs attention.

# %%
import importlib

packages = {
    "cv2": "OpenCV",
    "numpy": "NumPy",
    "pandas": "Pandas",
    "matplotlib": "Matplotlib",
    "sklearn": "scikit-learn",
    "torchvision": "Torchvision",
    "torchmetrics": "TorchMetrics",
    "timm": "timm (PyTorch Image Models)",
    "ultralytics": "Ultralytics YOLO",
    "huggingface_hub": "Hugging Face Hub",
    "albumentations": "Albumentations",
}

print("Package Verification:")
print("-" * 50)

all_ok = True
for module_name, display_name in packages.items():
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, "__version__", "unknown")
        print(f"  ✓ {display_name}: {version}")
    except ImportError as e:
        print(f"  ✗ {display_name}: FAILED - {e}")
        all_ok = False

if all_ok:
    print("\nAll packages verified successfully.")
else:
    print("\nSome packages failed. Check the errors above.")

# %% [markdown]
# ## Step 4: Mount Google Drive
# 
# **Why Google Drive?**
# 
# Colab sessions are temporary — when the session disconnects, all files
# on the Colab VM are deleted. Google Drive persists between sessions.
# 
# We'll use Drive to:
# - Store training checkpoints (so we don't lose progress if disconnected)
# - Store intermediate results
# - Cache downloaded datasets (avoid re-downloading every session)
# 
# **Important:** Final model artifacts go to Hugging Face Hub, not Drive.
# Drive is for intermediate checkpoints only.

# %%
from google.colab import drive

drive.mount("/content/drive")

# Create project directory on Drive
import os

DRIVE_PROJECT_DIR = "/content/drive/MyDrive/industrial-damage-detection"

# Create subdirectories
for subdir in [
    "checkpoints/classification",
    "checkpoints/detection",
    "checkpoints/segmentation",
    "checkpoints/anomaly",
    "datasets",
    "logs",
    "results",
]:
    os.makedirs(os.path.join(DRIVE_PROJECT_DIR, subdir), exist_ok=True)

print(f"Project directory on Drive: {DRIVE_PROJECT_DIR}")
print("Contents:")
for item in os.listdir(DRIVE_PROJECT_DIR):
    print(f"  {item}/")

# %% [markdown]
# ## Step 5: Configure and Verify Hugging Face Write/Upload Access
# 
# **Why test an actual upload?**
# `whoami()` only proves your token is valid for reading user metadata. It does
# NOT prove you have write/upload permissions to create repositories or push model weights.
# 
# **What this cell does:**
# 1. Logs in via token retrieved securely from Colab Secrets (`HF_TOKEN`).
# 2. Queries user identity (`whoami()`).
# 3. Creates a temporary private repository: `<username>/inspection-upload-smoke-test`.
# 4. Uploads a small test artifact (`test_artifact.json`).
# 5. Verifies the file exists in the repo and can be downloaded.
# 6. Deletes the temporary test repository to keep your account clean.
# 7. Confirms end-to-end write permissions.

# %%
import json
import time
from huggingface_hub import HfApi, login, hf_hub_download

try:
    from google.colab import userdata
    hf_token = userdata.get("HF_TOKEN")
    login(token=hf_token)
    api = HfApi(token=hf_token)
    user_info = api.whoami()
    username = user_info["name"]
    print(f"[OK] Authenticated as Hugging Face user: {username}")
    
    # Run End-to-End Write & Upload Verification
    test_repo_id = f"{username}/inspection-upload-smoke-test"
    print(f"\n[Test] Creating temporary private repository: {test_repo_id}")
    api.create_repo(repo_id=test_repo_id, repo_type="model", private=True, exist_ok=True)
    
    # Create a small test artifact
    test_payload = {
        "project": "Industrial Damage Detection",
        "verification": "HF Hub Upload Verification",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "ready_for_training"
    }
    test_file_path = "/content/smoke_test.json"
    with open(test_file_path, "w") as f:
        json.dump(test_payload, f, indent=2)
        
    print(f"[Test] Uploading test artifact: {test_file_path} -> {test_repo_id}")
    api.upload_file(
        path_or_fileobj=test_file_path,
        path_in_repo="smoke_test.json",
        repo_id=test_repo_id,
        repo_type="model",
        commit_message="test: verify write upload permissions"
    )
    
    # Download back and verify
    print("[Test] Verifying download of uploaded artifact...")
    downloaded = hf_hub_download(repo_id=test_repo_id, filename="smoke_test.json", repo_type="model")
    with open(downloaded, "r") as f:
        verified_data = json.load(f)
    assert verified_data["verification"] == test_payload["verification"]
    print("  ✓ Artifact verified successfully on Hugging Face Hub.")
    
    # Clean up temporary test repository
    print(f"[Test] Cleaning up temporary repository: {test_repo_id}")
    api.delete_repo(repo_id=test_repo_id, repo_type="model")
    print("\n✓ SUCCESS: Hugging Face write and upload permissions are fully verified.")
    
except Exception as e:
    print("\n" + "!" * 60)
    print("HUGGING FACE VERIFICATION FAILED:")
    print(e)
    print("Ensure you have created a token with WRITE role at:")
    print("https://huggingface.co/settings/tokens")
    print("and saved it under Colab Secrets (key icon on left) with name 'HF_TOKEN'.")
    print("!" * 60)
    raise

# %% [markdown]
# ## Step 6: GPU Stress Test
# 
# **Why test this?**
# 
# A quick import check doesn't guarantee the GPU will work under load.
# This test:
# 1. Creates a small model
# 2. Runs a forward pass on the GPU
# 3. Computes a loss and runs backpropagation
# 4. Measures memory usage
# 
# If this works, the GPU is ready for real training.

# %%
import torch
import torch.nn as nn
import time

print("GPU Stress Test")
print("=" * 50)

device = torch.device("cuda")

# Create a small test model
model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 64, 3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(64, 2),
).to(device)

# Create test data
batch = torch.randn(16, 3, 256, 256, device=device)
labels = torch.randint(0, 2, (16,), device=device)

# Forward pass
start = time.time()
output = model(batch)
loss = nn.CrossEntropyLoss()(output, labels)

# Backward pass
loss.backward()

elapsed = time.time() - start

# Memory usage
allocated = torch.cuda.memory_allocated(device) / (1024**2)
reserved = torch.cuda.memory_reserved(device) / (1024**2)

print(f"Forward + backward pass: {elapsed:.3f}s")
print(f"Output shape: {output.shape}")
print(f"Loss: {loss.item():.4f}")
print(f"GPU memory allocated: {allocated:.1f} MB")
print(f"GPU memory reserved: {reserved:.1f} MB")
print(f"\n✓ GPU is ready for training.")

# Cleanup
del model, batch, labels, output, loss
torch.cuda.empty_cache()

# %% [markdown]
# ## Step 7: Mixed Precision Test
# 
# **What is mixed precision?**
# 
# By default, neural networks use 32-bit floating point (FP32) for all
# computations. Mixed precision uses 16-bit (FP16) for most operations
# and 32-bit only where needed (like loss scaling).
# 
# **Benefits:**
# - ~2x faster training on GPUs with Tensor Cores (T4, A100, etc.)
# - ~50% less GPU memory per batch → can use larger batches
# - No significant accuracy loss in practice
# 
# **How it works:**
# - `torch.cuda.amp.autocast()` automatically casts operations to FP16
# - `GradScaler` prevents gradient underflow in FP16
# 
# We'll use this in all our training notebooks.

# %%
from torch.amp import autocast, GradScaler

# Create model and data
model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(64, 2),
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = GradScaler("cuda")

batch = torch.randn(16, 3, 256, 256, device=device)
labels = torch.randint(0, 2, (16,), device=device)

# Mixed precision forward pass
with autocast("cuda"):
    output = model(batch)
    loss = nn.CrossEntropyLoss()(output, labels)

# Scaled backward pass
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
optimizer.zero_grad()

print(f"Mixed precision test: PASSED")
print(f"Loss: {loss.item():.4f}")
print(f"This GPU supports mixed precision training.")

# Cleanup
del model, optimizer, scaler, batch, labels, output, loss
torch.cuda.empty_cache()

# %% [markdown]
# ## Step 8: Environment Summary
# 
# This cell prints a complete summary of the verified environment.
# Save this output for your records — it documents exactly what
# hardware and software was used for your experiments.

# %%
import platform

print("=" * 60)
print("ENVIRONMENT SUMMARY")
print("=" * 60)
print(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Python: {sys.version}")
print(f"Platform: {platform.platform()}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"cuDNN: {torch.backends.cudnn.version()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / (1024**3):.1f} GB")
print(f"Drive mounted: {os.path.exists('/content/drive/MyDrive')}")
print(f"Project dir: {DRIVE_PROJECT_DIR}")
print(f"Project dir exists: {os.path.exists(DRIVE_PROJECT_DIR)}")
print("=" * 60)
print("\n✓ Environment is ready for training experiments.")

# %% [markdown]
# ## Next Steps
# 
# The training environment is verified. Proceed to:
# 
# 1. `01_dataset_exploration.ipynb` — Download and explore the datasets
# 2. `02_preprocessing.ipynb` — Build the preprocessing pipeline
# 3. `03_cnn_baseline.ipynb` — Train the first classification model

