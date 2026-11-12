---
language:
- en
license: mit
tags:
- semantic-segmentation
- unet
- pixel-level
- defect-contours
- computer-vision
- industrial-inspection
metrics:
- dice
- iou
library_name: pytorch
pipeline_tag: image-segmentation
---

# Industrial Defect Segmentation: U-Net Architecture

## Model Overview
- **Architecture:** Symmetric Encoder-Decoder U-Net with Skip Connections (Ronneberger et al., MICCAI 2015)
- **Input Resolution:** $256 \times 256 \times 3$ (with reflection padding for odd dimensions)
- **Loss Function:** Combined Binary Cross-Entropy + Soft Dice Loss:
  $$\mathcal{L} = \alpha \mathcal{L}_{\text{BCE}} + (1 - \alpha) \mathcal{L}_{\text{Dice}}$$
- **Output:** Dense single-channel binary mask ($1$ for defect pixel, $0$ for nominal background).
- **Repository:** `TAC13/industrial-damage-unet-bottle`

## Morphology Extraction from Segmentation Masks
The segmented binary mask directly feeds the downstream `DefectMorphologyAnalyzer`:
1. **Contour Extraction:** OpenCV topological structural hierarchy.
2. **Geometric Descriptors:** Area ($A_{\text{px}}$), Perimeter ($P_{\text{px}}$), Aspect Ratio ($\lambda_1 / \lambda_2$), Circularity ($4\pi A / P^2$), Solidity ($A / A_{\text{hull}}$), Compactness, Equivalent Diameter.
3. **Physical Damage Area:** $A_{\text{mm}^2} = A_{\text{px}} \times s^2$.

## Surface Geometry Caution
Reported areas represent projected sensor plane damage. For non-planar surfaces (e.g., cylindrical bottles), true 3D surface area requires trigonometric surface curvature compensation.

