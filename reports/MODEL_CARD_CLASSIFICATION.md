---
language:
- en
license: mit
tags:
- defect-classification
- transfer-learning
- resnet18
- computer-vision
- industrial-inspection
metrics:
- accuracy
- f1
library_name: pytorch
pipeline_tag: image-classification
---

# Industrial Defect Classification: ResNet-18 Transfer Learning

## Model Overview
- **Architecture:** ResNet-18 (He et al., CVPR 2016)
- **Pretraining:** ImageNet-1k (V1 weights)
- **Modifications:** Replaced final fully-connected linear projection ($512 \rightarrow K$) for industrial defect taxonomy.
- **Input Resolution:** $224 \times 224 \times 3$ (RGB normalized with ImageNet mean and std).
- **Inference Mode:** Softmax probability distribution with entropy-based uncertainty quantification.
- **Repository:** `TAC13/industrial-damage-classification`

## Industrial Taxonomy & Domain Specialization
Designed for multi-class defect classification:
1. `nominal`: Defect-free surface.
2. `scratch` / `crack`: Filamentary structural fracture.
3. `pit` / `inclusion`: Compact localized flaw.
4. `patch` / `contamination`: Diffuse surface blemish.

## Validation Protocol & Limitations
- **Primary Use:** Tier 2 diagnostic escalation following Tier 1 anomaly detection.
- **Confidence Gating:** Inferences with maximum softmax probability $< 0.70$ or normalized entropy $> 0.85$ are flagged as uncertain and escalated to manual review.
- **Domain Constraint:** Must not be used on domains outside its calibrated distribution without domain adaptation.

