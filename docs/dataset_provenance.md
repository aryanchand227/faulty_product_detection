# Dataset Provenance, Attribution, and Licensing

## 1. Primary Dataset: MVTec Anomaly Detection (MVTec AD)

### 1.1 Official Attribution and Citation
The MVTec Anomaly Detection benchmark was created by **MVTec Software GmbH** (Munich, Germany).

If using this dataset in research or academic work, cite the foundational papers:
```bibtex
@inproceedings{bergmann2019mvtec,
  title={MVTec AD--A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection},
  author={Bergmann, Paul and Fauser, Michael and Sattlegger, David and Steger, Carsten},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={9592--9600},
  year={2019}
}

@article{bergmann2021mvtec,
  title={The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection},
  author={Bergmann, Paul and Batzner, Kilian and Fauser, Michael and Sattlegger, David and Steger, Carsten},
  journal={International Journal of Computer Vision (IJCV)},
  volume={129},
  number={4},
  pages={1038--1059},
  year={2021}
}
```

Official dataset website: [https://www.mvtec.com/company/research/datasets/mvtec-ad](https://www.mvtec.com/company/research/datasets/mvtec-ad)

---

### 1.2 Official License and Usage Terms
- **License:** [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
- **Academic and Educational Use:** Permitted for non-commercial academic research, coursework, and personal learning.
- **Commercial Use:** Strictly prohibited without an explicit commercial license agreement directly from MVTec Software GmbH.
- **Redistribution Policy:** Under CC BY-NC-SA 4.0, any shared copy must preserve the original copyright notice, disclaimers, and license. **We do not upload or redistribute the raw dataset to public cloud repositories or model hubs.**

---

### 1.3 Download Source and Mirror Integrity
For developer convenience and faster local category downloads, the `bottle` subset was fetched via the Hugging Face dataset mirror `foersben/mvtec-ad`.

#### Verification against Official Benchmark Specification:
We verified the file structure against the official MVTec AD specification:
- `train/good`: Exactly 209 images (Official count: 209)
- `test/good`: Exactly 20 images (Official count: 20)
- `test/broken_large`: Exactly 20 images (Official count: 20)
- `test/broken_small`: Exactly 22 images (Official count: 22)
- `test/contamination`: Exactly 21 images (Official count: 21)
- `ground_truth/broken_large`: Exactly 20 pixel masks (Official count: 20)
- `ground_truth/broken_small`: Exactly 22 pixel masks (Official count: 22)
- `ground_truth/contamination`: Exactly 21 pixel masks (Official count: 21)

**Integrity Status:** **100% verified match**. All 292 product images and 63 ground-truth masks (355 total PNG files) match the official MVTec AD bottle benchmark.

**Important Legal Note on Third-Party Mirrors:**
Third-party mirrors (such as Hugging Face community datasets or Kaggle uploads) do **not** grant additional redistribution or commercial rights. The dataset remains governed exclusively by the original CC BY-NC-SA 4.0 license from MVTec Software GmbH.

---

## 2. Secondary Datasets: VisA & NEU Surface Defect

### 2.1 VisA (Visual Anomaly)
- **Creators:** Amazon Science
- **License:** [CC BY 4.0](https://github.com/amazon-science/spot-diff)
- **Role:** Cross-category anomaly generalization testing (used in later stretch phases).

### 2.2 NEU Surface Defect Database
- **Creators:** Northeastern University (Shenyang, China) — Song & Yan (2013).
- **Role:** Supervised 6-class surface defect classification (Phase 4) and object detection (Phase 5).
- **Verified Sample Count:** Exactly **1,797 defect images** and **1,797 Pascal VOC XML files** (4,177 valid bounding boxes total).
- **Resolution:** Uniform $200 \times 200$ pixels.
- **Classes:** `crazing` (688 boxes), `inclusion` (1,010 boxes), `patches` (875 boxes), `pitted_surface` (430 boxes), `rolled-in_scale` (628 boxes), `scratches` (546 boxes).
- **Split Distribution:** Official ImageSets partition:
  - Train: 1,437 images
  - Val: 180 images
  - Test: 180 images
  - Overlap between splits: **0 (Strictly mutually exclusive image IDs)**.

#### Critical Engineering Limitation: Physical Specimen Independence
The NEU database provides image-level identifiers (`crazing_1.jpg`, `crazing_2.jpg`, etc.) but does **not** provide physical specimen or steel slab metadata.
If multiple image crops originated from the same physical hot-rolled steel strip, image-level splitting could place visually correlated crops into both the training and test sets.
In an industrial production deployment, data must be partitioned strictly by **heat/batch/coil ID** rather than random image crops to ensure genuine generalization across distinct production runs.

---

## 3. Cross-Domain Operational Reality (Bottles vs Steel Surfaces)

> [!WARNING]
> **Domain Shift Warning:**
> Our project investigates two distinct industrial manufacturing benchmarks:
> 1. **MVTec AD Bottle:** Transparent/translucent cylindrical glass containers with smooth contours.
> 2. **NEU Steel Strip:** Opaque, highly textured planar hot-rolled metal sheets.
>
> A model trained on normal glass bottle surfaces **cannot** be used to inspect steel strips, and vice versa.
> In our software architecture ([`src/config.py`](file:///d:/CSE_471/industrial-damage-detection/src/config.py) and FastAPI backend), inspection pipelines are strictly partitioned by `product_type`:
> - `product_type="bottle"`: Routes to MVTec-trained models (PatchCore Anomaly Detection + U-Net defect segmentation).
> - `product_type="steel_strip"`: Routes to NEU-trained models (Supervised Multi-Class Classifier + YOLO Object Detection).

