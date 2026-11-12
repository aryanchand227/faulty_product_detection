# External Tasks & User Action Register — Industrial Damage Detection

This register explicitly separates independent system engineering from external tasks requiring physical access, GPU hardware (Google Colab), or external user secrets.

**Status Legend:**
- `NOT REQUIRED`: Task is fully satisfied by local CPU inference, synthetic validation, or pre-existing published artifacts.
- `READY FOR USER`: Everything is implemented and packaged; user may execute at their convenience.
- `WAITING FOR USER`: Execution is currently blocking downstream task completion.
- `USER COMPLETED`: User has run the task and returned the artifacts/outputs.
- `VERIFIED`: Outputs have been inspected, integrated into the repository, and verified passing tests.

---

## Action Task Index

| Task ID | Phase | Component | Status | Description |
| :--- | :--- | :--- | :---: | :--- |
| **UAR-001** | Phase 7 | PatchCore Bottle Training | **VERIFIED** | Model trained on Tesla T4, evaluated, and published to Hugging Face Hub (`TAC13/industrial-damage-patchcore-bottle`). |
| **UAR-002** | Phase 14 | Model Card & HF Hub Verification | **VERIFIED** | HF Hub repository contains model checkpoint, ablation logs, evaluation figures, and model card. |
| **UAR-003** | Phase 4 | Optional: Fine-Tune Steel ResNet Classifier | **NOT REQUIRED** | Fallback ResNet-18 weights and heuristic classifier inference are implemented and fully functional locally. Standalone Colab notebook `training/colab/classification/03_train_classification_models.ipynb` is prepared if custom retraining is desired. |
| **UAR-004** | Phase 5 | Optional: Fine-Tune Steel YOLOv8n Detector | **NOT REQUIRED** | Pre-trained YOLOv8n weights (`yolov8n.pt`) and bounding box inference are operational. Standalone Colab notebook `training/colab/detection/05_train_yolo.ipynb` is ready. |
| **UAR-005** | Phase 6 | Optional: Fine-Tune Steel U-Net Segmenter | **NOT REQUIRED** | Lightweight U-Net architecture, forward/backward passes, and synthetic inference are tested locally. Standalone Colab notebook `training/colab/segmentation/06_train_unet.ipynb` is ready. |
| **UAR-006** | Phase 17 | Docker Hub Image Push (Optional) | **READY FOR USER** | Container configurations and Dockerfile are provided. Local build and execution can run directly via Docker Compose. |
| **UAR-007** | Phase 18 | Final GitHub Remote Push | **READY FOR USER** | Remote origin is configured to `https://github.com/aryanchand227/faulty_product_detection.git`. Final push requires user's Git push credentials. |

---

## Detailed Task Runbooks

### Task UAR-001: PatchCore Bottle Anomaly Model Training
- **Phase:** Phase 7 (Anomaly Detection)
- **Status:** **VERIFIED**
- **Outcome:** The PatchCore model was trained on a Google Colab Tesla T4 GPU using Wide-ResNet-50-2 with 10% greedy coreset subsampling. 
- **Metrics Achieved:** Image AUROC: 1.0000 | Pixel AUROC: 0.9848 | AUPRO: 0.9376 | Synchronized CUDA Latency: 26.25 ms (p50).
- **Artifact:** Uploaded and verified on Hugging Face Hub at `TAC13/industrial-damage-patchcore-bottle`.

---

### Task UAR-007: Final GitHub Repository Push
- **Phase:** Phase 18 (Delivery & Publishing)
- **Status:** **READY FOR USER**
- **Why Required:** Pushing the final audited, tested, and documented codebase to the user's GitHub remote repository requires user credentials or authorization token.
- **Repository Remote:** `https://github.com/aryanchand227/faulty_product_detection.git`
- **Exact Command:**
  ```powershell
  cd d:\CSE_471\industrial-damage-detection
  git push -u origin master
  ```
- **How to Verify:** Visit `https://github.com/aryanchand227/faulty_product_detection` and verify all 18 phases, code, tests, documentation, and frontend files are present on branch `master`.

