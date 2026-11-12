# Industrial Product Damage Detection & Intelligent Inspection System (AOI)

[![Python 3.12](https://img.shields.io/badge/Python-3.12.10-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C.svg)](https://pytorch.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-336791.svg)](https://www.postgresql.org/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade, modular, cryptographically auditable **Automated Optical Inspection (AOI)** microservice for high-speed industrial manufacturing lines. The system integrates deep anomaly detection, defect classification, bounding box localization, pixel-level contour segmentation, 2D morphological analysis, functional spatial zoning, camera geometric calibration, and HMAC-SHA256 signed compliance certificates.

---

## 1. System Architecture

```mermaid
flowchart TD
    Camera["High-Speed Camera Frame (RGB)"] --> Ingestion["Input Validation & Preprocessing (15MB Limit, Resizing)"]
    Ingestion --> DomainRouter{"Product Domain Router"}
    
    DomainRouter -->|GLASS_BOTTLE| Tier1["Tier 1: PatchCore Anomaly Gatekeeper (Wide-ResNet-50-2 + 10% Coreset)"]
    DomainRouter -->|STEEL_SHEET| Tier23["Tier 2/3: YOLOv8 Detector & U-Net Segmenter"]
    
    Tier1 --> NominalCheck{"Anomaly Score < tau?"}
    NominalCheck -->|Yes: Nominal (Early Exit)| PassVerdict["Verdict: PASS (Conveyor: LINE_CONTINUE, Latency < 27ms)"]
    NominalCheck -->|No: Anomaly Detected| MorphEngine["Defect Morphology & Geometric Analyzer"]
    
    Tier23 --> MorphEngine
    
    MorphEngine --> SpatialZoning["Functional Spatial Zoning & Risk Weighting (Sealing Rim, Neck, Sidewall, Base)"]
    SpatialZoning --> DecEngine["Decision Precedence Engine (PASS / REVIEW / FAIL)"]
    
    DecEngine --> HmacCert["Cryptographic HMAC-SHA256 Certificate Generator (Key Rotation Registry)"]
    HmacCert --> RelationalDB[("SQLAlchemy / Alembic DB (SQLite / PostgreSQL 16)")]
    HmacCert --> RestAPI["FastAPI REST Endpoints & Web Dashboard"]
```

---

## 2. Key Capabilities & Engineering Highlights

- **Multi-Tier Quality Triage**:
  - **Tier 1 (Anomaly Detection)**: PatchCore with Wide-ResNet-50-2 and 10% greedy coreset subsampling ($16,386$ prototypes). Evaluated on MVTec Bottle achieving **1.0000 Image AUROC**, **0.9848 Pixel AUROC**, **0.9376 AUPRO**, and **26.25 ms p50 latency**.
  - **Tier 2 (Classification)**: ResNet-18 transfer learning with entropy uncertainty gating.
  - **Tier 3 (Localization & Segmentation)**: YOLOv8n object detection and U-Net pixel-level semantic segmentation with combined BCE + Dice loss.
- **Defect Morphology & Geometric Analysis**:
  - Extracts 2D descriptors: Area ($mm^2$), Perimeter ($mm$), Aspect Ratio ($\lambda_1 / \lambda_2$), Circularity ($4\pi A / P^2$), Solidity ($A / A_{\text{hull}}$), and Maximum Feret Diameter.
  - Mitigates rasterization bias on sub-30-pixel defects with automated unreliability flags.
- **Functional Spatial Zoning**:
  - Dynamically weights defect severity based on functional stress zones (e.g. Sealing Rim multiplier $2.5\times$, Base Heel $1.5\times$, Sidewall $1.0\times$).
- **Cryptographic Audit Trail (FDA 21 CFR Part 11 / ISO 9001 Compliance)**:
  - Every inspection produces an authenticated HMAC-SHA256 signed certificate.
  - **Key Lifecycle Management**: Stamped with `key_id` (e.g. `KEY-2026-V1`). Secrets stored strictly outside the database in environment variables or cloud secrets managers. Supports historical key registry rotation.
  - **Permanent Audit Immutability**: Production mode prohibits hard deletion (`ALLOW_HARD_DELETE=False`), providing soft archival retention (`ACTIVE`, `ARCHIVED`, `LEGAL_HOLD`) with immutable actor logging.
- **Factory Quality Analytics & Concurrency Controls**:
  - Live calculation of factory yield (Pass %, Review %, Reject %, Early-Exit %).
  - Latency percentiles ($p_{50}, p_{95}, p_{99}$).
  - Bounded concurrency semaphore (`anyio.CapacityLimiter`) and per-item partial batch failure isolation.
- **Interactive Web Dashboard**:
  - Responsive single-page inspection GUI served directly at `/dashboard/`.
  - Supports live camera frame upload, batch upload, HUD visual overlay, morphology tables, audit history, and real-time Chart.js KPI graphs.

---

## 3. Environment & Hardware Blueprint

- **Development OS:** Windows 11 (PowerShell, VS Code, Python 3.12.10).
- **Local Compute:** Intel/AMD CPU-only execution (strictly used for API development, database operations, testing, frontend, and Docker orchestration).
- **Model Training Environment:** Google Colab with NVIDIA Tesla T4 GPU (15 GB VRAM).
- **Model Hub Storage:** Hugging Face Hub under organization `TAC13`:
  - Bottle PatchCore: [`TAC13/industrial-damage-patchcore-bottle`](https://huggingface.co/TAC13/industrial-damage-patchcore-bottle)
  - Defect Classification: [`TAC13/industrial-damage-classification`](https://huggingface.co/TAC13/industrial-damage-classification)
  - YOLOv8n Detection: [`TAC13/industrial-damage-yolov8n`](https://huggingface.co/TAC13/industrial-damage-yolov8n)
  - Bottle U-Net: [`TAC13/industrial-damage-unet-bottle`](https://huggingface.co/TAC13/industrial-damage-unet-bottle)

---

## 4. Quickstart Guide (Local Development)

### 4.1. Prerequisites & Environment Setup

```powershell
# Clone the repository
git clone https://github.com/aryanchand227/faulty_product_detection.git
cd faulty_product_detection

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install project dependencies
pip install -r requirements.txt
```

### 4.2. Database Migrations

The project uses Alembic for declarative database versioning supporting both SQLite and PostgreSQL 16:

```powershell
# Run all database migrations up to head (0001 initial -> 0002 provenance & soft delete)
alembic upgrade head
```

### 4.3. Launch the Backend Microservice & Dashboard

```powershell
# Start Uvicorn development server
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Interactive Web Dashboard:** [http://localhost:8000/dashboard/](http://localhost:8000/dashboard/)
- **Swagger / OpenAPI Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check Endpoint:** [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

---

## 5. Running Automated Tests

The repository maintains an extensive test suite with **137 passing automated tests** spanning unit, integration, cryptographic, database, and PostgreSQL DDL compatibility:

```powershell
# Run the complete test suite
pytest -v

# Run database and security tests specifically
pytest tests/test_database.py tests/test_history_and_analytics_api.py tests/test_postgres_compatibility.py -v
```

---

## 6. Docker & Containerized Deployment

Deploy the complete multi-container production stack (FastAPI backend + PostgreSQL 16 database + persistent media volume) with a single command:

```powershell
# Build and run containers in detached mode
docker compose up --build -d

# Check service logs
docker compose logs -f aoi-backend

# Tear down containers
docker compose down
```

The container automatically applies Alembic migrations on startup via `entrypoint.sh` before spawning production Uvicorn workers.

---

## 7. Performance Benchmarks

### 7.1. PatchCore Model Latency (Google Colab Tesla T4 GPU)
- **Mean Latency:** $26.31\text{ ms}$
- **Median Latency ($p_{50}$):** $26.25\text{ ms}$
- **95th Percentile ($p_{95}$):** $27.51\text{ ms}$
- **Throughput:** $\approx 38.0\text{ FPS}$

### 7.2. Database Query Load Benchmark (1,000 Records)
Measured over 100 timed iterations per query pattern:

| Query Operation | Mean Latency | Median ($p_{50}$) | 95th %ile ($p_{95}$) | Throughput (QPS) |
| :--- | :---: | :---: | :---: | :---: |
| **Indexed Primary Key Lookup** | $1.81\text{ ms}$ | $1.59\text{ ms}$ | $2.71\text{ ms}$ | **$552.1\text{ QPS}$** |
| **Indexed Certificate ID Lookup** | $1.57\text{ ms}$ | $1.35\text{ ms}$ | $2.34\text{ ms}$ | **$637.3\text{ QPS}$** |
| **Multi-Filter Paginated Query** | $5.42\text{ ms}$ | $4.54\text{ ms}$ | $7.13\text{ ms}$ | **$184.6\text{ QPS}$** |
| **Cryptographic HMAC Verification** | $2.07\text{ ms}$ | $1.93\text{ ms}$ | $3.12\text{ ms}$ | **$483.7\text{ QPS}$** |
| **Factory Analytics KPI Aggregation** | $12.97\text{ ms}$ | $12.54\text{ ms}$ | $15.21\text{ ms}$ | **$77.1\text{ QPS}$** |

---

## 8. Role-Based Access Control (RBAC) API Matrix

The microservice secures endpoints using constant-time API key verification (`X-API-Key` header):

| Endpoint | HTTP Method | Role Required | Description |
| :--- | :---: | :---: | :--- |
| `/api/v1/inspect/image` | `POST` | `OPERATOR` | Single image inspection with HUD rendering |
| `/api/v1/inspect/batch` | `POST` | `OPERATOR` | Bounded batch inspection (up to 32 images) |
| `/api/v1/inspections` | `GET` | `AUDITOR` | Multi-filter historical query with pagination |
| `/api/v1/inspections/{id}` | `GET` | `AUDITOR` | Full inspection record with defect components |
| `/api/v1/inspections/{id}/verify`| `POST` | `AUDITOR` | Recompute HMAC-SHA256 from database rows |
| `/api/v1/inspections/{id}/certificate` | `GET` | `AUDITOR` | Export certificate as JSON or Markdown |
| `/api/v1/analytics/summary` | `GET` | `AUDITOR` | Factory quality KPIs & latency percentiles |
| `/api/v1/inspect/thresholds` | `PUT` | `ADMIN` | Update operational decision thresholds |
| `/api/v1/inspect/calibration`| `PUT` | `ADMIN` | Update camera metric calibration scale |
| `/api/v1/inspections/{id}/archive` | `POST` | `ADMIN` | Soft-archive record with regulatory justification |

---

## 9. Known Limitations & Technical Constraints

1. **Curved 3D Surface Distortion**:
   Physical damage area conversion uses planar projection ($A_{\text{mm}^2} = A_{\text{px}} \times s^2$). For cylindrical bottles, apparent defect area shrinks near the perimeter due to perspective and curvature distortion. True 3D area measurement requires multi-view photometric stereo or CAD mesh ray-tracing.
2. **Sub-30-Pixel Contour Rasterization**:
   Very small defects exhibit discrete pixel grid quantization where circularity values can exceed $1.0$ mathematically. The system automatically tags contours with `is_shape_reliable=False` when area $< 30\text{ px}$.
3. **Hardware Isolation**:
   Model training requires GPU acceleration (Tesla T4 via Google Colab). The local Windows development environment executes inference and backend services on CPU.

---

## 10. Repository Structure

```
industrial-damage-detection/
├── backend/                        # FastAPI Microservice & Database
│   ├── alembic/                    # Database version migrations (0001, 0002)
│   └── app/
│       ├── api/routes/             # Inspection, History, Analytics, Models
│       ├── core/                   # Config & RBAC Security gatekeeper
│       ├── database/               # SQLAlchemy models, CRUD, connection
│       ├── schemas/                # Pydantic v2 validation contracts
│       └── services/               # Inspection singleton service wrapper
├── deployment/                     # Production Deployment Configurations
│   └── docker/                     # Multi-stage Dockerfile & docker-compose.yml
├── docs/                           # Architecture, Trackers & User Guides
│   ├── MASTER_PROJECT_TRACKER.md   # Complete phase-by-phase project tracker
│   ├── USER_ACTION_REQUIRED.md     # External tasks register
│   ├── TECHNICAL_DECISIONS.md      # Architectural Decision Records (ADRs)
│   └── FINAL_PROJECT_DELIVERY_REPORT.md # Final project delivery report
├── frontend/                       # Interactive Web Dashboard
│   ├── index.html                  # Single-page application GUI
│   └── src/                        # ES6 components, services & charts
├── reports/                        # Model Cards, Benchmarks & Figures
│   ├── MODEL_CARD_PATCHCORE_BOTTLE.md
│   ├── MODEL_CARD_CLASSIFICATION.md
│   ├── MODEL_CARD_YOLO_DETECTION.md
│   ├── MODEL_CARD_UNET_SEGMENTATION.md
│   └── db_query_benchmark.json
├── scripts/                        # Benchmarks, Demos & Verification
│   ├── benchmark_db_queries.py     # 1,000-record query latency benchmark
│   └── demo_historical_db_and_analytics.py
├── src/                            # Core Machine Learning & Inspection Logic
│   ├── data/                       # Preprocessing & annotation converters
│   ├── inspection/                 # Unified pipeline, decision engine, HUD visualizer
│   └── models/                     # PyTorch network architectures
├── tests/                          # Automated Pytest Suite (137 tests)
├── alembic.ini                     # Alembic configuration
├── pyproject.toml                  # Python package configuration
└── requirements.txt                # Pinned production dependencies
```

---

## 11. Authors & License

- **Author:** Aryan Chand ([@aryanchand227](https://github.com/aryanchand227))
- **Hugging Face Hub:** [TAC13](https://huggingface.co/TAC13)
- **License:** MIT License. See `LICENSE` for details.
