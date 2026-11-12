# Architectural & Technical Decision Records (ADR) — Industrial Inspection System

This document logs significant architectural, algorithmic, and engineering decisions made throughout the project lifecycle.

---

## ADR-001: Strict Domain Separation (Glass Bottle vs Steel Sheet)
- **Context:** The system inspects both glass bottles and planar steel sheets. Glass bottles are curved 3D objects where anomalies represent structural damage (chips, cracks, breaks) or cosmetic contamination. Planar steel sheets are 2D continuous surfaces with rolling defects (scratches, patches, inclusions).
- **Decision:** Enforce absolute domain separation in the inspection pipeline:
  - `GLASS_BOTTLE`: Evaluated via Tier 1 PatchCore anomaly detection. Metric scaling is explicitly labeled as *projected sensor plane* ($s^2 \times A_{\text{px}}$) with an explicit disclaimer regarding 3D curvature distortion.
  - `STEEL_SHEET`: Evaluated via Tier 2 classifier, Tier 3 YOLOv8, and Tier 3 U-Net. Metric scaling is true planar ($s^2 \times A_{\text{px}}$).
  - Steel-trained models are never automatically executed on glass bottles, and vice-versa.
- **Consequences:** Eliminates catastrophic false defect classifications and maintains physical measurement integrity.

---

## ADR-002: Deterministic Decision Hierarchy Precedence
- **Context:** In multi-tier inspection, different model tiers can produce conflicting signals (e.g., Tier 1 PatchCore reports high anomaly, but Tier 3 segmentation measures a defect area smaller than the critical threshold due to speckle noise).
- **Decision:** Formalize a strict hierarchical decision tree:
  1. If anomaly score exceeds $\tau_{\text{critical}}$, verdict is `FAIL` regardless of subthreshold segmented area.
  2. If an unknown/unclassified defect type is detected above threshold, triage routes to `REVIEW` for human bay inspection.
  3. Physical area and spatial risk multipliers refine severity level (`CRITICAL`, `MEDIUM`, `LOW`, `NONE`) and conveyor routing (`AUTOMATIC_REJECT_BIN`, `MANUAL_INSPECTION_BAY`, `LINE_CONTINUE`).
- **Consequences:** Guarantees deterministic, safety-critical fail-safe behavior on the factory floor.

---

## ADR-003: Cryptographic HMAC Key Lifecycle & Key Identification (`key_id`)
- **Context:** Earlier versions used a single static HMAC key. In production, security policies require regular cryptographic key rotation (e.g., annually). Rotating the key would break verification of past historical records.
- **Decision:** 
  1. Secret keys are strictly loaded from environment variables (`AOI_HMAC_SECRET_KEY`) or cloud secrets managers, never stored in the relational database.
  2. Every inspection certificate and database row explicitly stores a `key_id` (e.g. `KEY-2026-V1`).
  3. `Settings.KEY_REGISTRY` maintains an active mapping of key identifiers to secret bytes (`{"KEY-2026-V1": b"...", "KEY-2025-V0": b"..."}`).
  4. Cryptographic verification dynamically resolves the signing key via `key_id`, ensuring historical auditability across key rotation epochs.
- **Consequences:** Full compliance with industrial audit standards without sacrificing key rotation agility.

---

## ADR-004: Soft Archival Retention vs Permanent Audit Immutability
- **Context:** Hard deletion (`DELETE /inspections/{id}`) directly violates regulatory compliance mandates (FDA 21 CFR Part 11, IATF 16949, ISO 9001) which require permanent retention of inspection records.
- **Decision:**
  1. Replace hard deletion with soft archival: records have `record_status` (`ACTIVE`, `ARCHIVED`, `LEGAL_HOLD`), `archived_at`, `archived_by`, and `archive_reason`.
  2. Hard deletion is completely disabled in production compliance mode (`ALLOW_HARD_DELETE=False`), returning `405 Method Not Allowed`.
  3. Standard history queries and analytics automatically filter out archived records, while auditors can query archived history via `include_archived=true`.
  4. Archival actions are immutably logged to `audit_log_records`.
- **Consequences:** Complete audit trail permanence with zero data loss risk.

---

## ADR-005: Decoupled Relative Storage URIs
- **Context:** Storing absolute filesystem paths (e.g. `D:\CSE_471\...\data\inspection_media\...`) creates brittle database records that break upon migration to Docker, Linux containers, or S3/Blob storage.
- **Decision:** Store relative URI keys (`GLASS_BOTTLE/2026-11-12/CERT-..._annotated.jpg`) in the database, resolving absolute paths dynamically at runtime relative to `settings.STORAGE_DIR`.
- **Consequences:** Zero-effort portability across containerized and cloud environments.

---

## ADR-006: Bounded Concurrency & Partial Failure Isolation
- **Context:** Batch image uploads can cause memory exhaustion (OOM) if multiple deep learning inference jobs run simultaneously. Additionally, a single corrupted file should not abort an entire batch of 30+ good items.
- **Decision:**
  1. Enforce `anyio.CapacityLimiter(4)` to strictly throttle concurrent threadpool inference.
  2. Enforce strict upload bounds: max 15 MB per file, max 32 files per batch.
  3. Implement item-level transaction and error handling: corrupted files produce per-item error status without failing the remaining valid items.
- **Consequences:** Robust, high-throughput batching resilient to adversarial or malformed inputs.

