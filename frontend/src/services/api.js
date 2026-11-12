/**
 * Industrial AOI Inspection Service API Client
 * Wraps all FastAPI v1 endpoints with error handling and authentication header injection.
 */

const API_BASE = "/api/v1";

class ApiService {
  constructor() {
    this.apiKey = localStorage.getItem("aoi_api_key") || "aoi-admin-secret-key-2026";
  }

  setApiKey(key) {
    this.apiKey = key;
    localStorage.setItem("aoi_api_key", key);
  }

  getHeaders(customHeaders = {}) {
    const headers = { ...customHeaders };
    if (this.apiKey) {
      headers["X-API-Key"] = this.apiKey;
    }
    return headers;
  }

  async checkHealth() {
    const res = await fetch(`${API_BASE}/health`);
    return res.json();
  }

  async inspectImage(file, domain = "GLASS_BOTTLE", options = {}) {
    const formData = new FormData();
    formData.append("file", file);

    const params = new URLSearchParams({
      domain,
      render_visuals: options.renderVisuals !== false,
      force_full_pipeline: options.forceFullPipeline || false,
      extract_morphology: options.extractMorphology !== false,
      generate_certificate: options.generateCertificate !== false,
      persist_to_db: options.persistToDb !== false,
    });

    const res = await fetch(`${API_BASE}/inspect/image?${params.toString()}`, {
      method: "POST",
      headers: this.getHeaders(),
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "Inspection request failed");
    }

    return res.json();
  }

  async inspectBatch(files, domain = "GLASS_BOTTLE", persistToDb = true) {
    const formData = new FormData();
    for (const file of files) {
      formData.append("files", file);
    }

    const params = new URLSearchParams({
      domain,
      persist_to_db: persistToDb,
    });

    const res = await fetch(`${API_BASE}/inspect/batch?${params.toString()}`, {
      method: "POST",
      headers: this.getHeaders(),
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "Batch inspection request failed");
    }

    return res.json();
  }

  async listInspections(params = {}) {
    const query = new URLSearchParams(params);
    const res = await fetch(`${API_BASE}/inspections?${query.toString()}`, {
      headers: this.getHeaders(),
    });
    if (!res.ok) throw new Error("Failed to load inspection history");
    return res.json();
  }

  async getInspectionDetail(id) {
    const res = await fetch(`${API_BASE}/inspections/${id}`, {
      headers: this.getHeaders(),
    });
    if (!res.ok) throw new Error(`Failed to load inspection #${id}`);
    return res.json();
  }

  async verifyRecordIntegrity(id) {
    const res = await fetch(`${API_BASE}/inspections/${id}/verify`, {
      method: "POST",
      headers: this.getHeaders(),
    });
    if (!res.ok) throw new Error("Verification request failed");
    return res.json();
  }

  async exportCertificate(id, format = "json") {
    const res = await fetch(`${API_BASE}/inspections/${id}/certificate?format=${format}`, {
      headers: this.getHeaders(),
    });
    if (format === "markdown") {
      return res.text();
    }
    return res.json();
  }

  async archiveInspection(id, reason, status = "ARCHIVED") {
    const res = await fetch(`${API_BASE}/inspections/${id}/archive`, {
      method: "POST",
      headers: this.getHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify({ reason, status }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "Archival failed");
    }
    return res.json();
  }

  async getAnalytics(days = 7, domain = null) {
    const params = new URLSearchParams({ days });
    if (domain) params.append("domain", domain);

    const res = await fetch(`${API_BASE}/analytics/summary?${params.toString()}`, {
      headers: this.getHeaders(),
    });
    if (!res.ok) throw new Error("Failed to load factory analytics");
    return res.json();
  }

  async getZoneHeatmap(domain = null) {
    const params = domain ? `?domain=${domain}` : "";
    const res = await fetch(`${API_BASE}/analytics/zone-heatmap${params}`, {
      headers: this.getHeaders(),
    });
    if (!res.ok) throw new Error("Failed to load zone heatmap");
    return res.json();
  }

  async getModelRegistry() {
    const res = await fetch(`${API_BASE}/models/registry`, {
      headers: this.getHeaders(),
    });
    return res.json();
  }

  async getSystemInfo() {
    const res = await fetch(`${API_BASE}/inspect/system-info`, {
      headers: this.getHeaders(),
    });
    return res.json();
  }

  async getThresholds() {
    const res = await fetch(`${API_BASE}/inspect/thresholds`, {
      headers: this.getHeaders(),
    });
    return res.json();
  }

  async updateThresholds(payload) {
    const res = await fetch(`${API_BASE}/inspect/thresholds`, {
      method: "PUT",
      headers: this.getHeaders({ "Content-Type": "application/json" }),
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "Threshold update unauthorized or invalid");
    }
    return res.json();
  }
}

export const api = new ApiService();

