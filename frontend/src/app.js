import { api } from "./services/api.js";

// State Management
const state = {
  activeTab: "inspect",
  domain: "GLASS_BOTTLE",
  lastResult: null,
  historyData: [],
  analyticsData: null,
  modelRegistry: null,
  charts: {},
};

// DOM References
const els = {
  tabs: document.querySelectorAll(".nav-tab"),
  tabContents: document.querySelectorAll(".tab-content"),
  dropZone: document.getElementById("drop-zone"),
  fileInput: document.getElementById("file-input"),
  batchInput: document.getElementById("batch-input"),
  domainSelect: document.getElementById("domain-select"),
  inspectBtn: document.getElementById("inspect-btn"),
  batchBtn: document.getElementById("batch-btn"),
  apiKeyInput: document.getElementById("api-key-input"),
  saveApiKeyBtn: document.getElementById("save-api-key-btn"),
  verdictContainer: document.getElementById("verdict-container"),
  historyTbody: document.getElementById("history-tbody"),
  systemStatusBadge: document.getElementById("system-status-badge"),
};

// Initialize Dashboard
document.addEventListener("DOMContentLoaded", async () => {
  initNavigation();
  initUpload();
  initAuth();
  await loadSystemStatus();
  await loadHistory();
});

function initNavigation() {
  els.tabs.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tab;
      state.activeTab = target;

      els.tabs.forEach((b) => b.classList.remove("border-blue-600", "text-blue-600", "font-semibold"));
      btn.classList.add("border-blue-600", "text-blue-600", "font-semibold");

      els.tabContents.forEach((c) => {
        if (c.id === `tab-${target}`) {
          c.classList.remove("hidden");
        } else {
          c.classList.add("hidden");
        }
      });

      if (target === "history") loadHistory();
      if (target === "analytics") loadAnalytics();
      if (target === "system") loadSystemDiagnostics();
    });
  });
}

function initAuth() {
  els.apiKeyInput.value = api.apiKey;
  els.saveApiKeyBtn.addEventListener("click", () => {
    const key = els.apiKeyInput.value.trim();
    api.setApiKey(key);
    showToast("API Key updated in browser storage.", "success");
  });
}

function initUpload() {
  els.dropZone.addEventListener("click", () => els.fileInput.click());
  els.dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    els.dropZone.classList.add("border-blue-500", "bg-blue-50");
  });
  els.dropZone.addEventListener("dragleave", () => {
    els.dropZone.classList.remove("border-blue-500", "bg-blue-50");
  });
  els.dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    els.dropZone.classList.remove("border-blue-500", "bg-blue-50");
    if (e.dataTransfer.files.length > 0) {
      handleSingleFile(e.dataTransfer.files[0]);
    }
  });

  els.fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleSingleFile(e.target.files[0]);
    }
  });

  els.batchInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      handleBatchFiles(Array.from(e.target.files));
    }
  });

  els.domainSelect.addEventListener("change", (e) => {
    state.domain = e.target.value;
  });
}

let selectedFile = null;
function handleSingleFile(file) {
  selectedFile = file;
  document.getElementById("selected-filename").textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  document.getElementById("upload-preview-container").classList.remove("hidden");

  const reader = new FileReader();
  reader.onload = (e) => {
    document.getElementById("original-image-preview").src = e.target.result;
  };
  reader.readAsDataURL(file);

  els.inspectBtn.disabled = false;
}

els.inspectBtn.addEventListener("click", async () => {
  if (!selectedFile) return;
  els.inspectBtn.disabled = true;
  els.inspectBtn.innerHTML = `<span class="animate-spin inline-block mr-2">⚙️</span> Processing...`;

  try {
    const result = await api.inspectImage(selectedFile, state.domain, {
      renderVisuals: true,
      extractMorphology: true,
      generateCertificate: true,
      persistToDb: true,
    });
    state.lastResult = result;
    renderInspectionResults(result);
    showToast("Inspection completed and record persisted to audit log.", "success");
  } catch (err) {
    showToast(`Inspection failed: ${err.message}`, "error");
  } finally {
    els.inspectBtn.disabled = false;
    els.inspectBtn.innerHTML = `Execute Multi-Tier Inspection`;
  }
});

async function handleBatchFiles(files) {
  if (files.length === 0) return;
  showToast(`Uploading batch of ${files.length} images...`, "info");
  try {
    const res = await api.inspectBatch(files, state.domain, true);
    showToast(`Batch completed: ${res.successful_inspections} succeeded, ${res.failed_inspections} failed.`, "success");
    loadHistory();
  } catch (err) {
    showToast(`Batch execution error: ${err.message}`, "error");
  }
}

function renderInspectionResults(data) {
  els.verdictContainer.classList.remove("hidden");
  const verdict = data.verdict || {};
  const cert = data.certificate || {};

  // Decision badge styling
  const badge = document.getElementById("verdict-badge");
  badge.textContent = verdict.decision;
  badge.className = "text-xl font-bold px-4 py-1 rounded-full uppercase ";
  if (verdict.decision === "PASS") {
    badge.classList.add("bg-green-100", "text-green-800", "border", "border-green-300");
  } else if (verdict.decision === "REVIEW") {
    badge.classList.add("bg-amber-100", "text-amber-800", "border", "border-amber-300");
  } else {
    badge.classList.add("bg-red-100", "text-red-800", "border", "border-red-300");
  }

  // Metrics
  document.getElementById("metric-severity").textContent = verdict.severity || "NONE";
  document.getElementById("metric-routing").textContent = verdict.routing || "LINE_CONTINUE";
  document.getElementById("metric-anomaly").textContent = (verdict.anomaly_score || 0).toFixed(2);
  document.getElementById("metric-area").textContent = `${(cert.total_defect_area_mm2 || 0).toFixed(2)} mm²`;
  document.getElementById("metric-morphology").textContent = cert.dominant_morphology || "NONE";
  document.getElementById("metric-latency").textContent = `${(data.total_latency_ms || 0).toFixed(1)} ms`;

  // HUD Image
  if (data.annotated_image_base64) {
    document.getElementById("hud-image-preview").src = data.annotated_image_base64;
    document.getElementById("hud-container").classList.remove("hidden");
  } else {
    document.getElementById("hud-container").classList.add("hidden");
  }

  // Defect Components Table
  const tbody = document.getElementById("defect-components-tbody");
  tbody.innerHTML = "";
  const components = cert.defect_components || [];
  if (components.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" class="p-3 text-center text-gray-500">No segmented defect contours detected (Nominal/Early-Exit).</td></tr>`;
  } else {
    components.forEach((c) => {
      const tr = document.createElement("tr");
      tr.className = "border-b text-sm";
      tr.innerHTML = `
        <td class="p-2 font-mono">#${c.contour_id}</td>
        <td class="p-2">${c.morphology_type}</td>
        <td class="p-2 font-semibold text-gray-800">${c.physical_area_mm2.toFixed(2)} mm²</td>
        <td class="p-2">${c.aspect_ratio.toFixed(2)}</td>
        <td class="p-2">${c.circularity.toFixed(3)}</td>
        <td class="p-2"><span class="px-2 py-0.5 rounded text-xs bg-gray-100">${c.spatial_zone}</span></td>
        <td class="p-2 font-mono text-xs">${c.heuristic_severity_score.toFixed(1)}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  // Certificate Box
  document.getElementById("cert-id").textContent = cert.certificate_id;
  document.getElementById("cert-key-id").textContent = cert.key_id || "KEY-2026-V1";
  document.getElementById("cert-hash").textContent = cert.sha256_integrity_hash || "N/A";
  document.getElementById("cert-signature").textContent = cert.hmac_sha256_signature || "N/A";

  // Certificate Verification Button
  document.getElementById("verify-cert-btn").onclick = async () => {
    try {
      const ver = await api.verifyRecordIntegrity(cert.certificate_id);
      if (ver.is_valid) {
        showToast("✓ Cryptographic Signature & Hash 100% Authentic and Untampered", "success");
      } else {
        showToast("⚠ Tamper Detected or Invalid Cryptographic Signature!", "error");
      }
    } catch (e) {
      showToast(`Verification error: ${e.message}`, "error");
    }
  };

  // Export Buttons
  document.getElementById("export-json-btn").onclick = () => {
    downloadFile(`${cert.certificate_id}.json`, JSON.stringify(cert, null, 2), "application/json");
  };
  document.getElementById("export-md-btn").onclick = async () => {
    try {
      const md = await api.exportCertificate(cert.certificate_id, "markdown");
      downloadFile(`${cert.certificate_id}.md`, md, "text/markdown");
    } catch (e) {
      showToast("Markdown export error: " + e.message, "error");
    }
  };
}

async function loadHistory() {
  try {
    const data = await api.listInspections({ limit: 50 });
    state.historyData = data.items || [];
    renderHistoryTable(state.historyData);
  } catch (err) {
    els.historyTbody.innerHTML = `<tr><td colspan="8" class="p-4 text-center text-red-500">Failed to load history: ${err.message}</td></tr>`;
  }
}

function renderHistoryTable(items) {
  els.historyTbody.innerHTML = "";
  if (items.length === 0) {
    els.historyTbody.innerHTML = `<tr><td colspan="8" class="p-4 text-center text-gray-500">No inspection records logged.</td></tr>`;
    return;
  }

  items.forEach((item) => {
    const tr = document.createElement("tr");
    tr.className = "border-b hover:bg-gray-50 text-sm";

    let decBadge = `<span class="px-2 py-0.5 rounded text-xs font-bold uppercase `;
    if (item.decision === "PASS") decBadge += `bg-green-100 text-green-800">PASS</span>`;
    else if (item.decision === "REVIEW") decBadge += `bg-amber-100 text-amber-800">REVIEW</span>`;
    else decBadge += `bg-red-100 text-red-800">FAIL</span>`;

    const ts = new Date(item.timestamp_utc).toLocaleString();

    tr.innerHTML = `
      <td class="p-3 font-mono text-xs">${item.certificate_id}</td>
      <td class="p-3">${ts}</td>
      <td class="p-3 font-mono text-xs">${item.product_domain}</td>
      <td class="p-3">${decBadge}</td>
      <td class="p-3">${item.anomaly_score.toFixed(2)}</td>
      <td class="p-3">${item.total_defect_area_mm2.toFixed(2)} mm²</td>
      <td class="p-3">${item.total_latency_ms.toFixed(1)} ms</td>
      <td class="p-3 space-x-1">
        <button class="px-2 py-1 bg-gray-100 hover:bg-gray-200 rounded text-xs view-btn" data-id="${item.id}">View</button>
        <button class="px-2 py-1 bg-blue-50 text-blue-600 hover:bg-blue-100 rounded text-xs verify-btn" data-id="${item.id}">Verify</button>
        <button class="px-2 py-1 bg-red-50 text-red-600 hover:bg-red-100 rounded text-xs archive-btn" data-id="${item.id}">Archive</button>
      </td>
    `;

    tr.querySelector(".view-btn").onclick = async () => {
      const detail = await api.getInspectionDetail(item.id);
      renderInspectionResults(detail);
      // Switch to inspect tab
      document.querySelector('[data-tab="inspect"]').click();
    };

    tr.querySelector(".verify-btn").onclick = async () => {
      const res = await api.verifyRecordIntegrity(item.id);
      if (res.is_valid) {
        showToast(`Record ${item.certificate_id} HMAC verified authentic.`, "success");
      } else {
        showToast(`Record ${item.certificate_id} integrity mismatch!`, "error");
      }
    };

    tr.querySelector(".archive-btn").onclick = async () => {
      const reason = prompt("Enter justification for regulatory archiving:", "Periodic retention management");
      if (reason) {
        try {
          await api.archiveInspection(item.id, reason);
          showToast(`Record ${item.certificate_id} successfully archived.`, "success");
          loadHistory();
        } catch (e) {
          showToast(e.message, "error");
        }
      }
    };

    els.historyTbody.appendChild(tr);
  });
}

async function loadAnalytics() {
  try {
    const data = await api.getAnalytics(30);
    state.analyticsData = data;

    document.getElementById("stat-total").textContent = data.total_inspections;
    document.getElementById("stat-pass-rate").textContent = `${data.pass_rate_pct}%`;
    document.getElementById("stat-review-rate").textContent = `${data.review_rate_pct}%`;
    document.getElementById("stat-fail-rate").textContent = `${data.fail_rate_pct}%`;
    document.getElementById("stat-early-exit").textContent = `${data.early_exit_rate_pct}%`;
    document.getElementById("stat-mean-latency").textContent = `${data.mean_latency_ms} ms`;

    if (data.latency_metrics) {
      document.getElementById("stat-p50").textContent = `${data.latency_metrics.median_p50_ms} ms`;
      document.getElementById("stat-p95").textContent = `${data.latency_metrics.p95_ms} ms`;
      document.getElementById("stat-p99").textContent = `${data.latency_metrics.p99_ms} ms`;
    }

    renderCharts(data);
  } catch (err) {
    showToast("Failed to load analytics: " + err.message, "error");
  }
}

function renderCharts(data) {
  // 1. Defect Category Distribution Bar Chart
  const defectCtx = document.getElementById("defect-dist-chart").getContext("2d");
  const defectLabels = Object.keys(data.defect_type_distribution || {});
  const defectValues = Object.values(data.defect_type_distribution || {});

  if (state.charts.defectChart) state.charts.defectChart.destroy();
  state.charts.defectChart = new Chart(defectCtx, {
    type: "bar",
    data: {
      labels: defectLabels.length ? defectLabels : ["None"],
      datasets: [{
        label: "Detected Instances",
        data: defectValues.length ? defectValues : [0],
        backgroundColor: "rgba(59, 130, 246, 0.7)",
        borderColor: "rgba(59, 130, 246, 1)",
        borderWidth: 1,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, ticks: { precision: 0 } } },
    },
  });

  // 2. Spatial Zone Breakdown Chart
  const zoneCtx = document.getElementById("zone-dist-chart").getContext("2d");
  const zoneLabels = Object.keys(data.spatial_zone_distribution || {});
  const zoneValues = Object.values(data.spatial_zone_distribution || {});

  if (state.charts.zoneChart) state.charts.zoneChart.destroy();
  state.charts.zoneChart = new Chart(zoneCtx, {
    type: "doughnut",
    data: {
      labels: zoneLabels.length ? zoneLabels : ["No Defect Zones"],
      datasets: [{
        data: zoneValues.length ? zoneValues : [1],
        backgroundColor: [
          "#EF4444", "#F59E0B", "#3B82F6", "#10B981", "#8B5CF6"
        ],
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { position: "right" } },
    },
  });
}

async function loadSystemStatus() {
  try {
    const health = await api.checkHealth();
    if (health.status === "healthy" && health.database_connected) {
      els.systemStatusBadge.className = "px-3 py-1 bg-green-100 text-green-800 rounded-full text-xs font-semibold flex items-center";
      els.systemStatusBadge.innerHTML = `<span class="w-2 h-2 rounded-full bg-green-500 mr-1.5 animate-pulse"></span> Service Operational`;
    } else {
      els.systemStatusBadge.className = "px-3 py-1 bg-amber-100 text-amber-800 rounded-full text-xs font-semibold";
      els.systemStatusBadge.textContent = "Service Degraded";
    }
  } catch (e) {
    els.systemStatusBadge.className = "px-3 py-1 bg-red-100 text-red-800 rounded-full text-xs font-semibold";
    els.systemStatusBadge.textContent = "Service Offline";
  }
}

async function loadSystemDiagnostics() {
  try {
    const sysInfo = await api.getSystemInfo();
    const models = await api.getModelRegistry();
    const thresholds = await api.getThresholds();

    document.getElementById("diag-device").textContent = sysInfo.device;
    document.getElementById("diag-version").textContent = sysInfo.version;
    document.getElementById("diag-cfg-version").textContent = sysInfo.config_version;
    document.getElementById("diag-scale").textContent = `${sysInfo.active_calibration_scale} mm/px`;

    // Populate Thresholds form
    document.getElementById("thresh-anomaly").value = thresholds.anomaly_threshold;
    document.getElementById("thresh-margin").value = thresholds.anomaly_margin;
    document.getElementById("thresh-max-area").value = thresholds.max_acceptable_area_mm2;
    document.getElementById("thresh-critical-area").value = thresholds.critical_area_mm2;

    // Model Cards
    const container = document.getElementById("models-container");
    container.innerHTML = "";
    (models.models || []).forEach((m) => {
      const card = document.createElement("div");
      card.className = "p-4 bg-white border rounded-lg shadow-sm";
      card.innerHTML = `
        <div class="flex justify-between items-center mb-2">
          <span class="text-xs font-bold uppercase tracking-wider text-blue-600">${m.tier}</span>
          <span class="px-2 py-0.5 rounded text-xs ${m.is_loaded ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-600'}">
            ${m.is_loaded ? 'Ready' : 'Standby'}
          </span>
        </div>
        <h4 class="font-bold text-gray-800">${m.model_name}</h4>
        <p class="text-xs text-gray-500 font-mono mt-1">${m.architecture}</p>
        <p class="text-xs text-gray-400 mt-2 truncate">Source: ${m.checkpoint_source}</p>
      `;
      container.appendChild(card);
    });

    // Save Thresholds Event
    document.getElementById("save-thresholds-btn").onclick = async () => {
      const payload = {
        anomaly_threshold: parseFloat(document.getElementById("thresh-anomaly").value),
        anomaly_margin: parseFloat(document.getElementById("thresh-margin").value),
        max_acceptable_area_mm2: parseFloat(document.getElementById("thresh-max-area").value),
        critical_area_mm2: parseFloat(document.getElementById("thresh-critical-area").value),
      };
      try {
        await api.updateThresholds(payload);
        showToast("Operational thresholds updated and audit log entry created.", "success");
      } catch (err) {
        showToast(err.message, "error");
      }
    };
  } catch (err) {
    showToast("Failed to load diagnostics: " + err.message, "error");
  }
}

function showToast(message, type = "info") {
  const toast = document.createElement("div");
  let bg = "bg-gray-800 text-white";
  if (type === "success") bg = "bg-green-700 text-white";
  if (type === "error") bg = "bg-red-700 text-white";
  if (type === "info") bg = "bg-blue-700 text-white";

  toast.className = `fixed bottom-5 right-5 px-4 py-2.5 rounded-lg shadow-lg text-sm z-50 transition-opacity duration-300 ${bg}`;
  toast.textContent = message;
  document.body.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = "0";
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

function downloadFile(filename, content, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

