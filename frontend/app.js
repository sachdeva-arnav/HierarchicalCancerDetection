const form = document.getElementById("predictionForm");
const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const previewPlaceholder = document.getElementById("previewPlaceholder");
const manualOverride = document.getElementById("manualOverride");
const organOverride = document.getElementById("organOverride");
const submitButton = document.getElementById("submitButton");
const reportButton = document.getElementById("reportButton");
const reportStatus = document.getElementById("reportStatus");
const requestState = document.getElementById("requestState");
const modelStatus = document.getElementById("modelStatus");
const finalDecisionPanel = document.getElementById("finalDecisionPanel");
const warningList = document.getElementById("warningList");
const modalityCard = document.getElementById("modalityCard");
const tissueCard = document.getElementById("tissueCard");
const normalityCard = document.getElementById("normalityCard");
const subtypeCard = document.getElementById("subtypeCard");
const gradcamPanel = document.getElementById("gradcamPanel");
const organChart = document.getElementById("organChart");
const subtypeChart = document.getElementById("subtypeChart");
const jsonOutput = document.getElementById("jsonOutput");

let latestResult = null;
let latestFilename = null;
let latestModelStatus = null;

function setRequestState(label, className = "") {
  requestState.textContent = label;
  requestState.className = `badge ${className}`.trim();
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function formatConfidence(value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) {
    return "N/A";
  }
  return `${(Number(value) * 100).toFixed(1)}%`;
}

function toneClass(color, status) {
  const normalizedColor = String(color || "").toLowerCase();
  if (normalizedColor) {
    return `tone-${normalizedColor}`;
  }
  const normalizedStatus = String(status || "").toLowerCase();
  if (normalizedStatus.includes("reject") || normalizedStatus.includes("low")) return "tone-red";
  if (normalizedStatus.includes("normal") || normalizedStatus.includes("high")) return "tone-green";
  if (normalizedStatus.includes("close") || normalizedStatus.includes("abnormal") || normalizedStatus.includes("pending") || normalizedStatus.includes("not_evaluated")) return "tone-blue";
  return "tone-yellow";
}

function resetResults() {
  latestResult = null;
  latestFilename = null;
  reportButton.disabled = true;
  reportStatus.textContent = "Reports are saved to your Documents folder.";
  jsonOutput.textContent = "{}";
  finalDecisionPanel.className = "decision-banner empty-state";
  finalDecisionPanel.textContent = "Upload an image to see modality validation, tissue routing, normality, subtype analysis, warnings, and raw JSON.";
  warningList.innerHTML = "";
  [modalityCard, tissueCard, normalityCard, subtypeCard].forEach((card, index) => {
    const titles = ["Step 0: Modality", "Level 1: Tissue", "Level 2: Normality", "Level 3: Subtype"];
    card.className = "mini-card empty";
    card.innerHTML = `<h3>${titles[index]}</h3><p>No result available for this stage.</p>`;
  });
  renderGradcam(null);
  renderChart(organChart, null, "Organ Probability Graph");
  renderChart(subtypeChart, null, "Subtype Probability Graph");
}

function renderWarnings(warnings) {
  if (!warnings || warnings.length === 0) {
    warningList.innerHTML = "";
    return;
  }
  warningList.innerHTML = `
    <div class="warning-list">
      ${warnings.map((warning) => `<div class="warning-item">${escapeHtml(warning)}</div>`).join("")}
    </div>
  `;
}

function populateOrganOverride(options) {
  const currentValue = organOverride.value;
  organOverride.innerHTML = '<option value="">No override</option>';
  (options || []).forEach((option) => {
    const element = document.createElement("option");
    element.value = option.label;
    element.textContent = option.label;
    organOverride.appendChild(element);
  });
  organOverride.value = currentValue;
}

function renderModelStatus(payload) {
  const status = payload.model_status;
  latestModelStatus = status;
  populateOrganOverride(status.organ_options || []);
  const organReady = Boolean(status.organ_ready);
  const subtypeReady = Boolean(status.subtype_ready);
  const organState = organReady ? "Ready" : status.organ_error ? "Load failed" : "Waiting for checkpoint";
  const subtypeState = subtypeReady
    ? "Ready"
    : status.subtype_error
      ? "Load failed"
      : status.subtype_checkpoint_exists
        ? "Checkpoint found, waiting to load"
        : "Waiting for subtype checkpoint";
  modelStatus.innerHTML = `
    <h2>Model Status</h2>
    <div class="metric"><span>Device</span><strong>${escapeHtml(status.device)}</strong></div>
    <div class="metric"><span>Organ/Tissue Model</span><span class="status-pill ${organReady ? toneClass("GREEN") : toneClass("RED")}">${escapeHtml(organState)}</span></div>
    <div class="metric"><span>Subtype Model</span><span class="status-pill ${subtypeReady ? toneClass("GREEN") : status.subtype_error ? toneClass("RED") : toneClass("BLUE")}">${escapeHtml(subtypeState)}</span></div>
    <div class="metric"><span>Organ Classes</span><strong>${escapeHtml(status.organ_class_count)}</strong></div>
    <div class="metric"><span>Subtype Classes</span><strong>${escapeHtml(status.subtype_class_count)}</strong></div>
    <div class="metric"><span>Report Folder</span><strong>${escapeHtml(status.report_output_dir || "Documents")}</strong></div>
    ${status.organ_error ? `<p class="helper-text">${escapeHtml(status.organ_error)}</p>` : ""}
    ${status.subtype_error ? `<p class="helper-text">${escapeHtml(status.subtype_error)}</p>` : ""}
  `;
}

async function fetchHealth() {
  try {
    const response = await fetch("/api/health");
    const payload = await response.json();
    renderModelStatus(payload);
  } catch (error) {
    modelStatus.innerHTML = `
      <h2>Model Status</h2>
      <p>Could not reach the local inference server.</p>
    `;
  }
}

function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Could not read the selected image."));
    reader.readAsDataURL(file);
  });
}

function stageMetric(label, value) {
  return `<div class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
}

function renderStageCard(container, title, stage, formatter) {
  if (!stage) {
    container.className = "mini-card empty";
    container.innerHTML = `<h3>${escapeHtml(title)}</h3><p>No result available for this stage.</p>`;
    return;
  }
  const tone = toneClass(stage.color, stage.status);
  container.className = `mini-card ${tone}`;
  container.innerHTML = `
    <h3>${escapeHtml(title)}</h3>
    <div class="status-line">
      <span class="status-pill ${tone}">${escapeHtml(stage.status || "N/A")}</span>
      <span class="stage-color ${tone}">${escapeHtml(stage.color || "")}</span>
    </div>
    ${formatter(stage)}
  `;
}

function renderDecisionBanner(result) {
  const tone = toneClass(result.level3?.color || result.level2?.color || result.level1?.color || result.step0?.color, result.status);
  const isNotOrganImage = result.modality?.rejection_code === "not_organ_image" || result.final_decision === "Not an organ image";
  finalDecisionPanel.className = `decision-banner ${tone}`;
  finalDecisionPanel.innerHTML = `
    <div class="decision-kicker">${escapeHtml(isNotOrganImage ? "Rejected Input" : "Final Decision")}</div>
    <div class="decision-title">${escapeHtml(result.final_decision || "No decision")}</div>
    <div class="decision-meta">
      <span class="status-pill ${tone}">${escapeHtml(result.status || "N/A")}</span>
      ${isNotOrganImage ? `<span class="status-pill ${tone}">${escapeHtml("Not an organ image")}</span>` : ""}
      <span>${escapeHtml(result.override_used ? "Override used" : "No override used")}</span>
    </div>
    <p>${escapeHtml(result.reason || "Decision-support output generated successfully.")}</p>
  `;
}

function renderChart(container, chart, fallbackTitle = "Probability Graph") {
  if (!chart || !chart.items || chart.items.length === 0) {
    container.className = "chart-card empty";
    container.innerHTML = `<h3>${escapeHtml(fallbackTitle)}</h3><p>No chart data available.</p>`;
    return;
  }
  container.className = "chart-card";
  container.innerHTML = `
    <h3>${escapeHtml(chart.title || fallbackTitle)}</h3>
    <div class="chart-list">
      ${chart.items.map((item) => `
        <div class="chart-row ${item.highlight ? "top-hit" : ""}">
          <div class="chart-label">${escapeHtml(item.label)}</div>
          <div class="chart-track">
            <div class="chart-fill" style="width: ${Math.max(Number(item.confidence) * 100, 1)}%"></div>
          </div>
          <div class="chart-value">${formatConfidence(item.confidence)}</div>
        </div>
      `).join("")}
    </div>
  `;
}

function renderGradcam(gradcam) {
  const organCam = gradcam?.organ;
  const subtypeCam = gradcam?.subtype;
  const items = [organCam, subtypeCam].filter(Boolean);
  if (items.length === 0) {
    gradcamPanel.className = "chart-card gradcam-card empty";
    gradcamPanel.innerHTML = `
      <h3>Grad-CAM Review</h3>
      <p>No Grad-CAM visual available for this prediction.</p>
    `;
    return;
  }

  gradcamPanel.className = "chart-card gradcam-card";
  gradcamPanel.innerHTML = `
    <div class="gradcam-header">
      <h3>Grad-CAM Review</h3>
      <p>Model attention heatmaps for organ routing and final outcome.</p>
    </div>
    <div class="gradcam-grid">
      ${items.map((item) => `
        <article class="gradcam-item">
          <div class="gradcam-meta">
            <h4>${escapeHtml(item.title || "Grad-CAM")}</h4>
            <p>${escapeHtml(item.label || "Model attention map")}</p>
          </div>
          <img
            class="gradcam-image"
            src="data:${escapeHtml(item.mime_type || "image/png")};base64,${item.image_base64}"
            alt="${escapeHtml(item.title || "Grad-CAM visualization")}"
          >
        </article>
      `).join("")}
    </div>
  `;
}

function renderResult(result) {
  latestResult = result;
  reportButton.disabled = false;
  jsonOutput.textContent = JSON.stringify(result, null, 2);
  renderDecisionBanner(result);
  renderWarnings(result.warnings || []);
  renderGradcam(result.gradcam);

  renderStageCard(modalityCard, "Step 0: Modality", result.modality, (stage) => `
    ${stageMetric("Type", stage.type || "N/A")}
    ${stageMetric("Confidence", formatConfidence(stage.confidence))}
    ${stageMetric("Gap", formatConfidence(stage.confidence_gap))}
    ${stageMetric("Override Allowed", toneClass(stage.color, stage.status) === "tone-red" ? "Yes but proceed with caution" : stage.override_allowed ? "Yes" : "No")}
    ${stageMetric("Reason", stage.reason || "None")}
  `);

  renderStageCard(tissueCard, "Level 1: Tissue", result.organ_prediction, (stage) => `
    ${stageMetric("Label", stage.selected_label || stage.label || "N/A")}
    ${stageMetric("Confidence", formatConfidence(stage.selected_confidence ?? stage.confidence))}
    ${stageMetric("Gap", formatConfidence(stage.confidence_gap))}
    ${stageMetric("Top 2", `${stage.top2_label || "N/A"} (${formatConfidence(stage.top2_confidence)})`)}
    ${stageMetric("Reason", stage.reason || stage.message || "None")}
  `);

  renderStageCard(normalityCard, "Level 2: Normality", result.normality, (stage) => `
    ${stageMetric("Outcome", stage.label || stage.status || "N/A")}
    ${stageMetric("Confidence", formatConfidence(stage.confidence))}
    ${stageMetric("Normal Label", stage.normal_label || "N/A")}
    ${stageMetric("Entropy", stage.entropy ?? "N/A")}
    ${stageMetric("Reason", stage.reason || "None")}
  `);

  renderStageCard(subtypeCard, "Level 3: Subtype", result.subtype_prediction, (stage) => `
    ${stageMetric("Label", stage.interpreted_label || stage.label || "N/A")}
    ${stageMetric("Confidence", formatConfidence(stage.confidence))}
    ${stageMetric("Gap", formatConfidence(stage.confidence_gap))}
    ${stageMetric("Alternatives", (stage.alternatives || []).join(", ") || "None")}
    ${stageMetric("Reason", stage.reason || stage.message || "None")}
  `);

  renderChart(organChart, result.charts?.organ, "Organ Probability Graph");
  renderChart(subtypeChart, result.charts?.subtype, "Subtype Probability Graph");

  if (result.model_status) {
    renderModelStatus({ model_status: result.model_status });
  }
}

imageInput.addEventListener("change", (event) => {
  const [file] = event.target.files;
  latestFilename = file ? file.name : null;
  if (!file) {
    imagePreview.style.display = "none";
    previewPlaceholder.style.display = "block";
    imagePreview.removeAttribute("src");
    resetResults();
    return;
  }
  const previewUrl = URL.createObjectURL(file);
  imagePreview.src = previewUrl;
  imagePreview.style.display = "block";
  previewPlaceholder.style.display = "none";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const [file] = imageInput.files;
  if (!file) {
    setRequestState("Select image", "tone-red");
    return;
  }
  latestFilename = file.name;
  submitButton.disabled = true;
  reportButton.disabled = true;
  reportStatus.textContent = "Reports are saved to your Documents folder.";
  setRequestState("Running", "tone-blue");
  finalDecisionPanel.className = "decision-banner tone-blue";
  finalDecisionPanel.innerHTML = `
    <div class="decision-kicker">Pipeline Running</div>
    <div class="decision-title">Analyzing uploaded image</div>
    <p>Step 0 validation, tissue routing, normality screening, and subtype analysis are running now.</p>
  `;
  try {
    const imageData = await readFileAsDataUrl(file);
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: file.name,
        image_data: imageData,
        manual_override: manualOverride.checked,
        organ_override: organOverride.value || null,
      }),
    });
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || "Prediction request failed.");
    }
    renderResult(payload.result);
    setRequestState("Complete", "tone-green");
  } catch (error) {
    resetResults();
    finalDecisionPanel.className = "decision-banner tone-red";
    finalDecisionPanel.innerHTML = `<div class="decision-title">${escapeHtml(error.message)}</div>`;
    setRequestState("Error", "tone-red");
  } finally {
    submitButton.disabled = false;
  }
});

reportButton.addEventListener("click", async () => {
  if (!latestResult) {
    return;
  }
  reportButton.disabled = true;
  reportStatus.textContent = "Generating report...";
  try {
    const [file] = imageInput.files;
    const imageData = file ? await readFileAsDataUrl(file) : null;
    const response = await fetch("/api/report", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: latestFilename || latestResult.input?.source || "upload",
        result: latestResult,
        image_data: imageData,
        output_dir: latestModelStatus?.report_output_dir || null,
      }),
    });
    const payload = await response.json();
    if (!response.ok || !payload.ok) {
      throw new Error(payload.error || "Report generation failed.");
    }
    reportStatus.textContent = `Report saved: ${payload.report_path}`;
  } catch (error) {
    reportStatus.textContent = error.message;
  } finally {
    reportButton.disabled = false;
  }
});

resetResults();
fetchHealth();
