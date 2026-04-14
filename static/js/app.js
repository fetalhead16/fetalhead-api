const form = document.getElementById("analysis-form");
const fileInput = document.getElementById("image");
const pixelSpacingInput = document.getElementById("pixel_spacing_mm");
const gestationalAgeInput = document.getElementById("gestational_age_weeks");
const dropzone = document.querySelector(".dropzone");
const analyzeButton = document.getElementById("analyze-button");
const localPreview = document.getElementById("local-preview");
const selectedFile = document.getElementById("selected-file");
const statusBox = document.getElementById("status-box");
const statusChip = document.getElementById("status-chip");
const emptyState = document.getElementById("empty-state");
const resultsBlock = document.getElementById("results-block");
const metricsGrid = document.getElementById("metrics-grid");
const calibrationList = document.getElementById("calibration-list");
const qualityList = document.getElementById("quality-list");
const notesList = document.getElementById("notes-list");
const assessmentTitle = document.getElementById("assessment-title");
const assessmentText = document.getElementById("assessment-text");
const overlayPreview = document.getElementById("overlay-preview");
const maskPreview = document.getElementById("mask-preview");
const preprocessedPreview = document.getElementById("preprocessed-preview");

const numberFormatter = new Intl.NumberFormat(undefined, {
  maximumFractionDigits: 2,
});

function setStatus(message, tone = "neutral") {
  statusBox.textContent = message;
  statusChip.className = `status-chip ${tone}`;
  statusChip.textContent = tone === "neutral" ? "Idle" : tone === "ok" ? "Ready" : tone === "warn" ? "Review" : "Issue";
}

function updateSelectedFile() {
  const [file] = fileInput.files;
  if (!file) {
    selectedFile.hidden = true;
    localPreview.hidden = true;
    return;
  }

  selectedFile.hidden = false;
  selectedFile.textContent = `${file.name} • ${(file.size / 1024 / 1024).toFixed(2)} MB`;

  if (file.type.startsWith("image/")) {
    const reader = new FileReader();
    reader.onload = () => {
      localPreview.src = reader.result;
      localPreview.hidden = false;
    };
    reader.readAsDataURL(file);
  } else {
    localPreview.hidden = true;
  }
}

function renderStackItems(container, items) {
  container.innerHTML = items
    .map(
      ([label, value]) => `
        <div class="stack-item">
          <span>${label}</span>
          <strong>${value}</strong>
        </div>
      `,
    )
    .join("");
}

function renderMetrics(measurements) {
  metricsGrid.innerHTML = Object.values(measurements)
    .map(
      (metric) => `
        <article class="metric-card">
          <p class="metric-label">${metric.label}</p>
          <p class="metric-value">${numberFormatter.format(metric.value)} <span class="metric-unit">${metric.unit}</span></p>
        </article>
      `,
    )
    .join("");
}

function renderNotes(notes) {
  notesList.innerHTML = notes.map((note) => `<li>${note}</li>`).join("");
}

function renderResult(data) {
  emptyState.hidden = true;
  resultsBlock.hidden = false;

  renderMetrics(data.measurements);
  renderNotes([...data.assessment.notes, ...data.notes]);

  renderStackItems(calibrationList, [
    ["Source", data.calibration.source],
    ["Absolute values", data.calibration.absolute_measurements ? "Yes" : "No"],
    [
      "Pixel spacing",
      data.calibration.pixel_spacing_mm == null ? "Not available" : `${numberFormatter.format(data.calibration.pixel_spacing_mm)} mm/pixel`,
    ],
    ["Image size", `${data.image_size[0]} × ${data.image_size[1]}`],
  ]);

  renderStackItems(qualityList, [
    ["Confidence", numberFormatter.format(data.quality.confidence)],
    ["Fit score", numberFormatter.format(data.quality.fit_score)],
    ["Contour points", `${data.quality.contour_points}`],
    ["Center offset", `${numberFormatter.format(data.quality.center_offset_px)} px`],
  ]);

  assessmentTitle.textContent = data.assessment.summary;
  assessmentText.textContent = `Mode: ${data.assessment.classifier_mode}. Status: ${data.assessment.status}.`;

  overlayPreview.src = data.previews.overlay;
  maskPreview.src = data.previews.mask;
  preprocessedPreview.src = data.previews.preprocessed;

  if (data.assessment.status === "review_recommended" || data.assessment.status === "abnormal") {
    setStatus("Analysis complete. The output suggests the scan should be reviewed carefully.", "warn");
  } else if (data.assessment.status === "low_confidence") {
    setStatus("Analysis complete, but contour confidence is low.", "bad");
  } else {
    setStatus("Analysis complete.", "ok");
  }
}

async function submitForm(event) {
  event.preventDefault();

  if (!fileInput.files.length) {
    setStatus("Please choose an image before running the analysis.", "bad");
    return;
  }

  analyzeButton.disabled = true;
  setStatus("Processing the scan and extracting biometric values...", "neutral");

  const formData = new FormData();
  formData.append("image", fileInput.files[0]);

  if (pixelSpacingInput.value) {
    formData.append("pixel_spacing_mm", pixelSpacingInput.value);
  }
  if (gestationalAgeInput.value) {
    formData.append("gestational_age_weeks", gestationalAgeInput.value);
  }

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Analysis failed.");
    }

    renderResult(payload);
  } catch (error) {
    emptyState.hidden = false;
    resultsBlock.hidden = true;
    setStatus(error.message, "bad");
  } finally {
    analyzeButton.disabled = false;
  }
}

fileInput.addEventListener("change", updateSelectedFile);
form.addEventListener("submit", submitForm);

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, () => dropzone.classList.add("is-dragging"));
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, () => dropzone.classList.remove("is-dragging"));
});
