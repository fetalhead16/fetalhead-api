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
const notesCard = document.getElementById("notes-card");
const notesList = document.getElementById("notes-list");
const assessmentTitle = document.getElementById("assessment-title");
const assessmentText = document.getElementById("assessment-text");
const overlayPreview = document.getElementById("overlay-preview");
const maskPreview = document.getElementById("mask-preview");
const preprocessedPreview = document.getElementById("preprocessed-preview");
const registrationForm = document.getElementById("registration-form");
const registrationStatus = document.getElementById("registration-status");
const navToggle = document.getElementById("nav-toggle");
const siteNav = document.getElementById("site-nav");

const numberFormatter = new Intl.NumberFormat(undefined, {
  maximumFractionDigits: 2,
});

function setStatus(message, tone = "neutral") {
  statusBox.textContent = message;
  statusChip.className = `status-chip ${tone}`;
  statusChip.textContent = tone === "neutral" ? "Idle" : tone === "ok" ? "Ready" : tone === "warn" ? "Review" : "Issue";
}

function setRegistrationStatus(message) {
  registrationStatus.textContent = message;
}

function showElement(element) {
  element.hidden = false;
  element.classList.remove("is-hidden");
}

function hideElement(element) {
  element.hidden = true;
  element.classList.add("is-hidden");
}

function uniqueNotes(notes) {
  return [...new Set(notes.filter(Boolean))];
}

function updateSelectedFile() {
  const [file] = fileInput.files;
  if (!file) {
    hideElement(selectedFile);
    hideElement(localPreview);
    return;
  }

  showElement(selectedFile);
  selectedFile.textContent = `${file.name} - ${(file.size / 1024 / 1024).toFixed(2)} MB`;

  if (file.type.startsWith("image/")) {
    const reader = new FileReader();
    reader.onload = () => {
      localPreview.src = reader.result;
      showElement(localPreview);
    };
    reader.readAsDataURL(file);
  } else {
    hideElement(localPreview);
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
  const resolvedNotes = uniqueNotes(notes);
  if (!resolvedNotes.length) {
    notesList.innerHTML = "";
    hideElement(notesCard);
    return resolvedNotes;
  }

  notesList.innerHTML = resolvedNotes.map((note) => `<li>${note}</li>`).join("");
  showElement(notesCard);
  return resolvedNotes;
}

function getCondition(status) {
  return ["normal", "no_shape_flag"].includes(status) ? "Normal" : "Abnormal";
}

function hasReviewWarning(notes) {
  return notes.some((note) =>
    [
      "not reliable enough for medical fetal head biometry",
      "not clinically plausible for a standard fetal head biometry plane",
      "Absolute millimeter values need image calibration",
      "did not expose PixelSpacing metadata",
    ].some((marker) => note.includes(marker)),
  );
}

function renderResult(data) {
  hideElement(emptyState);
  showElement(resultsBlock);

  renderMetrics(data.measurements);
  const resolvedNotes = renderNotes([...data.assessment.notes, ...data.notes]);

  renderStackItems(calibrationList, [
    ["Source", data.calibration.source],
    ["Absolute values", data.calibration.absolute_measurements ? "Yes" : "No"],
    [
      "Pixel spacing",
      data.calibration.pixel_spacing_mm == null ? "Not available" : `${numberFormatter.format(data.calibration.pixel_spacing_mm)} mm/pixel`,
    ],
    ["Image size", `${data.image_size[0]} x ${data.image_size[1]}`],
  ]);

  renderStackItems(qualityList, [
    ["Condition", getCondition(data.assessment.status)],
    ["Confidence", numberFormatter.format(data.quality.confidence)],
    ["Fit score", numberFormatter.format(data.quality.fit_score)],
    ["Contour points", `${data.quality.contour_points}`],
    ["Center offset", `${numberFormatter.format(data.quality.center_offset_px)} px`],
  ]);

  assessmentTitle.textContent = data.assessment.summary;
  assessmentText.textContent = `Condition: ${getCondition(data.assessment.status)}.`;

  overlayPreview.src = data.previews.overlay;
  maskPreview.src = data.previews.mask;
  preprocessedPreview.src = data.previews.preprocessed;

  if (["review_recommended", "abnormal", "invalid_plane"].includes(data.assessment.status) || hasReviewWarning(resolvedNotes)) {
    setStatus("Analysis complete. Review the overlay and input plane carefully.", "warn");
  } else if (data.assessment.status === "low_confidence") {
    setStatus("Analysis complete, but contour confidence is low.", "bad");
  } else {
    setStatus("Analysis complete.", "ok");
  }

  requestAnimationFrame(() => {
    resultsBlock.scrollIntoView({ behavior: "smooth", block: "start" });
  });
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
    showElement(emptyState);
    hideElement(resultsBlock);
    setStatus(error.message, "bad");
  } finally {
    analyzeButton.disabled = false;
  }
}

async function submitRegistration(event) {
  event.preventDefault();
  const formData = new FormData(registrationForm);
  const payload = Object.fromEntries(formData.entries());

  setRegistrationStatus("Submitting registration...");

  try {
    const response = await fetch("/api/register", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.detail || "Registration failed.");
    }

    registrationForm.reset();
    setRegistrationStatus(body.message);
  } catch (error) {
    setRegistrationStatus(error.message);
  }
}

function setupRevealAnimations() {
  const sections = document.querySelectorAll(".reveal");
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
        }
      });
    },
    { threshold: 0.16 },
  );

  sections.forEach((section) => observer.observe(section));
}

function setupCounters() {
  const counters = document.querySelectorAll("[data-count]");
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) {
          return;
        }

        const target = Number(entry.target.dataset.count || 0);
        let current = 0;
        const step = Math.max(1, Math.ceil(target / 24));
        const interval = setInterval(() => {
          current += step;
          if (current >= target) {
            current = target;
            clearInterval(interval);
          }
          entry.target.textContent = `${current}`;
        }, 40);

        observer.unobserve(entry.target);
      });
    },
    { threshold: 0.45 },
  );

  counters.forEach((counter) => observer.observe(counter));
}

function setupNavigation() {
  navToggle.addEventListener("click", () => {
    siteNav.classList.toggle("is-open");
  });

  siteNav.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => siteNav.classList.remove("is-open"));
  });

  const sections = document.querySelectorAll("main section[id]");
  const navLinks = [...siteNav.querySelectorAll("a")];
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) {
          return;
        }

        navLinks.forEach((link) => {
          link.classList.toggle("is-active", link.getAttribute("href") === `#${entry.target.id}`);
        });
      });
    },
    { threshold: 0.45 },
  );

  sections.forEach((section) => observer.observe(section));
}

fileInput.addEventListener("change", updateSelectedFile);
form.addEventListener("submit", submitForm);
registrationForm.addEventListener("submit", submitRegistration);

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, () => dropzone.classList.add("is-dragging"));
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, () => dropzone.classList.remove("is-dragging"));
});

setupRevealAnimations();
setupCounters();
setupNavigation();
hideElement(selectedFile);
hideElement(localPreview);
hideElement(notesCard);
hideElement(resultsBlock);
