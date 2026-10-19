"use strict";

/* ==================== Cuffless BP Frontend ====================
 * Chart.js waveforms, API integration, validation, results,
 * model evaluation, and animated BP Care AI Assistant.
 * ==================== */

const API_BASE = (() => {
  const u = new URL(window.location.origin);
  return u.origin + ":8000";
})();

const charts = {};
let assistantOpen = false;

// ---------- helpers ----------
const $ = (id) => document.getElementById(id);
const fmt = (v) => (v === null || v === undefined || v === "") ? "--" : v;
const money = (v) => (v === null || v === undefined) ? "--" : Number(v).toFixed(1);

function showNotice(text, kind = "info") {
  const box = $("noticeArea");
  const el = document.createElement("div");
  el.className = `status-box ${kind} alert fade-in`;
  el.textContent = text;
  box.innerHTML = "";
  box.appendChild(el);
  setTimeout(() => { el.style.opacity = "0"; setTimeout(() => box.innerHTML = "", 400); }, 7000);
}

function setModeChip(trained) {
  const chip = $("modeChip");
  const dot = $("modeChipDot");
  if (trained) {
    chip.className = "mode-chip trained";
    dot.style.background = "var(--ok)";
    chip.innerHTML = `<span style="width:8px;height:8px;border-radius:50%;background:var(--ok);display:inline-block;"></span> Trained model loaded &mdash; AI predictions active`;
  } else {
    chip.className = "mode-chip demo";
    dot.style.background = "var(--warn)";
    chip.innerHTML = `<span style="width:8px;height:8px;border-radius:50%;background:var(--warn);display:inline-block;"></span> Demo mode &mdash; trained model not loaded`;
  }
}

// ---------- Charts ----------
function initCharts() {
  const cfg = (color) => ({
    type: "line",
    data: { labels: [], datasets: [{ data: [], borderColor: color, borderWidth: 1.6, pointRadius: 0, fill: false, tension: 0.2 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: false,
      plugins: { legend: { display: false }, title: { display: false } },
      scales: {
        x: { display: false },
        y: { display: false },
      },
    },
  });
  charts.raw = new Chart($("cRaw").getContext("2d"), cfg("#0f244d"));
  charts.filtered = new Chart($("cFiltered").getContext("2d"), cfg("#c62828"));
  charts.vppg = new Chart($("cVppg").getContext("2d"), cfg("#15803d"));
  charts.appg = new Chart($("cAppg").getContext("2d"), cfg("#b45309"));
}

function drawSignal(chartKey, summary) {
  const chart = charts[chartKey];
  if (!chart || !summary || !summary.v || !summary.v.length) return;
  const n = summary.v.length;
  chart.data.labels = Array.from({ length: n }, (_, i) => i);
  chart.data.datasets[0].data = summary.v;
  chart.update();
}

// ---------- Health / mode ----------
async function checkHealth() {
  try {
    const r = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(8000) });
    const j = await r.json();
    setModeChip(j.mode === "trained");
    return j;
  } catch (e) {
    setModeChip(false);
    showNotice("Could not reach the backend. Start uvicorn on port 8000.", "error");
    return null;
  }
}

// ---------- Evaluation ----------
async function loadEvaluation() {
  try {
    const r = await fetch(`${API_BASE}/api/evaluate`, { signal: AbortSignal.timeout(15000) });
    if (!r.ok) return;
    const d = await r.json();
    $("evalPanel").style.display = "block";
    const sbp = d.sbp || {};
    const dbp = d.dbp || {};
    $("evalGrid").innerHTML = `
      <div class="metric"><div class="k">Dataset</div><div class="v" style="font-size:15px;">${fmt(d.dataset)}</div><div class="sub">${d.n_test_samples || "--"} test samples</div></div>
      <div class="metric"><div class="k">SBP MAE</div><div class="v">${money(sbp.mae)}</div><div class="sub">mmHg</div></div>
      <div class="metric"><div class="k">SBP RMSE</div><div class="v">${money(sbp.rmse)}</div><div class="sub">mmHg</div></div>
      <div class="metric"><div class="k">DBP MAE</div><div class="v">${money(dbp.mae)}</div><div class="sub">mmHg</div></div>
      <div class="metric"><div class="k">DBP RMSE</div><div class="v">${money(dbp.rmse)}</div><div class="sub">mmHg</div></div>
      <div class="metric"><div class="k">Split</div><div class="v" style="font-size:13px;">Participant-level</div><div class="sub">70/15/15, no leakage</div></div>
    `;
    $("evalTable").innerHTML = `
      <thead><tr><th>Metric</th><th>SBP</th><th>DBP</th></tr></thead>
      <tbody>
        <tr><td>MAE</td><td>${money(sbp.mae)} mmHg</td><td>${money(dbp.mae)} mmHg</td></tr>
        <tr><td>RMSE</td><td>${money(sbp.rmse)} mmHg</td><td>${money(dbp.rmse)} mmHg</td></tr>
        <tr><td>ME (mean error)</td><td>${money(sbp.me)} mmHg</td><td>${money(dbp.me)} mmHg</td></tr>
        <tr><td>Std of errors</td><td>${money(sbp.std)} mmHg</td><td>${money(dbp.std)} mmHg</td></tr>
      </tbody>
    `;
    $("evalNote").textContent = d.limitations || "";
  } catch (e) {
    // evaluation optional; silently skip
  }
}

// ---------- Upload / preview ----------
function parseManualSignal(text) {
  if (!text || !text.trim()) return null;
  const tokens = text.split(/[\s,;]+/).map(t => t.trim()).filter(Boolean);
  const nums = tokens.map(Number);
  if (nums.some(n => !Number.isFinite(n))) return null;
  return nums;
}

async function previewSignal() {
  const file = $("signalFile").files[0];
  const text = $("signalText").value.trim();
  const fs = Number($("fsInput").value);

  let samples = null;
  let source = "";

  if (file) {
    const content = await file.text();
    // client-side parse for preview: extract numerics
    const nums = content.match(/[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?/g);
    if (!nums) { showNotice("No numeric samples found in the file.", "error"); return; }
    samples = nums.map(Number);
    source = `file: ${file.name}`;
  } else if (text) {
    samples = parseManualSignal(text);
    if (!samples) { showNotice("Pasted values must be numeric signal samples.", "error"); return; }
    source = "manual input";
  } else {
    showNotice("Please upload a file or paste signal values.", "error");
    return;
  }

  // validation preview
  const v = validateSamples(samples, fs);
  if (!v.ok) {
    showNotice(v.errors.join(" "), "error");
    return;
  }
  if (v.warnings.length) {
    showNotice(v.warnings.join(" "), "warn");
  }

  $("previewArea").innerHTML = `
    <div style="background:#fafafa;border:1px solid var(--border);border-radius:10px;padding:12px;margin-top:8px;">
      <div style="font-weight:700;color:var(--navy);margin-bottom:6px;">Preview</div>
      <div class="muted" style="font-size:13px;">${source}</div>
      <div style="margin-top:6px;">Samples: <strong>${v.n_samples}</strong> &bull; Duration: <strong>${v.duration}s</strong> &bull; FS: <strong>${fs} Hz</strong></div>
    </div>
  `;
  showNotice("Signal previewed and validated successfully.", "ok");

  // draw a quick raw waveform preview using first channel chart
  charts.raw.data.labels = Array.from({length: samples.length}, (_, i) => i);
  charts.raw.data.datasets[0].data = samples;
  charts.raw.update();
}

function validateSamples(samples, fs) {
  const errors = [];
  const warnings = [];
  if (!fs || fs <= 0 || fs > 2000 || !Number.isFinite(fs)) {
    errors.push("Invalid sampling frequency.");
    return { ok: false, errors, warnings, n_samples: 0, duration: 0 };
  }
  const finite = samples.filter(n => Number.isFinite(n));
  const missing = samples.length - finite.length;
  if (finite.length === 0) {
    errors.push("No valid numeric samples.");
    return { ok: false, errors, warnings, n_samples: 0, duration: 0 };
  }
  if (missing > 0) warnings.push(`${missing} non-numeric value(s) ignored.`);
  if (finite.length < 32) {
    errors.push(`Insufficient samples (${finite.length}); need at least 32.`);
  } else if (finite.length < 250) {
    warnings.push(`Short signal (${finite.length} samples, ${(finite.length/fs).toFixed(1)}s); results may be unreliable.`);
  }
  return { ok: errors.length === 0, errors, warnings, n_samples: finite.length, duration: finite.length / fs };
}

// ---------- Run prediction ----------
async function runPrediction() {
  const file = $("signalFile").files[0];
  const text = $("signalText").value.trim();
  const fs = Number($("fsInput").value);

  // decide source
  let formData = null;
  let jsonBody = null;

  if (file) {
    formData = new FormData();
    formData.append("file", file);
    formData.append("fs", fs);
  } else if (text) {
    const samples = parseManualSignal(text);
    if (!samples || samples.length < 32) {
      showNotice("Please provide at least 32 numeric signal samples.", "error");
      return;
    }
    jsonBody = { signal: samples, fs };
  } else {
    showNotice("Please upload a file or paste signal values.", "error");
    return;
  }

  $("resultsPanel").style.display = "none";
  $("validationPanel").style.display = "none";
  showNotice("Processing signal &hellip;", "info");

  try {
    let result;
    if (formData) {
      const r = await fetch(`${API_BASE}/api/predict`, {
        method: "POST", body: formData,
        signal: AbortSignal.timeout(120000),
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({ detail: "Upload failed" }));
        throw new Error(err.detail || "Upload failed");
      }
      result = await r.json();
    } else {
      const r = await fetch(`${API_BASE}/api/predict/json`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(jsonBody),
        signal: AbortSignal.timeout(120000),
      });
      if (!r.ok) {
        const err = await r.json().catch(() => ({ detail: "Prediction failed" }));
        throw new Error(err.detail || "Prediction failed");
      }
      result = await r.json();
    }

    renderValidation(result);
    renderResult(result);
    loadEvaluation();

    // auto-open assistant after successful prediction
    openAssistant();
    assistantAddIntro(result);

  } catch (e) {
    showNotice("Prediction failed: " + e.message, "error");
    console.error(e);
  }
}

function renderValidation(result) {
  const v = result.validation || {};
  $("validationPanel").style.display = "block";
  const errors = v.errors || [];
  const warns = v.warnings || [];
  let html = "";
  if (errors.length) {
    html += `<div class="status-box error">${errors.map(e => `<div>${e}</div>`).join("")}</div>`;
  }
  if (warns.length) {
    html += `<div class="status-box warn">${warns.map(w => `<div>${w}</div>`).join("")}</div>`;
  }
  if (!errors.length && !warns.length) {
    html += `<div class="status-box ok">Input valid &mdash; ${v.n_samples || result.n_samples} samples, ${v.duration || result.duration}s at ${result.sample_rate} Hz.</div>`;
  }
  $("validationContent").innerHTML = html;
}

function renderResult(result) {
  const p = result.prediction || {};
  $("sbpBig").textContent = money(p.sbp);
  $("dbpBig").textContent = money(p.dbp);
  $("mapBig").textContent = money(p.map);
  $("ppBig").textContent = money(p.pulse_pressure);
  $("modelStatusBig").textContent = p.model_status ? p.model_status.charAt(0).toUpperCase() + p.model_status.slice(1) : (result.mode || "demo");
  const q = result.signal_quality || {};
  $("qualityBig").textContent = q.level ? q.level.charAt(0).toUpperCase() + q.level.slice(1) : "--";
  $("qualityBig").className = q.level === "good" ? "ok" : (q.level === "needs_attention" ? "warn" : "danger");
  $("samplesBig").textContent = `${result.n_samples || "--"} / ${result.duration || "--"}s`;
  $("reliabilityText").textContent = p.reliability || "";
  $("predictionNote").textContent = p.model_status === "demo"
    ? "This is a calibrated demo estimate (trained weights not loaded). Not an AI prediction."
    : "AI estimate produced by the trained ResNet-50 + MHCA regression head.";

  // mode chip
  setModeChip(result.mode === "trained");

  // signals
  drawSignal("raw", result.signals && result.signals.raw);
  drawSignal("filtered", result.signals && result.signals.filtered);
  drawSignal("vppg", result.signals && result.signals.vppg);
  drawSignal("appg", result.signals && result.signals.appg);

  // images
  $("imagesPanel").style.display = "block";
  const grid = $("imagesGrid");
  grid.innerHTML = "";
  if (result.images) {
    Object.entries(result.images).forEach(([name, uri]) => {
      const card = document.createElement("div");
      card.className = "image-card";
      card.innerHTML = `<img src="${uri}" alt="${name}" /><h4>${name.toUpperCase()}</h4>`;
      grid.appendChild(card);
    });
  }

  // stages
  $("stagesList").innerHTML = "";
  (result.stages || []).forEach(s => {
    const li = document.createElement("li");
    li.innerHTML = `<div><div class="label">${s.label}</div><div class="detail">${s.detail || ""}</div></div><span class="badge ${s.status}">${s.status}</span>`;
    $("stagesList").appendChild(li);
  });

  // quality box
  const qbox = $("qualityBox");
  if (q && q.reasons && q.reasons.length) {
    qbox.className = `status-box ${q.level === "good" ? "ok" : q.level === "needs_attention" ? "warn" : "error"}`;
    qbox.innerHTML = `<strong>Signal quality: ${q.level} (score ${q.score})</strong><br>${q.reasons.map(r => `&bull; ${r}`).join("<br>")}`;
    qbox.style.display = "block";
  } else {
    qbox.style.display = "none";
  }

  $("resultsPanel").style.display = "block";
  $("resultsPanel").scrollIntoView({ behavior: "smooth", block: "start" });
  showNotice(result.mode === "trained" ? "AI estimation complete." : "Demo estimate complete.", result.mode === "trained" ? "ok" : "warn");
}

// ---------- Demo signal ----------
async function runDemo() {
  try {
    const r = await fetch(`${API_BASE}/api/demo?fs=100&heart_rate=72`, { signal: AbortSignal.timeout(120000) });
    if (!r.ok) throw new Error("Demo failed");
    const result = await r.json();
    renderValidation(result);
    renderResult(result);
    loadEvaluation();
    openAssistant();
    assistantAddIntro(result);
  } catch (e) {
    showNotice("Demo run failed: " + e.message, "error");
  }
}

// ---------- Assistant ----------
function openAssistant() {
  const panel = $("assistantPanel");
  if (panel) panel.classList.add("open");
  assistantOpen = true;
}

function closeAssistant() {
  const panel = $("assistantPanel");
  if (panel) panel.classList.remove("open");
  assistantOpen = false;
}

function assistantSetTyping(show) {
  const el = $("assistantChat");
  // remove existing typing
  const existing = el.querySelector(".typing-msg");
  if (existing) existing.remove();
  if (show) {
    const div = document.createElement("div");
    div.className = "msg assistant typing-msg";
    div.innerHTML = `<div class="avatar"><span class="face">&#128520;</span></div><div class="typing"><span></span><span></span><span></span></div>`;
    el.appendChild(div);
    el.scrollTop = el.scrollHeight;
  }
}

function assistantAddBubble(text, who) {
  const el = $("assistantChat");
  const div = document.createElement("div");
  div.className = `msg ${who} fade-in`;
  const face = who === "assistant" ? "&#128520;" : "&#128100;";
  div.innerHTML = `<div class="avatar"><span class="face">${face}</span></div><div class="bubble ${who}">${escapeHtml(text)}</div>`;
  if (who === "user") div.style.flexDirection = "row-reverse";
  el.appendChild(div);
  el.scrollTop = el.scrollHeight;
}

function assistantAddIntro(result) {
  const p = result.prediction || {};
  const q = result.signal_quality || {};
  const mode = result.mode || "demo";
  let intro;
  if (p.sbp != null && p.dbp != null) {
    intro = `Hello! I've processed your PPG signal. Your estimated blood pressure is ${money(p.sbp)} / ${money(p.dbp)} mmHg. I'm the BP Care AI Assistant &mdash; I can explain these results, suggest general heart-healthy eating, and share food-portion and lifestyle guidance. This is an AI estimate, not a confirmed blood-pressure measurement.`;
  } else {
    intro = "Hello! I'm the BP Care AI Assistant. Upload a PPG signal to get a blood-pressure estimate, then I can explain it and suggest general healthy habits.";
  }
  assistantAddBubble(intro, "assistant");
}

function assistantSend() {
  const input = $("assistantInput");
  const text = input.value.trim();
  if (!text) return;
  assistantAddBubble(text, "user");
  input.value = "";
  assistantSetTyping(true);
  const btn = $("assistantSend");
  const orig = btn.disabled;
  btn.disabled = true;

  // gather latest context
  const p = {}; // read from DOM
  const sbpEl = $("sbpBig").textContent;
  const dbpEl = $("dbpBig").textContent;
  const qEl = $("qualityBig").textContent;
  const modeEl = $("modelStatusBig").textContent;

  fetch(`${API_BASE}/api/assistant`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message: text,
      sbp: sbpEl === "--" ? null : Number(sbpEl),
      dbp: dbpEl === "--" ? null : Number(dbpEl),
      signal_quality: qEl === "--" ? null : qEl.toLowerCase(),
      mode: modeEl === "Demo" ? "demo" : "trained",
    }),
    signal: AbortSignal.timeout(30000),
  })
    .then(async r => {
      if (!r.ok) throw new Error("Assistant error");
      return r.json();
    })
    .then(d => {
      assistantSetTyping(false);
      assistantAddBubble(d.reply || "I'm sorry, I couldn't generate a response.", "assistant");
      btn.disabled = false;
    })
    .catch(e => {
      assistantSetTyping(false);
      assistantAddBubble("I'm having trouble connecting right now. Please try again.", "assistant");
      btn.disabled = false;
      console.error(e);
    });
}

function assistantAsk(preset) {
  $("assistantInput").value = preset;
  assistantSend();
}

function escapeHtml(t) {
  return String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// ---------- init ----------
function init() {
  initCharts();
  checkHealth();
  loadEvaluation();

  $("btnPreview").addEventListener("click", previewSignal);
  $("btnRun").addEventListener("click", runPrediction);
  $("btnDemo").addEventListener("click", runDemo);

  $("assistantBtn").addEventListener("click", openAssistant);
  $("assistantClose").addEventListener("click", closeAssistant);
  $("assistantSend").addEventListener("click", assistantSend);
  $("assistantInput").addEventListener("keypress", e => {
    if (e.key === "Enter") assistantSend();
  });

  // suggested chips already wired via onclick in HTML

  window.addEventListener("resize", () => {
    Object.values(charts).forEach(c => c.resize());
  });
}

init();
