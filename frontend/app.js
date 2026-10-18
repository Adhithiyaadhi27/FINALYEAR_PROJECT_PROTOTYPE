const $ = (id) => document.getElementById(id);

function apiBase() { return $("#apiBase").value.replace(/\/$/, ""); }

function fmt(v) { return (v === null || v === undefined) ? "--" : v; }

async function checkHealth() {
  const badge = $("#statusBadge");
  badge.textContent = "Connecting...";
  try {
    const r = await fetch(apiBase() + "/api/health");
    const j = await r.json();
    badge.textContent = `Connected - ${j.mode.toUpperCase()} mode`;
    badge.style.borderColor = "#5fe08f";
    $("#runInfo").textContent = `Backend ready on ${apiBase()} (${j.runtime.device}, torch=${j.runtime.torch_available})`;
    return j;
  } catch (e) {
    badge.textContent = "Connection failed";
    badge.style.borderColor = "#ff6b6b";
    $("#runInfo").textContent = "Could not reach backend. Start uvicorn on port 8000.";
    return null;
  }
}

function renderStages(stages) {
  const ul = $("#stages");
  ul.innerHTML = "";
  stages.forEach((s) => {
    const li = document.createElement("li");
    li.innerHTML = `
      <div>
        <div class="label">${s.label}</div>
        <div class="detail">${s.detail || ""}</div>
      </div>
      <span class="badge ${s.status}">${s.status}</span>
    `;
    ul.appendChild(li);
  });
}

function renderPrediction(p) {
  $("#sbp").textContent = p.sbp;
  $("#dbp").textContent = p.dbp;
  $("#map").textContent = p.map;
  $("#pp").textContent = p.pulse_pressure;
  $("#confidenceText").textContent = `Confidence: ${p.confidence}`;
  $("#categoryText").textContent = `Category: ${p.category}`;
}

function renderFeatures(features) {
  const box = $("#features");
  box.innerHTML = "";
  Object.entries(features).forEach(([k, v]) => {
    const div = document.createElement("div");
    div.className = "feature-item";
    div.innerHTML = `<div class="k">${k}</div><div class="v">${typeof v === "number" ? v.toFixed(4) : v}</div>`;
    box.appendChild(div);
  });
}

function renderImages(images) {
  const box = $("#images");
  box.innerHTML = "";
  Object.entries(images).forEach(([name, uri]) => {
    const card = document.createElement("div");
    card.className = "image-card";
    card.innerHTML = `<h4>${name.toUpperCase()}</h4><img src="${uri}" alt="${name}" />`;
    box.appendChild(card);
  });
}

function renderAttention(attn) {
  const wrap = $("#attention");
  if (!Array.isArray(attn) || !attn.length) { wrap.innerHTML = "<p class='muted'>No attention data.</p>"; return; }
  let html = "<table><thead><tr><th></th>" + attn[0].map((_, i) => `<th>form ${i}</th>`).join("") + "</tr></thead><tbody>";
  attn.forEach((row, i) => {
    html += `<tr><th>form ${i}</th>` + row.map((v) => `<td>${Number(v).toFixed(4)}</td>`).join("") + "</tr>";
  });
  html += "</tbody></table>";
  wrap.innerHTML = html;
}

function drawSignal(canvasId, summary) {
  const canvas = $(canvasId);
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  const W = rect.width, H = rect.height;
  ctx.clearRect(0, 0, W, H);
  if (!summary || !summary.v || !summary.v.length) return;

  const vals = summary.v;
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const range = (max - min) || 1;
  const pad = 8;

  ctx.strokeStyle = "rgba(91,140,255,0.9)";
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  vals.forEach((v, i) => {
    const x = pad + (i / (vals.length - 1)) * (W - pad * 2);
    const y = H - pad - ((v - min) / range) * (H - pad * 2);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.fillStyle = "rgba(91,140,255,0.08)";
  ctx.fillRect(0, 0, W, H);
}

function renderAll(result) {
  $("#modeText").textContent = `Mode: ${result.mode}`;
  $("#sourceText").textContent = `Source: ${result.source}`;
  $("#disclaimer").textContent = result.disclaimer;
  renderStages(result.stages);
  renderPrediction(result.prediction);
  renderFeatures(result.features);
  renderImages(result.images);
  renderAttention(result.attention);
  const s = result.signals;
  drawSignal("cRaw", s.raw);
  drawSignal("cFiltered", s.filtered);
  drawSignal("cVppg", s.vppg);
  drawSignal("cAppg", s.appg);
}

async function runDemo() {
  const fs = Number($("#fs").value);
  const hr = Number($("#hr").value);
  $("#runInfo").textContent = "Running demo pipeline...";
  try {
    const r = await fetch(`${apiBase()}/api/demo?fs=${fs}&heart_rate=${hr}`);
    const j = await r.json();
    renderAll(j);
    $("#runInfo").textContent = `Demo completed in ${j.duration}s (${j.n_samples} samples)`;
  } catch (e) {
    $("#runInfo").textContent = "Demo run failed: " + e.message;
  }
}

async function runPredict() {
  const fs = Number($("#fs").value);
  const hr = Number($("#hr").value);
  const fileInput = $("#file");
  $("#runInfo").textContent = "Running predict pipeline...";
  try {
    const fd = new FormData();
    fd.append("fs", fs);
    fd.append("heart_rate", hr);
    if (fileInput.files.length) {
      fd.append("file", fileInput.files[0]);
    }
    const r = await fetch(`${apiBase()}/api/predict`, { method: "POST", body: fd });
    const j = await r.json();
    if (j.detail) throw new Error(j.detail);
    renderAll(j);
    $("#runInfo").textContent = `Predict completed (${j.source})`;
  } catch (e) {
    $("#runInfo").textContent = "Predict failed: " + e.message;
  }
}

function bootstrap() {
  $("#btnHealth").addEventListener("click", checkHealth);
  $("#btnDemo").addEventListener("click", runDemo);
  $("#btnPredict").addEventListener("click", runPredict);
  checkHealth();
  window.addEventListener("resize", () => {
    // redraw last if exists (best-effort)
  });
}

bootstrap();
