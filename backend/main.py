"""FastAPI backend for the Cuffless Blood Pressure Estimation prototype.

Run:
    uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import os
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app import signal_processing as sp
from app.pipeline import get_model, run_pipeline
from app.sample_signal import generate_ppg

app = FastAPI(
    title="Cuffless BP Estimation API",
    description="PPG -> Butterworth -> PPG/vPPG/aPPG -> ResNet-50 -> MHCA -> SBP/DBP",
    version="1.0.0",
)

_FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.isdir(_FRONTEND_DIR):
    app.mount("/ui", StaticFiles(directory=_FRONTEND_DIR, html=True), name="ui")


@app.get("/")
def root():
    return RedirectResponse(url="/ui/")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    from app.model import torch_info
    info = torch_info()
    return {
        "status": "ok",
        "mode": "trained" if info.get("trained_weights") else "demo",
        "runtime": info,
    }


@app.get("/api/sample")
def sample(duration: float = 20.0, fs: float = 125.0, heart_rate: float = 72.0):
    """Return a synthetic raw PPG signal (for demos / offline testing)."""
    sig = generate_ppg(duration=duration, fs=fs, heart_rate=heart_rate)
    return {
        "source": "synthetic-demo",
        "sample_rate": fs,
        "heart_rate": heart_rate,
        "signal": [round(float(v), 4) for v in sig],
        "summary": sp.signal_summary(sig, fs),
    }


@app.post("/api/predict")
async def predict(
    file: UploadFile | None = File(default=None),
    fs: float = Form(default=125.0),
    heart_rate: float = Form(default=72.0),
    demo: bool = Form(default=False),
):
    """Run the full pipeline on an uploaded signal or the demo sample."""
    raw = None
    source = "upload"

    if file is not None:
        try:
            content = await file.read()
            text = content.decode("utf-8", errors="ignore")
            raw = sp.demux(text)
            source = f"upload:{file.filename}"
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not parse file: {exc}")

    if raw is None:
        raw = generate_ppg(duration=20.0, fs=fs, heart_rate=heart_rate)
        source = "synthetic-demo"

    if raw.size < 32:
        raise HTTPException(status_code=400, detail="Signal too short (need >= 32 samples).")

    try:
        result = run_pipeline(raw, fs=fs, source=source)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")
    return result


@app.get("/api/demo")
def demo(heart_rate: float = 72.0, fs: float = 125.0):
    """Convenience: run the full pipeline on the synthetic demo signal."""
    raw = generate_ppg(duration=20.0, fs=fs, heart_rate=heart_rate)
    return run_pipeline(raw, fs=fs, source="synthetic-demo")


@app.on_event("startup")
def _warm_model():
    # Building ResNet-50 can take a few seconds (plus first-run weight download).
    try:
        get_model()
    except Exception:
        pass
