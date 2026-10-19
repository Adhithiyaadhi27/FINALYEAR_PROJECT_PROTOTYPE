"""FastAPI backend for the Cuffless Blood Pressure Estimation prototype.

Run:
    uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app import signal_processing as sp
from app.pipeline import get_model, run_pipeline
from app.sample_signal import generate_ppg
from app.model import torch_info, load_training_config

app = FastAPI(
    title="Cuffless BP Estimation API",
    description="PPG -> Butterworth -> PPG/vPPG/aPPG -> ResNet-50 -> MHCA -> SBP/DBP",
    version="1.0.0",
)

_FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.isdir(_FRONTEND_DIR):
    app.mount("/ui", StaticFiles(directory=_FRONTEND_DIR, html=True), name="ui")

_MODELS_DIR = Path(__file__).resolve().parent / "models"
_DATA_DIR = Path(__file__).resolve().parent / "data" / "ppg_bp_dataset"


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------
class PredictIn(BaseModel):
    signal: list[float]
    fs: float = 125.0


class AssistantIn(BaseModel):
    message: str
    sbp: float | None = None
    dbp: float | None = None
    signal_quality: str | None = None
    reliability: str | None = None
    mode: str | None = None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return RedirectResponse(url="/ui/")


@app.get("/api/health")
def health():
    from app.model import WEIGHTS_PATH, torch_info
    info = torch_info()
    weights_present = os.path.isfile(WEIGHTS_PATH)
    return {
        "status": "ok",
        "mode": "trained" if weights_present else "demo",
        "runtime": info,
    }


# ---------------------------------------------------------------------------
# Sample / demo
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Prediction (file upload OR manual JSON signal)
# ---------------------------------------------------------------------------
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

    # --- Manual text/JSON signal input (alternative to file) ---
    signal_json = None  # reserved for future multipart form field

    if file is not None:
        try:
            content = await file.read()
            text = content.decode("utf-8", errors="ignore")
            raw = sp.demux(text)
            source = f"upload:{file.filename}"
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not parse file: {exc}")

    if demo or raw is None:
        raw = generate_ppg(duration=20.0, fs=fs, heart_rate=heart_rate)
        source = "synthetic-demo"

    # --- Sampling frequency validation ---
    try:
        sp.validate_sampling_frequency(fs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if raw.size < 32:
        raise HTTPException(status_code=400, detail="Signal too short (need >= 32 samples).")

    try:
        result = run_pipeline(raw, fs=fs, source=source)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover - inference failure
        raise HTTPException(status_code=500, detail=f"Inference failure: {exc}")
    return result


@app.post("/api/predict/json")
async def predict_json(body: PredictIn):
    """Run the pipeline on a manually supplied signal (JSON list of samples)."""
    if not body.signal:
        raise HTTPException(status_code=400, detail="No signal samples provided.")
    try:
        sp.validate_sampling_frequency(body.fs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    raw = sp.coerce_signal(body.signal)
    if raw.size < 32:
        raise HTTPException(status_code=400, detail="Signal too short (need >= 32 samples).")
    try:
        result = run_pipeline(raw, fs=body.fs, source="manual-json")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"Inference failure: {exc}")
    return result


# ---------------------------------------------------------------------------
# Model evaluation results
# ---------------------------------------------------------------------------
@app.get("/api/evaluate")
def evaluate():
    """Return held-out test-set evaluation metrics (MAE, RMSE, ME, std)."""
    eval_path = _MODELS_DIR / "evaluation_results.json"
    config_path = _MODELS_DIR / "training_config.json"
    if not eval_path.exists():
        raise HTTPException(status_code=404, detail="No evaluation results found. Train the model first.")
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    # Strip large prediction/target arrays from the summary endpoint; keep metrics.
    summary = {
        "dataset": load_training_config().get("dataset"),
        "n_test_samples": data.get("n_samples"),
        "split_strategy": "participant-level 70/15/15 (no participant leakage)",
        "fs": load_training_config().get("fs"),
        "window_seconds": load_training_config().get("window_seconds"),
        "sbp": data.get("sbp"),
        "dbp": data.get("dbp"),
        "limitations": (
            "Offline dataset evaluation only. Not clinical validation. "
            "Single public dataset; results may not generalise to other "
            "populations or devices."
        ),
    }
    if config_path.exists():
        summary["training_config"] = {
            k: load_training_config().get(k)
            for k in ("epochs", "batch_size", "lr", "d_model", "num_heads", "device")
        }
    return summary


@app.get("/api/evaluate/full")
def evaluate_full():
    """Return the complete evaluation payload including per-sample predictions."""
    eval_path = _MODELS_DIR / "evaluation_results.json"
    if not eval_path.exists():
        raise HTTPException(status_code=404, detail="No evaluation results found. Train the model first.")
    return json.loads(eval_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# AI Health Assistant (explanation + dietary + lifestyle)
# ---------------------------------------------------------------------------
_BP_CATEGORIES = [
    ((90, 60), "Low"),
    ((120, 80), "Normal"),
    ((129, 84), "Elevated"),
    ((139, 89), "Hypertension Stage 1"),
    ((159, 99), "Hypertension Stage 2"),
    ((float("inf"), float("inf")), "Hypertensive Crisis"),
]


def _category(sbp: float, dbp: float) -> str:
    for (s_max, d_max), label in _BP_CATEGORIES:
        if sbp <= s_max and dbp <= d_max:
            return label
    return "Hypertensive Crisis"


_DIETARY_GENERAL = (
    "General heart-healthy eating: enjoy plenty of vegetables and fruits, "
    "whole grains (brown rice, whole-wheat chapati, oats), pulses and legumes "
    "(lentils, chickpeas, beans), and suitable protein sources. Use unsalted "
    "nuts in moderation. Reduce excessive sodium and highly processed foods. "
    "Indian/South Indian examples: idli with sambar, vegetable-rich meals, "
    "sundar (sundal), whole-grain foods, and suitable fruit portions. "
    "No specific food can treat or cure high blood pressure."
)

_LIFESTYLE_GENERAL = (
    "General lifestyle: balanced eating, avoid excess salt, appropriate "
    "physical activity (e.g. brisk walking most days), healthy sleep "
    "routines, stress management, avoid tobacco, limit or avoid alcohol, and "
    "seek professional medical advice when needed. No food or lifestyle "
    "change promises immediate BP reduction."
)


def _assistant_reply(message: str, ctx: dict) -> str:
    """Generate a rule-based assistant reply grounded in the actual context.

    This is a transparent, deterministic helper (not a claim of medical
    diagnosis). It reads SBP/DBP from the real prediction context when
    provided.
    """
    msg = (message or "").strip().lower()
    sbp = ctx.get("sbp")
    dbp = ctx.get("dbp")
    quality = ctx.get("signal_quality", "unknown")
    reliability = ctx.get("reliability", "")
    mode = ctx.get("mode", "demo")

    # Greeting / intro
    if any(g in msg for g in ("hello", "hi", "hey", "assistant", "who are you")):
        return (
            "Hello! I'm the BP Care AI Assistant. I can explain your PPG-based "
            "blood-pressure estimate, suggest general heart-healthy eating "
            "habits, and share food-portion and lifestyle guidance. "
            "Remember: I am an AI assistant, not a doctor, and this estimate "
            "is not a confirmed blood-pressure measurement."
        )

    # Result explanation
    if any(k in msg for k in ("explain", "result", "meaning", "mean", "my bp", "sbp", "dbp")) or ("sbp" in msg or "dbp" in msg):
        parts = []
        if sbp is not None and dbp is not None:
            cat = _category(float(sbp), float(dbp))
            parts.append(
                f"Your estimated blood pressure is {sbp:.0f}/{dbp:.0f} mmHg "
                f"(SBP/DBP), which falls in the general '{cat}' range based on "
                "common reference categories. "
            )
            parts.append(
                "This is an AI estimate from a PPG signal, not a confirmed "
                "blood-pressure measurement. A single estimate should not be "
                "used to diagnose hypertension or hypotension."
            )
            parts.append(
                "Please confirm with a validated blood-pressure cuff and "
                "discuss any concerns with a qualified healthcare professional."
            )
        else:
            parts.append(
                "I don't have a current prediction to explain. Please upload a "
                "PPG signal and run the estimation first."
            )
        if quality != "unknown":
            parts.append(f"Signal quality: {quality}.")
        if reliability:
            parts.append(reliability)
        parts.append(_LIFESTYLE_GENERAL)
        return " ".join(parts)

    # Dietary
    if any(k in msg for k in ("diet", "food", "eat", "breakfast", "lunch", "dinner", "meal", "salt", "sodium", "portion")):
        reply = _DIETARY_GENERAL + " "
        if "salt" in msg or "sodium" in msg:
            reply += ("To reduce excess salt: cook at home more often, use "
                      "herbs and spices instead of salt, limit pickles/papads/"
                      "packaged snacks, and check labels on processed foods. ")
        if any(k in msg for k in ("breakfast", "lunch", "dinner", "meal", "portion", "snack")):
            reply += (
                "Balanced-plate idea: half the plate vegetables, a quarter "
                "whole grains or a suitable carbohydrate, and a quarter "
                "protein-rich food (lentils, beans, pulses, fish, egg). "
                "South Indian examples: idli with sambar and vegetables, or "
                "a vegetable-rich meal with sundal and a portion of fruit. "
                "Portions are general guidance, not calculated from your PPG "
                "estimate."
            )
        reply += "For a therapeutic or calorie-specific diet, consult a qualified dietitian or healthcare professional."
        return reply

    # Lifestyle
    if any(k in msg for k in ("lifestyle", "exercise", "activity", "sleep", "stress", "tobacco", "alcohol", "smoking")):
        return _LIFESTYLE_GENERAL

    # Limitations / reliability
    if any(k in msg for k in ("reliable", "accurate", "limit", "trust", "confidence")):
        return (
            f"Model mode: {mode}. {reliability} This is an offline, "
            "prototype AI model evaluated on one public dataset; it is not "
            "clinically validated and may not generalise. Do not make treatment "
            "decisions from this estimate alone."
        )

    # Default / help
    return (
        "I'm here to help explain your BP estimate and share general, safe "
        "dietary and lifestyle guidance. You can ask me to explain your "
        "results, suggest meals or food portions, advise on reducing salt, or "
        "share general lifestyle tips. I cannot diagnose or recommend "
        "medication changes."
    )


@app.post("/api/assistant")
def assistant(body: AssistantIn):
    """AI assistant chat endpoint.

    Reads the real SBP/DBP/signal-quality context when provided and returns a
    safe, educational explanation. Never invents BP values.
    """
    ctx = {
        "sbp": body.sbp,
        "dbp": body.dbp,
        "signal_quality": body.signal_quality,
        "reliability": body.reliability,
        "mode": body.mode,
    }
    reply = _assistant_reply(body.message, ctx)
    return {"reply": reply, "mode": ctx["mode"]}


@app.post("/api/dietary")
def dietary(body: AssistantIn):
    """Dietary guidance endpoint (general, educational, no medical claims)."""
    return {
        "reply": _DIETARY_GENERAL,
        "disclaimer": (
            "General educational guidance only. Not a therapeutic diet. "
            "Account for any known allergies or restrictions. For medical "
            "conditions, pregnancy, or prescribed diets, consult a qualified "
            "healthcare professional or dietitian."
        ),
    }
