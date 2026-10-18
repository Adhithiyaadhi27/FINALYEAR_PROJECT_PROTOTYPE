"""End-to-end pipeline orchestration.

Raw PPG -> Butterworth -> PPG/vPPG/aPPG -> images -> ResNet-50 -> MHCA ->
SBP/DBP prediction. Returns a single JSON-serialisable payload that the React
frontend renders stage by stage.
"""
from __future__ import annotations

import numpy as np

from . import image_generation as ig
from . import signal_processing as sp
from .demo_model import predict_demo_bp
from .model import BPModel, torch_info

_MODEL = None


def get_model() -> BPModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = BPModel()
    return _MODEL


STAGE_DEFS = [
    ("ppg_upload", "PPG Upload", "Raw PPG signal received"),
    ("raw_viz", "Raw PPG Visualization", "Raw waveform prepared for display"),
    ("butterworth", "Butterworth Filtering", "Band-pass 0.5-5 Hz noise removal"),
    ("derivatives", "PPG / vPPG / aPPG Generation", "1st & 2nd derivatives derived"),
    ("image_gen", "Signal-to-Image Conversion", "Time-frequency images (224x224)"),
    ("resnet", "ResNet-50 Feature Extraction", "2048-d features per signal form"),
    ("mhca", "Multi-Head Cross Attention", "Features fused across PPG/vPPG/aPPG"),
    ("prediction", "SBP / DBP Prediction", "Regression head estimates BP"),
    ("results", "Results Display", "Final blood pressure values"),
]


def _stages(done_ids, details=None):
    details = details or {}
    out = []
    for sid, label, desc in STAGE_DEFS:
        out.append({
            "id": sid,
            "label": label,
            "description": desc,
            "status": "done" if sid in done_ids else "pending",
            "detail": details.get(sid, ""),
        })
    return out


def run_pipeline(raw, fs: float = sp.DEFAULT_FS, source: str = "upload") -> dict:
    raw = sp.coerce_signal(raw)
    if raw.size < 32:
        raise ValueError("Signal too short - need at least 32 samples.")

    overall_mode = "demo"

    # --- Stage 3: Butterworth filtering ---
    filtered = sp.bandpass_filter(raw, fs)
    filtered_norm = sp.normalize(filtered)
    # --- Stage 4: derivative forms ---
    vppg = sp.normalize(sp.derivative(filtered_norm, fs))
    appg = sp.normalize(sp.derivative(vppg, fs))

    forms = {"ppg": filtered_norm, "vppg": vppg, "appg": appg}

    # --- Stage 5: signal -> image ---
    images = ig.build_feature_images(forms, fs)
    channel_stack = ig.stack_form_images(images)

    # --- Stages 6/7: ResNet-50 + MHCA ---
    model = get_model()
    net_result = model.infer(channel_stack, forms, fs)
    info = torch_info()

    # --- Feature extraction (interpretable) ---
    features = sp.extract_features(filtered_norm, appg, fs)

    # --- Stage 8: prediction ---
    if net_result is not None and info.get("trained_weights"):
        # Trained head available -> use real network output.
        sbp, dbp = net_result["raw_output"]
        sbp = float(np.clip(sbp, 80.0, 200.0))
        dbp = float(np.clip(dbp, 40.0, 130.0))
        if dbp >= sbp - 10:
            dbp = sbp - 10.0
        map_val = dbp + (sbp - dbp) / 3.0
        prediction = {
            "sbp": round(sbp, 1),
            "dbp": round(dbp, 1),
            "map": round(map_val, 1),
            "pulse_pressure": round(sbp - dbp, 1),
            "confidence": 0.9,
            "category": "Model Output",
        }
        overall_mode = "trained"
    else:
        prediction = predict_demo_bp(features, channel_stack)

    # --- assemble signals payload ---
    signals = {
        "raw": sp.signal_summary(raw, fs),
        "filtered": sp.signal_summary(filtered_norm, fs),
        "vppg": sp.signal_summary(vppg, fs),
        "appg": sp.signal_summary(appg, fs),
    }

    feature_vectors = {name: _decimate(sig, 40) for name, sig in forms.items()}

    attention = net_result["attention"] if net_result else [[0.33] * 3] * 3

    done = ["ppg_upload", "raw_viz", "butterworth", "derivatives",
            "image_gen", "prediction", "results"]
    if net_result is not None:
        done += ["resnet", "mhca"]

    details = {
        "ppg_upload": f"{raw.size} samples @ {fs:.0f} Hz ({raw.size / fs:.1f}s) - {source}",
        "butterworth": f"order-4 band-pass, {len(filtered)} samples retained",
        "image_gen": "3 images, 224x224, stacked as 3 channels",
        "resnet": ("pretrained ImageNet backbone" if info.get("pretrained_backbone")
                   else "randomly-initialised backbone") if net_result else "unavailable (torch off)",
        "mhca": f"{info.get('device')} - attention over 3 signal-form tokens",
        "prediction": ("Trained head" if overall_mode == "trained"
                       else "DEMO MODE calibrated estimator (no trained weights)"),
    }

    return {
        "mode": overall_mode,
        "runtime": {
            "torch_available": info.get("available"),
            "pretrained_backbone": info.get("pretrained_backbone"),
            "trained_weights": info.get("trained_weights"),
            "device": info.get("device"),
            "torch_error": info.get("error"),
        },
        "source": source,
        "sample_rate": fs,
        "n_samples": int(raw.size),
        "duration": round(float(raw.size / fs), 3),
        "stages": _stages(done, details),
        "signals": signals,
        "images": {
            name: ig.to_png_data_uri(img) for name, img in images.items()
        },
        "features": features,
        "feature_vectors": feature_vectors,
        "attention": attention,
        "prediction": prediction,
        "disclaimer": (
            "DEMO MODE: trained ResNet-50 + MHCA weights not found. Prediction "
            "uses a calibrated PPG-feature estimator. The neural network still "
            "runs for real feature extraction and attention visualisation."
            if overall_mode == "demo" else
            "Prediction produced by the trained ResNet-50 + MHCA regression head."
        ),
    }


def _decimate(sig: np.ndarray, points: int) -> list:
    sig = np.asarray(sig, dtype=float)
    n = sig.size
    if n > points:
        idx = np.linspace(0, n - 1, points).astype(int)
        sig = sig[idx]
    return [round(float(v), 3) for v in sig]
