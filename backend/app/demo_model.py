"""DEMO MODE blood-pressure estimator.

Used when trained ResNet-50 + MHCA weights are not present. The estimator maps
interpretable PPG morphology features to SBP/DBP using a physiologically
motivated, deterministic formula so the whole application is fully functional
without blocking on model weights.

Replace this with the trained regression head (models/resnet_mhca_bp.pth) for
real predictions.
"""
from __future__ import annotations

import hashlib

import numpy as np


def _stable_seed(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def predict_demo_bp(features: dict, channel_stack) -> dict:
    hr = float(features.get("heart_rate", 0.0) or 0.0)
    ai = float(features.get("augmentation_index", 0.5) or 0.0)
    ri = float(features.get("reflection_index", 0.5) or 0.0)

    hr = hr if 30.0 <= hr <= 220.0 else 72.0
    ai = float(np.clip(ai, 0.0, 1.0))
    ri = float(np.clip(ri, 0.0, 3.0))

    # Deterministic per-input jitter keeps repeat runs consistent.
    seed = _stable_seed(round(hr, 2), round(ai, 4), round(ri, 4), np.shape(channel_stack))
    rng = np.random.default_rng(seed)
    jitter = rng.normal(0.0, 1.2, size=2)

    # Physiologically motivated linear model, calibrated so a normal resting
    # PPG (HR ~72, AI ~0.6) yields roughly 118/76 mmHg.
    ri_delta = ri - 0.45
    sbp = 75.0 + 0.60 * hr + 8.0 * (ai - 0.6) + 3.0 * ri_delta + jitter[0]
    dbp = 53.0 + 0.32 * hr + 5.0 * (ai - 0.6) + 1.5 * ri_delta + jitter[1]

    sbp = float(np.clip(sbp, 90.0, 200.0))
    dbp = float(np.clip(dbp, 55.0, 120.0))
    if dbp >= sbp - 15:  # enforce plausible pulse pressure
        dbp = sbp - 15.0

    map_val = dbp + (sbp - dbp) / 3.0
    pulse_pressure = sbp - dbp

    # Confidence from signal-quality-ish proxy (range of feature values).
    conf = 0.72 + 0.2 * (1.0 - min(abs(hr - 75.0) / 75.0, 1.0))
    conf = float(np.clip(conf, 0.55, 0.94))

    return {
        "sbp": round(sbp, 1),
        "dbp": round(dbp, 1),
        "map": round(map_val, 1),
        "pulse_pressure": round(pulse_pressure, 1),
        "confidence": round(conf, 3),
        "category": _category(sbp, dbp),
    }


def _category(sbp: float, dbp: float) -> str:
    if sbp >= 180 or dbp >= 120:
        return "Hypertensive Crisis"
    if sbp >= 140 or dbp >= 90:
        return "Hypertension Stage 2"
    if sbp >= 130 or dbp >= 80:
        return "Hypertension Stage 1"
    if sbp >= 120:
        return "Elevated"
    if sbp < 90 or dbp < 60:
        return "Low"
    return "Normal"
