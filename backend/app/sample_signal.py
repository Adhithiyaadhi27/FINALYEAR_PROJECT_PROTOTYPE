"""Synthetic PPG signal generator for the DEMO sample and offline testing.

Produces a realistic-looking PPG waveform: a fundamental cardiac component with
a dicrotic notch (secondary peak) plus mild baseline wander, motion and mains
noise so the filtering stage visibly cleans it.
"""
from __future__ import annotations

import numpy as np

DEFAULT_FS = 125.0


def generate_ppg(duration: float = 20.0, fs: float = DEFAULT_FS,
                 heart_rate: float = 72.0, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1.0 / fs)
    hr = heart_rate + rng.normal(0.0, 2.0)
    rr = 60.0 / hr

    phase = (t % rr) / rr
    # Main systolic upstroke.
    main = np.exp(-((phase - 0.18) ** 2) / (2 * 0.045 ** 2))
    # Dicrotic notch (reflected wave).
    notch = 0.35 * np.exp(-((phase - 0.42) ** 2) / (2 * 0.05 ** 2))
    # Diastolic decay tail.
    tail = 0.18 * np.exp(-((phase - 0.70) ** 2) / (2 * 0.08 ** 2))
    pulse = main + notch + tail

    # Baseline wander + breathing modulation.
    wander = 0.06 * np.sin(2 * np.pi * 0.25 * t)
    resp = 0.04 * np.sin(2 * np.pi * 0.3 * t + 0.7)

    # Motion artefact and 50 Hz mains hum.
    motion = np.cumsum(rng.normal(0.0, 0.01, t.size))
    motion = motion - np.linspace(motion[0], motion[-1], t.size)
    mains = 0.02 * np.sin(2 * np.pi * 50.0 * t)
    noise = rng.normal(0.0, 0.015, t.size)

    signal = pulse + wander + resp + 0.5 * motion + mains + noise
    return (signal * 1000.0).astype(float)
