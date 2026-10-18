"""Signal processing utilities for the cuffless BP pipeline.

Implements the stages:
    Raw PPG -> Butterworth band-pass filter -> PPG / vPPG / aPPG

Pure numpy/scipy so it works even if torch is unavailable.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks
from scipy.ndimage import uniform_filter1d

DEFAULT_FS = 125.0  # Hz, typical wearable PPG sampling rate


def coerce_signal(raw) -> np.ndarray:
    """Convert arbitrary input into a clean 1-D float array."""
    arr = np.asarray(raw, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("Input signal is empty.")
    return arr


def bandpass_filter(x: np.ndarray, fs: float,
                    low: float = 0.5, high: float = 5.0,
                    order: int = 4) -> np.ndarray:
    """Butterworth band-pass filter (zero-phase, forward-backward)."""
    x = np.asarray(x, dtype=float)
    nyq = 0.5 * fs
    high = min(high, nyq * 0.95)
    low = max(low, 1e-3)
    if low >= high:
        return x.copy()
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    padlen = min(3 * (sos.shape[0] * 2), max(0, x.size - 1))
    if padlen < 1:
        return x.copy()
    return sosfiltfilt(sos, x, padlen=padlen)


def normalize(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Min-max normalise to [-scale, scale]."""
    x = np.asarray(x, dtype=float)
    rng = np.ptp(x)
    if rng == 0:
        return np.zeros_like(x)
    return ((x - x.min()) / rng) * (2 * scale) - scale


def derivative(x: np.ndarray, fs: float) -> np.ndarray:
    """Time derivative (scaled by sample rate)."""
    return np.gradient(np.asarray(x, dtype=float)) * fs


def compute_forms(raw: np.ndarray, fs: float = DEFAULT_FS):
    """Return (filtered PPG, vPPG, aPPG) as normalised signals."""
    filtered = bandpass_filter(coerce_signal(raw), fs)
    filtered = normalize(filtered)
    vppg = normalize(derivative(filtered, fs))
    appg = normalize(derivative(vppg, fs))
    return filtered, vppg, appg


def extract_features(ppg: np.ndarray, appg: np.ndarray, fs: float) -> dict:
    """Extract interpretable PPG morphology features used by the demo head."""
    feats = {
        "heart_rate": 0.0,
        "rr_interval": 0.0,
        "pulse_rise_time": 0.0,
        "augmentation_index": 0.0,
        "reflection_index": 0.0,
        "systolic_peak": 0.0,
        "ppv": 0.0,
    }
    try:
        # Smooth first so single-sample noise spikes do not create false peaks.
        win = max(1, int(0.05 * fs))
        sm = uniform_filter1d(ppg, size=win) if win > 1 else ppg

        # --- Heart rate via autocorrelation (robust to the dicrotic notch) ---
        x = sm - np.mean(sm)
        ac = np.correlate(x, x, mode="full")[x.size - 1:]
        if ac[0] > 0:
            ac = ac / ac[0]
        min_lag = max(1, int(0.33 * fs))
        max_lag = min(ac.size - 1, int(2.0 * fs))
        if max_lag > min_lag:
            seg = ac[min_lag:max_lag]
            lag = min_lag + int(np.argmax(seg))
            if lag > 0:
                feats["rr_interval"] = float(lag / fs)
                feats["heart_rate"] = float(60.0 * fs / lag)

        # --- Peak morphology (prominent systolic peaks only) ---
        distance = max(1, int(0.4 * fs))
        prominence = 0.4 * float(np.std(sm))
        peaks, props = find_peaks(sm, distance=distance, prominence=prominence)
        if peaks.size >= 2:
            heights = props.get("peak_heights", sm[peaks])
            feats["systolic_peak"] = float(np.mean(heights))
            feats["ppv"] = float(np.std(heights) / (np.abs(np.mean(heights)) + 1e-9))

            seg = int(0.35 * fs)
            rises = []
            for p in peaks:
                start = max(0, p - seg)
                if p > start:
                    foot = sm[start:p].min()
                    rises.append((sm[p] - foot))
            feats["pulse_rise_time"] = float(np.mean(rises)) if rises else 0.0

        # Augmentation index from the 2nd derivative (aPPG) b/a ratio
        ap, ap_props = find_peaks(appg, distance=max(1, int(0.1 * fs)))
        if ap.size >= 2:
            ah = np.abs(ap_props.get("peak_heights", appg[ap]))
            main = np.max(ah)
            secondary = np.median(ah)
            feats["augmentation_index"] = float(secondary / (main + 1e-9))
            feats["reflection_index"] = float(np.std(ah) / (np.mean(ah) + 1e-9))
    except Exception:
        pass
    return feats


def signal_summary(x: np.ndarray, fs: float, points: int = 320) -> dict:
    """Down-sample a signal for lightweight JSON transfer to the UI."""
    x = np.asarray(x, dtype=float)
    n = x.size
    if n > points:
        idx = np.linspace(0, n - 1, points).astype(int)
    else:
        idx = np.arange(n)
    step = 1.0 / fs
    return {
        "t": [round(float(i * step), 4) for i in idx],
        "v": [round(float(x[i]), 4) for i in idx],
        "fs": fs,
        "n": int(n),
        "duration": round(float(n / fs), 3),
    }


def demux(text: str) -> np.ndarray:
    """Parse CSV / TXT / whitespace separated numeric signal data."""
    import re
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not nums:
        raise ValueError("No numeric samples found in the file.")
    return np.asarray([float(n) for n in nums], dtype=float)
