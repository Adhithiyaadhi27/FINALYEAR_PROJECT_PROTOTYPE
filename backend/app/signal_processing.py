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
    """Parse CSV / TXT / whitespace separated numeric signal data.

    Every numeric token in the file is treated as a signal sample. Non-numeric
    tokens (headers, labels, timestamps) are ignored so a plain numeric PPG
    file parses cleanly, but the caller is responsible for confirming that the
    extracted values actually represent PPG samples (see validate_input).
    """
    import re
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not nums:
        raise ValueError("No numeric samples found in the file.")
    return np.asarray([float(n) for n in nums], dtype=float)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
MIN_SAMPLES = 32          # absolute minimum for the pipeline to run
RECOMMENDED_SAMPLES = 250 # ~2 s at 125 Hz; below this quality is questionable


def validate_sampling_frequency(fs: float) -> None:
    """Validate the sampling frequency; raise ValueError if unusable."""
    if fs is None or not np.isfinite(fs):
        raise ValueError("Sampling frequency is required and must be a number.")
    fs = float(fs)
    if fs <= 0:
        raise ValueError("Sampling frequency must be greater than 0.")
    if fs < 10:
        raise ValueError("Sampling frequency is too low (< 10 Hz) for PPG analysis.")
    if fs > 2000:
        raise ValueError("Sampling frequency is implausibly high (> 2000 Hz).")


def validate_input(raw: np.ndarray, fs: float) -> dict:
    """Validate raw signal + sampling frequency.

    Returns a dict with:
        ok: bool
        errors: list[str]
        warnings: list[str]
        n_samples: int
        duration: float
        missing_values: int
    """
    errors = []
    warnings = []

    try:
        validate_sampling_frequency(fs)
    except ValueError as exc:
        return {"ok": False, "errors": [str(exc)], "warnings": [],
                "n_samples": 0, "duration": 0.0, "missing_values": 0}

    fs = float(fs)
    arr = np.asarray(raw, dtype=float).reshape(-1)
    total = arr.size
    finite = arr[np.isfinite(arr)]
    missing = int(total - finite.size)

    if total == 0:
        errors.append("The uploaded file contains no data.")
        return {"ok": False, "errors": errors, "warnings": warnings,
                "n_samples": 0, "duration": 0.0, "missing_values": missing}

    if missing > 0:
        warnings.append(f"{missing} non-numeric / missing sample(s) were removed.")

    if finite.size < MIN_SAMPLES:
        errors.append(
            f"Insufficient signal samples ({finite.size}); need at least "
            f"{MIN_SAMPLES} for processing."
        )
    elif finite.size < RECOMMENDED_SAMPLES:
        warnings.append(
            f"Short signal ({finite.size} samples, {finite.size/fs:.1f}s). "
            "Results may be unreliable."
        )

    duration = float(finite.size / fs)
    ok = len(errors) == 0
    return {
        "ok": ok,
        "errors": errors,
        "warnings": warnings,
        "n_samples": int(finite.size),
        "duration": round(duration, 3),
        "missing_values": missing,
    }


# ---------------------------------------------------------------------------
# Signal quality assessment
# ---------------------------------------------------------------------------
def assess_signal_quality(filtered: np.ndarray, fs: float,
                          appg: np.ndarray | None = None) -> dict:
    """Assess PPG signal quality.

    Technically justified checks:
      * amplitude (flatline / clipping detection)
      * signal-to-noise ratio (variance of detrended signal vs noise band)
      * pulse regularity (coefficient of variation of peak-to-peak intervals)
      * presence of clear systolic peaks in vPPG

    Returns:
        level: "good" | "needs_attention" | "insufficient"
        score: float (0-1)
        reasons: list[str]
    """
    reasons = []
    score = 1.0
    x = np.asarray(filtered, dtype=float)
    x = x[np.isfinite(x)]

    if x.size < MIN_SAMPLES:
        return {"level": "insufficient", "score": 0.0,
                "reasons": ["Signal too short for quality assessment."]}

    # --- 1. Amplitude / flatline check ---
    rng = float(np.ptp(x))
    if rng < 1e-4:
        return {"level": "insufficient", "score": 0.0,
                "reasons": ["Flatline signal (no detectable amplitude)."]}
    if rng < 0.02:
        score -= 0.3
        reasons.append("Very low signal amplitude (possible poor contact).")

    # --- 2. Signal-to-noise ratio estimate ---
    # Noise proxy: high-frequency content after smoothing.
    win = max(3, int(0.05 * fs))
    sm = uniform_filter1d(x, size=win) if win > 1 else x
    residual = x - sm
    signal_power = float(np.var(sm))
    noise_power = float(np.var(residual)) + 1e-9
    snr_db = 10.0 * np.log10(signal_power / noise_power)
    if snr_db < 6:
        score -= 0.3
        reasons.append("Low signal-to-noise ratio (noisy recording).")
    elif snr_db < 12:
        score -= 0.1
        reasons.append("Moderate noise level.")

    # --- 3. Pulse regularity (peak-to-peak interval CV) ---
    distance = max(1, int(0.35 * fs))
    prominence = 0.25 * float(np.std(sm))
    peaks, _ = find_peaks(sm, distance=distance, prominence=prominence)
    if peaks.size >= 3:
        intervals = np.diff(peaks) / fs
        cv = float(np.std(intervals) / (np.mean(intervals) + 1e-9))
        if cv > 0.25:
            score -= 0.2
            reasons.append("Irregular pulse intervals (motion artefact likely).")
    else:
        score -= 0.25
        reasons.append("Too few clear systolic pulses detected.")

    # --- 4. vPPG / aPPG derivative shape (if provided) ---
    if appg is not None and appg.size >= MIN_SAMPLES:
        ap = np.asarray(appg, dtype=float)
        ap = ap[np.isfinite(ap)]
        ap_peaks, _ = find_peaks(ap, distance=max(1, int(0.1 * fs)))
        if ap_peaks.size < 2:
            score -= 0.1
            reasons.append("Weak aPPG waveform morphology.")

    score = float(np.clip(score, 0.0, 1.0))
    if score >= 0.7:
        level = "good"
    elif score >= 0.4:
        level = "needs_attention"
    else:
        level = "insufficient"

    return {"level": level, "score": round(score, 3), "reasons": reasons}
