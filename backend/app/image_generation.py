"""Signal-to-image conversion.

Each signal form (PPG / vPPG / aPPG) is converted into a time-frequency
representation (STFT spectrogram). The three images are also stacked as a
3-channel tensor that feeds ResNet-50, matching the base-paper pipeline.
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image
from scipy.signal import spectrogram

IMG_SIZE = 224


def _spectrogram_image(sig: np.ndarray, fs: float, size: int = IMG_SIZE) -> Image.Image:
    sig = np.asarray(sig, dtype=float)
    n = sig.size
    nperseg = int(min(max(32, n // 12), 256, n))
    if nperseg < 8 or n < 16:
        # Fall back to a simple waveform strip when the signal is too short.
        grid = np.resize(sig, (size, size))
        norm = (grid - grid.min()) / (np.ptp(grid) + 1e-9)
        return Image.fromarray((norm * 255).astype(np.uint8), mode="L")

    noverlap = int(nperseg * 0.85)
    f, t, Sxx = spectrogram(sig, fs=fs, nperseg=nperseg, noverlap=noverlap,
                            mode="magnitude")
    S = np.log1p(Sxx)
    S = S - S.min()
    if S.max() > 0:
        S = S / S.max()
    img = (S * 255).astype(np.uint8)
    # Upscale along time axis to a square image.
    im = Image.fromarray(img, mode="L")
    return im.resize((size, size), Image.BILINEAR)


def build_feature_images(forms: dict, fs: float, size: int = IMG_SIZE):
    """Return dict of PIL images keyed by form, normalised to `size`."""
    return {name: _spectrogram_image(sig, fs, size) for name, sig in forms.items()}


def stack_form_images(images: dict, order=("ppg", "vppg", "appg")) -> np.ndarray:
    """Build the ResNet-50 input tensor: one 3-channel image per signal form.

    Returns shape (num_forms, 3, H, W). Each grayscale spectrogram is replicated
    across the 3 channels so the standard RGB ImageNet backbone can consume it.
    """
    mats = []
    for k in order:
        gray = np.asarray(images[k].convert("L"), dtype=np.float32) / 255.0
        mats.append(np.stack([gray, gray, gray], axis=0))  # (3, H, W)
    return np.stack(mats, axis=0)  # (forms, 3, H, W)


def to_png_data_uri(img: Image.Image, scale: int = 2) -> str:
    """Encode a PIL image as a data: URI, optionally upscaled for crisp UI."""
    if scale != 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.convert("L").save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"
