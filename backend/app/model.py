"""ResNet-50 + Multi-Head Cross-Attention (MHCA) model.

Architecture (matches the proposed research methodology):
    Signal representations (PPG / vPPG / aPPG spectrograms)
        -> ResNet-50 feature extractor (frozen ImageNet backbone)
        -> per-form feature tokens (D_MODEL)
        -> Multi-Head Cross-Attention: PPG-form tokens act as QUERIES that
           attend over the vPPG/aPPG-form tokens (KEYS/VALUES)
        -> fused feature vector
        -> Fully-Connected regression head -> SBP and DBP

The network is built whenever torch/torchvision are available. If trained
weights (models/resnet_mhca_bp.pth) are present they are loaded and the
pipeline reports "trained" mode; otherwise it reports "demo" mode and the
calibrated estimator in demo_model.py is used.
"""
from __future__ import annotations

import os

import numpy as np

D_MODEL = 256
NUM_HEADS = 8
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "resnet_mhca_bp.pth")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "training_config.json")

_TORCH_INFO = {"available": False, "pretrained_backbone": False, "device": "cpu",
               "trained_weights": False, "error": None}


def torch_info() -> dict:
    """Return runtime info, lazily determining torch availability if needed."""
    if not _TORCH_INFO["available"] and _TORCH_INFO["error"] is None:
        # Attempt import here so /api/health reports correctly even if no
        # BPModel instance has been constructed yet.
        torch, nn, resnet50, ResNet50_Weights = _try_import_torch()
        if torch is not None:
            _TORCH_INFO["available"] = True
            _TORCH_INFO["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return dict(_TORCH_INFO)


def _try_import_torch():
    try:
        import torch  # noqa: F401
        import torch.nn as nn
        import torchvision
        from torchvision.models import resnet50, ResNet50_Weights
        return torch, nn, resnet50, ResNet50_Weights
    except Exception as exc:  # pragma: no cover - environment dependent
        _TORCH_INFO["error"] = str(exc)
        return None, None, None, None


class BPModel:
    """Wrapper that builds the network and runs inference safely."""

    def __init__(self):
        self.ready = False
        self.net = None
        self.torch = None
        self.nn = None
        torch, nn, resnet50, ResNet50_Weights = _try_import_torch()
        if torch is None:
            return
        self.torch = torch
        self.nn = nn
        _TORCH_INFO["available"] = True
        _TORCH_INFO["device"] = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.net = self._build(nn, resnet50, ResNet50_Weights)
            self.ready = True
        except Exception as exc:  # pragma: no cover
            _TORCH_INFO["error"] = f"model build failed: {exc}"
            self.ready = False

    def _build(self, nn, resnet50, ResNet50_Weights):
        """Build ResNet-50 + MHCA + regression head."""
        weights = None
        try:
            weights = ResNet50_Weights.DEFAULT
        except Exception:
            weights = None

        class ResNetMHCA(nn.Module):
            def __init__(self):
                super().__init__()
                try:
                    backbone = resnet50(weights=weights)
                except Exception:
                    backbone = resnet50(weights=None)
                    _TORCH_INFO["pretrained_backbone"] = False
                else:
                    _TORCH_INFO["pretrained_backbone"] = weights is not None
                # Stem: everything up to the global average pool -> (B,2048,1,1)
                self.stem = nn.Sequential(*list(backbone.children())[:-1])
                self.proj = nn.Linear(2048, D_MODEL)
                # Multi-Head CROSS-Attention.
                # Query = PPG form; Keys/Values = all three forms (PPG/vPPG/aPPG).
                self.mhca = nn.MultiheadAttention(D_MODEL, NUM_HEADS,
                                                  batch_first=True)
                self.head = nn.Sequential(
                    nn.Linear(D_MODEL, 64),
                    nn.ReLU(),
                    nn.Linear(64, 2),
                )

            def forward(self, x):
                # x: (B, forms, channels, H, W)  forms = (ppg, vppg, appg)
                b, forms, c, h, w = x.shape
                flat = x.reshape(b * forms, c, h, w)
                feat = self.stem(flat).flatten(1)          # (B*forms, 2048)
                tokens = self.proj(feat).reshape(b, forms, D_MODEL)  # (B,forms,D)
                # --- Cross-attention: PPG (form 0) queries the context ---
                query = tokens[:, 0:1, :]                  # (B, 1, D)
                attn_out, attn_w = self.mhca(
                    query, tokens, tokens,
                    need_weights=True, average_attn_weights=False,
                )                                          # attn_out (B,1,D)
                fused = attn_out.squeeze(1)                # (B, D)
                out = self.head(fused)                     # (B, 2)
                # attention averaged over heads -> (B, 1, forms)
                attn = attn_w.mean(dim=1)                  # (B, 1, forms)
                return out, attn, tokens

        model = ResNetMHCA()
        model.eval()
        if os.path.isfile(WEIGHTS_PATH):
            try:
                state = self.torch.load(WEIGHTS_PATH, map_location="cpu")
                model.load_state_dict(state, strict=True)
                _TORCH_INFO["trained_weights"] = True
            except Exception as exc:
                _TORCH_INFO["error"] = f"weights load failed: {exc}"
        return model

    def infer(self, channel_stack: np.ndarray, forms: dict, fs: float):
        """Run the network. Returns plain numpy values for the pipeline."""
        if not self.ready:
            return None
        import torch
        x = torch.tensor(channel_stack, dtype=torch.float32).unsqueeze(0)  # (1,3,3,H,W)
        with torch.no_grad():
            out, attn_w, tokens = self.net(x)
        raw_out = out.squeeze(0).cpu().numpy().tolist()   # [sbp, dbp]
        attn = attn_w.squeeze(0).cpu().numpy().tolist()   # (1, forms)
        embedding = tokens.squeeze(0).cpu().numpy()
        return {
            "raw_output": raw_out,
            "attention": attn,
            "embedding_norm": float(np.linalg.norm(embedding)),
            "embedding": embedding.tolist(),
        }


def load_training_config() -> dict:
    """Return saved training config (target stats, hyperparams) or empty dict."""
    if os.path.isfile(CONFIG_PATH):
        import json
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}
