"""ResNet-50 + Multi-Head Cross-Attention (MHCA) model.

The real network is built whenever torch/torchvision are available. If trained
weights are missing the pipeline still runs and reports a DEMO MODE prediction
(see demo_model.py); the network is then used for genuine feature extraction and
attention visualisation.
"""
from __future__ import annotations

import os

import numpy as np

D_MODEL = 256
NUM_HEADS = 8
WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "resnet_mhca_bp.pth")

_TORCH_INFO = {"available": False, "pretrained_backbone": False, "device": "cpu",
               "trained_weights": False, "error": None}


def torch_info() -> dict:
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
        torch, nn, resnet50, ResNet50_Weights = _try_import_torch()
        if torch is None:
            return
        self.torch = torch
        _TORCH_INFO["available"] = True
        _TORCH_INFO["device"] = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self.net = self._build(nn, resnet50, ResNet50_Weights)
            self.ready = True
        except Exception as exc:  # pragma: no cover
            _TORCH_INFO["error"] = f"model build failed: {exc}"
            self.ready = False

    def _build(self, nn, resnet50, ResNet50_Weights):
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
                self.stem = nn.Sequential(*list(backbone.children())[:-1])  # -> (B,2048,1,1)
                self.proj = nn.Linear(2048, D_MODEL)
                self.mhca = nn.MultiheadAttention(D_MODEL, NUM_HEADS, batch_first=True)
                self.head = nn.Sequential(
                    nn.Linear(D_MODEL, 64), nn.ReLU(), nn.Linear(64, 2)
                )

            def forward(self, x):
                # x: (B, 3forms, 3ch, H, W) -> per-form feature tokens
                b, forms, c, h, w = x.shape
                flat = x.reshape(b * forms, c, h, w)
                feat = self.stem(flat).flatten(1)             # (B*forms, 2048)
                tokens = self.proj(feat).reshape(b, forms, D_MODEL)
                attn_out, attn_w = self.mhca(tokens, tokens, tokens,
                                             need_weights=True, average_attn_weights=False)
                pooled = attn_out.mean(dim=1)
                out = self.head(pooled)
                return out, attn_w, tokens

        model = ResNetMHCA()
        model.eval()
        if os.path.isfile(WEIGHTS_PATH):
            try:
                state = self.torch.load(WEIGHTS_PATH, map_location="cpu")
                model.load_state_dict(state, strict=False)
                _TORCH_INFO["trained_weights"] = True
            except Exception as exc:
                _TORCH_INFO["error"] = f"weights load failed: {exc}"
        return model

    def infer(self, channel_stack: np.ndarray, forms: dict, fs: float):
        """Run the network. Returns (embedding, attention_matrix, raw_out).

        All values are plain numpy so callers do not depend on torch.
        """
        if not self.ready:
            return None
        import torch
        x = torch.tensor(channel_stack, dtype=torch.float32).unsqueeze(0)  # (1,3,3,H,W)
        with torch.no_grad():
            out, attn_w, tokens = self.net(x)
        raw_out = out.squeeze(0).cpu().numpy().tolist()
        # attention averaged over heads -> (forms, forms)
        attn = attn_w[0].mean(dim=0).cpu().numpy()
        embedding = tokens.squeeze(0).cpu().numpy()
        return {
            "raw_output": raw_out,
            "attention": attn.tolist(),
            "embedding_norm": float(np.linalg.norm(embedding)),
            "embedding": embedding.tolist(),
        }
