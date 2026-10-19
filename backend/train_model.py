"""Train the ResNet-50 + Multi-Head Cross-Attention BP regression model.

Pipeline:
  1. Load the public PPG-based-BP-assessment dataset (backend/data/ppg_bp_dataset).
  2. Segment each 2-min subject recording into non-overlapping 10 s windows
     (1000 samples at fs=100 Hz) labelled with that subject's SBP/DBP.
  3. Participant-level split (train/val/test) from splits.json -> no leakage.
  4. Preprocess every window: band-pass filter -> PPG/vPPG/aPPG -> STFT
     spectrogram images -> stack to (3, 3, 224, 224).
  5. Freeze the ImageNet ResNet-50 backbone; precompute its 2048-d features.
  6. Train only the projection (2048->256), MHCA cross-attention and the
     regression head (256->64->2) with MSE loss on standardised SBP/DBP.
  7. Save full state_dict to backend/models/resnet_mhca_bp.pth plus
     training_config.json (target stats, hyperparameters).
  8. Evaluate on held-out test split: MAE, RMSE, ME, std of errors per SBP and
     DBP -> backend/models/evaluation_results.json.

Run:
    cd backend
    python train_model.py
"""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app import signal_processing as sp
from app import image_generation as ig
from app.model import BPModel, D_MODEL, NUM_HEADS, WEIGHTS_PATH, CONFIG_PATH

# ---------------------------------------------------------------------------
# Paths / hyperparameters
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent          # project root
DATA_DIR = ROOT / "backend" / "data" / "ppg_bp_dataset"
MODELS_DIR = ROOT / "backend" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FS = 100.0                      # dataset sampling frequency (Hz)
WINDOW_SECONDS = 10.0
WINDOW_SAMPLES = int(WINDOW_SECONDS * FS)   # 1000
STRIDE_SAMPLES = WINDOW_SAMPLES             # non-overlapping

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_subject_metadata() -> list[dict]:
    """Return list of subject dicts from metadata.csv."""
    meta_path = DATA_DIR / "metadata.csv"
    subjects = []
    with open(meta_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["sbp"] = float(row["sbp"])
            row["dbp"] = float(row["dbp"])
            row["hr"] = float(row["hr"])
            row["n_samples"] = int(row["n_samples"])
            subjects.append(row)
    return subjects


def load_splits() -> dict:
    with open(DATA_DIR / "splits.json", encoding="utf-8") as f:
        return json.load(f)


def load_subject_signal(subject_id: str) -> np.ndarray:
    """Load raw PPG signal samples for a subject from its CSV.

    The CSV is a single row:
      Record#, Age, Gender, Diagnosed, Treatment, SystolicBP, DiastolicBP,
      HR, JNC, AHA, <signal samples...>
    """
    fname = None
    num = subject_id.split("_")[-1]          # "01"
    for f in (DATA_DIR / "raw").glob(f"*wInfo_{num}.csv"):
        fname = f
        break
    if fname is None:
        raise FileNotFoundError(f"Raw CSV not found for {subject_id}")
    with open(fname, encoding="utf-8") as f:
        row = next(csv.reader(f))
    samples = [float(v) for v in row[10:]]
    return sp.coerce_signal(np.asarray(samples, dtype=float))


def make_windows(signal: np.ndarray) -> list[np.ndarray]:
    """Split a 1D signal into non-overlapping windows of WINDOW_SAMPLES."""
    windows = []
    n = signal.size
    start = 0
    while start + WINDOW_SAMPLES <= n:
        windows.append(signal[start:start + WINDOW_SAMPLES].copy())
        start += STRIDE_SAMPLES
    return windows


# ---------------------------------------------------------------------------
# Preprocessing -> image stack
# ---------------------------------------------------------------------------
def window_to_stack(signal_window: np.ndarray, fs: float) -> np.ndarray:
    """Preprocess one window and return the ResNet input stack (3,3,224,224)."""
    filtered = sp.normalize(sp.bandpass_filter(signal_window, fs))
    vppg = sp.normalize(sp.derivative(filtered, fs))
    appg = sp.normalize(sp.derivative(vppg, fs))
    forms = {"ppg": filtered, "vppg": vppg, "appg": appg}
    images = ig.build_feature_images(forms, fs)
    return ig.stack_form_images(images)   # (3, 3, H, W)


# ---------------------------------------------------------------------------
# Build feature cache (frozen backbone forward once per window)
# ---------------------------------------------------------------------------
@torch.no_grad()
def precompute_backbone_features(stacks: np.ndarray, sbps: np.ndarray,
                                 dbps: np.ndarray, subject_ids: list[str],
                                 backbone: nn.Module) -> dict:
    """Run frozen ResNet-50 stem + proj input over all windows.

    Returns dict with X (N, 3, D_MODEL) per-form projected tokens ready for
    MHCA, plus y (N,2) standardised targets and subject ids.
    """
    model = BPModel()
    model.ready = True
    net = model.net
    net.stem.eval()
    net.proj.eval()
    device = torch.device(DEVICE)

    X_list, y_list, sid_list = [], [], []
    batch = []
    meta_batch = []
    for i in range(stacks.shape[0]):
        batch.append(stacks[i])
        meta_batch.append((sbps[i], dbps[i], subject_ids[i]))
        if len(batch) == BATCH_SIZE:
            _flush(batch, meta_batch, net, device, X_list, y_list, sid_list)
            batch, meta_batch = [], []
    if batch:
        _flush(batch, meta_batch, net, device, X_list, y_list, sid_list)

    X = np.stack(X_list)          # (N, 3, D_MODEL)
    y = np.stack(y_list)          # (N, 2)
    return {"X": X, "y": y, "subject_ids": sid_list}


def _flush(batch, meta_batch, net, device, X_list, y_list, sid_list):
    x = torch.tensor(np.stack(batch), dtype=torch.float32).to(device)  # (B,3,3,H,W)
    b, forms, c, h, w = x.shape
    flat = x.reshape(b * forms, c, h, w)
    feat = net.stem(flat).flatten(1)           # (B*forms, 2048)
    tokens = net.proj(feat).reshape(b, forms, -1)  # (B, forms, D_MODEL)
    tokens = tokens.detach().cpu().numpy()
    for i in range(b):
        X_list.append(tokens[i])               # (3, D_MODEL)
        sbp, dbp, sid = meta_batch[i]
        y_list.append([sbp, dbp])
        sid_list.append(sid)


# ---------------------------------------------------------------------------
# Tiny head trainer (proj + MHCA + head), backbone frozen
# ---------------------------------------------------------------------------
class HeadOnly(nn.Module):
    def __init__(self, proj, mhca, head):
        super().__init__()
        self.proj = proj
        self.mhca = mhca
        self.head = head

    def forward(self, x):  # x: (B, forms, D_MODEL)
        query = x[:, 0:1, :]
        attn_out, _ = self.mhca(query, x, x,
                                need_weights=False, average_attn_weights=False)
        fused = attn_out.squeeze(1)
        return self.head(fused)


def train_head(X_train, y_train, X_val, y_val,
               sbp_mean, sbp_std, dbp_mean, dbp_std):
    """Train proj + MHCA + head with MSE on standardised targets."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    model = BPModel()
    net = model.net
    # Freeze backbone stem
    for p in net.stem.parameters():
        p.requires_grad = False
    net.stem.eval()

    head = HeadOnly(net.proj, net.mhca, net.head).to(DEVICE)
    optimizer = torch.optim.Adam(head.parameters(), lr=LR)
    criterion = nn.MSELoss()

    def standardise(y):
        y = y.copy()
        y[:, 0] = (y[:, 0] - sbp_mean) / sbp_std
        y[:, 1] = (y[:, 1] - dbp_mean) / dbp_std
        return y

    y_train_s = standardise(y_train)
    y_val_s = standardise(y_val)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train_s, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val_s, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    best_val = float("inf")
    best_state = None
    history = []
    for epoch in range(1, EPOCHS + 1):
        head.train()
        train_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = head(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss /= n

        head.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                preds = head(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)
                n += xb.size(0)
        val_loss /= n
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        print(f"  epoch {epoch:02d} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    # Restore best
    head.load_state_dict(best_state)
    return head, sbp_mean, sbp_std, dbp_mean, dbp_std, history


@torch.no_grad()
def evaluate_head(head, X, y, sbp_mean, sbp_std, dbp_mean, dbp_std,
                  subject_ids):
    """Evaluate on standardised targets; return metrics in mmHg."""
    head.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    preds = head(X_t).detach().cpu().numpy()
    # Destandardise
    preds[:, 0] = preds[:, 0] * sbp_std + sbp_mean
    preds[:, 1] = preds[:, 1] * dbp_std + dbp_mean

    err_sbp = preds[:, 0] - y[:, 0]
    err_dbp = preds[:, 1] - y[:, 1]

    def metrics(err):
        return {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "me": float(np.mean(err)),
            "std": float(np.std(err)),
        }

    return {
        "n_samples": int(len(y)),
        "subject_ids": subject_ids,
        "sbp": metrics(err_sbp),
        "dbp": metrics(err_dbp),
        "predictions": [
            {"sbp": round(float(p[0]), 2), "dbp": round(float(p[1]), 2)}
            for p in preds
        ],
        "targets": [
            {"sbp": round(float(t[0]), 2), "dbp": round(float(t[1]), 2)}
            for t in y
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    subjects = load_subject_metadata()
    splits = load_splits()
    split_of = {}
    for split_name, ids in splits.items():
        for sid in ids:
            split_of[sid] = split_name

    print("Loading signals, segmenting into 10 s windows ...")
    # Collect per split
    data = {"train": {"X": [], "y": [], "sid": []},
            "val": {"X": [], "y": [], "sid": []},
            "test": {"X": [], "y": [], "sid": []}}

    for subj in subjects:
        sid = subj["subject_id"]
        split = split_of.get(sid)
        if split is None:
            continue
        signal = load_subject_signal(sid)
        windows = make_windows(signal)
        print(f"  {sid} ({split}): {len(windows)} windows, "
              f"SBP={subj['sbp']} DBP={subj['dbp']}")
        for w in windows:
            stack = window_to_stack(w, FS)
            data[split]["X"].append(stack)
            data[split]["y"].append([subj["sbp"], subj["dbp"]])
            data[split]["sid"].append(sid)

    # Convert to arrays
    arrays = {}
    for split in ("train", "val", "test"):
        X = np.stack(data[split]["X"])          # (N,3,3,224,224)
        y = np.array(data[split]["y"], dtype=float)
        sid = data[split]["sid"]
        arrays[split] = (X, y, sid)
        print(f"{split}: X={X.shape} y={y.shape}")

    # Standardisation stats from TRAIN only
    y_train = arrays["train"][1]
    sbp_mean, sbp_std = y_train[:, 0].mean(), y_train[:, 0].std() + 1e-6
    dbp_mean, dbp_std = y_train[:, 1].mean(), y_train[:, 1].std() + 1e-6

    print("\nPrecomputing frozen ResNet-50 backbone features ...")
    model = BPModel()
    backbone_features = {}
    for split in ("train", "val", "test"):
        X, y, sid = arrays[split]
        feat = precompute_backbone_features(X, y[:, 0], y[:, 1], sid,
                                            model.net.stem)
        backbone_features[split] = feat
        print(f"  {split}: features {feat['X'].shape}")

    print("\nTraining projection + MHCA + regression head ...")
    head, sbp_mean, sbp_std, dbp_mean, dbp_std, history = train_head(
        backbone_features["train"]["X"], backbone_features["train"]["y"],
        backbone_features["val"]["X"], backbone_features["val"]["y"],
        sbp_mean, sbp_std, dbp_mean, dbp_std,
    )

    # Copy trained head weights back into the full BPModel so a single
    # state_dict (stem+proj+mhca+head) can be saved/loaded.
    model = BPModel()
    net = model.net
    net.proj.load_state_dict(head.proj.state_dict())
    net.mhca.load_state_dict(head.mhca.state_dict())
    net.head.load_state_dict(head.head.state_dict())

    torch.save(net.state_dict(), WEIGHTS_PATH)
    print(f"\nSaved model weights -> {WEIGHTS_PATH}")

    config = {
        "dataset": "sanvsquezsz/PPG-based-BP-assessment",
        "fs": FS,
        "window_seconds": WINDOW_SECONDS,
        "window_samples": WINDOW_SAMPLES,
        "samples_per_subject": 12,
        "split": {"train_subjects": len(splits["train"]),
                  "val_subjects": len(splits["val"]),
                  "test_subjects": len(splits["test"])},
        "d_model": D_MODEL,
        "num_heads": NUM_HEADS,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "device": DEVICE,
        "target_stats": {
            "sbp_mean": sbp_mean, "sbp_std": sbp_std,
            "dbp_mean": dbp_mean, "dbp_std": dbp_std,
        },
        "history": history,
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Saved training config -> {CONFIG_PATH}")

    print("\nEvaluating on held-out TEST split ...")
    test_metrics = evaluate_head(
        head,
        backbone_features["test"]["X"],
        backbone_features["test"]["y"],
        sbp_mean, sbp_std, dbp_mean, dbp_std,
        backbone_features["test"]["subject_ids"],
    )
    eval_path = MODELS_DIR / "evaluation_results.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)
    print(f"Saved evaluation -> {eval_path}")
    print(f"  SBP: MAE={test_metrics['sbp']['mae']:.2f} RMSE={test_metrics['sbp']['rmse']:.2f} "
          f"ME={test_metrics['sbp']['me']:.2f} std={test_metrics['sbp']['std']:.2f}")
    print(f"  DBP: MAE={test_metrics['dbp']['mae']:.2f} RMSE={test_metrics['dbp']['rmse']:.2f} "
          f"ME={test_metrics['dbp']['me']:.2f} std={test_metrics['dbp']['std']:.2f}")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
