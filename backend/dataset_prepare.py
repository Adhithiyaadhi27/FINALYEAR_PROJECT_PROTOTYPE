"""Dataset preparation for the cuffless BP prototype.

Downloads and verifies the public PPG-based-BP-assessment dataset
(sanvsquezsz/PPG-based-BP-assessment on GitHub).

Dataset facts (verified from the repository README):
  * 56 outpatient subjects, lab-recorded PPG
  * 2-min (120 s) recording per subject, fs = 100 Hz
  * Reference SBP and DBP in mmHg (cuff-measured)
  * Participant id: PPG_subj_##
  * Public research dataset; citation requested:
    Argüello-Prada, E.J., Castaño Mosquera, C.D. (2025).
    "Exploring supervised machine learning models to estimate blood pressure
    using non-fiducial features of the photoplethysmogram (PPG) and its
    derivatives." Phys Eng Sci Med.

CSV layout (PPG_csv_info/PPG_subj_wInfo_##.csv):
  Record#, Age, Gender, Diagnosed, Treatment, SystolicBP, DiastolicBP,
  HR, JNC, AHA, <12000 PPG signal samples>
"""

from __future__ import annotations

import csv
import json
import os
import random
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent  # project root
DATA_DIR = BASE / "backend" / "data" / "ppg_bp_dataset"
RAW_DIR = DATA_DIR / "raw"

GITHUB_RAW = (
    "https://raw.githubusercontent.com/sanvsquezsz/"
    "PPG-based-BP-assessment/main/Dataset/PPG_csv_info/"
)

FS = 100.0            # Hz, documented sampling frequency
RECORD_SECONDS = 120  # 2 minutes per recording
EXPECTED_SAMPLES = int(RECORD_SECONDS * FS)  # 12000
NUM_SUBJECTS = 56

# 70 / 15 / 15 participant-level split (no participant leakage across splits)
N_TRAIN = 39
N_VAL = 8
N_TEST = 9


def download_subject(subject_id: int) -> bytes:
    """Download one subject's info CSV from GitHub raw."""
    fname = f"PPG_subj_wInfo_{subject_id:02d}.csv"
    url = GITHUB_RAW + fname
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read()


def parse_subject_csv(text: str) -> dict:
    """Parse a subject info CSV into metadata + signal samples."""
    reader = csv.reader(text.splitlines())
    row = next(reader)
    if len(row) < 11:
        raise ValueError(f"Row too short: {len(row)} columns")

    signal = [float(v) for v in row[10:]]
    return {
        "record_id": row[0],
        "age": int(float(row[1])),
        "gender": row[2],
        "diagnosed": row[3],
        "treatment": row[4],
        "sbp": float(row[5]),
        "dbp": float(row[6]),
        "hr": float(row[7]),
        "jnc": row[8],
        "aha": row[9],
        "signal": signal,
        "n_samples": len(signal),
    }


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    metadata = []

    print(f"Downloading {NUM_SUBJECTS} subjects from GitHub ...")
    for sid in range(1, NUM_SUBJECTS + 1):
        fname = f"PPG_subj_wInfo_{sid:02d}.csv"
        dest = RAW_DIR / fname
        if not dest.exists():
            data = download_subject(sid)
            dest.write_bytes(data)
            print(f"  downloaded {fname} ({len(data)} bytes)")
        else:
            print(f"  skip (exists) {fname}")

        text = dest.read_text(encoding="utf-8", errors="ignore")
        subj = parse_subject_csv(text)
        if subj["n_samples"] != EXPECTED_SAMPLES:
            print(f"  WARNING {fname}: expected {EXPECTED_SAMPLES} samples, "
                  f"got {subj['n_samples']}")
        if not (60 <= subj["sbp"] <= 220 and 30 <= subj["dbp"] <= 140):
            print(f"  WARNING {fname}: unusual BP {subj['sbp']}/{subj['dbp']}")
        metadata.append({
            "subject_id": f"PPG_subj_{sid:02d}",
            "record_id": subj["record_id"],
            "age": subj["age"],
            "gender": subj["gender"],
            "diagnosed": subj["diagnosed"],
            "treatment": subj["treatment"],
            "sbp": subj["sbp"],
            "dbp": subj["dbp"],
            "hr": subj["hr"],
            "jnc": subj["jnc"],
            "aha": subj["aha"],
            "fs": FS,
            "n_samples": subj["n_samples"],
            "source_file": fname,
        })

    # Save metadata
    meta_path = DATA_DIR / "metadata.csv"
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metadata[0].keys()))
        writer.writeheader()
        writer.writerows(metadata)
    print(f"\nWrote {meta_path} ({len(metadata)} subjects)")

    # Participant-level split (shuffle with fixed seed for reproducibility)
    rng = random.Random(42)
    order = list(range(1, NUM_SUBJECTS + 1))
    rng.shuffle(order)
    splits = {
        "train": [f"PPG_subj_{s:02d}" for s in order[:N_TRAIN]],
        "val": [f"PPG_subj_{s:02d}" for s in order[N_TRAIN:N_TRAIN + N_VAL]],
        "test": [f"PPG_subj_{s:02d}" for s in order[N_TRAIN + N_VAL:]],
    }
    splits_path = DATA_DIR / "splits.json"
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    print(f"Wrote {splits_path}")
    print(f"  train={len(splits['train'])} val={len(splits['val'])} "
          f"test={len(splits['test'])} subjects")

    # Summary stats
    sbps = [m["sbp"] for m in metadata]
    dbps = [m["dbp"] for m in metadata]
    print(f"\nSBP: min={min(sbps):.0f} max={max(sbps):.0f} "
          f"mean={sum(sbps)/len(sbps):.1f}")
    print(f"DBP: min={min(dbps):.0f} max={max(dbps):.0f} "
          f"mean={sum(dbps)/len(dbps):.1f}")


if __name__ == "__main__":
    main()
