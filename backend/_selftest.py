import json, time
from app.sample_signal import generate_ppg
from app.pipeline import run_pipeline

sig = generate_ppg(duration=20.0, fs=125.0)
t0 = time.time()
res = run_pipeline(sig, fs=125.0, source="selftest")
dt = time.time() - t0

print("RUNTIME:", json.dumps(res["runtime"], indent=2))
print("MODE:", res["mode"])
print("PRED:", res["prediction"])
print("FEATURES:", {k: round(v, 3) for k, v in res["features"].items()})
print("ATTN:", res["attention"])
print("IMG LEN:", {k: len(v) for k, v in res["images"].items()})
print("STAGES:", [(s["id"], s["status"]) for s in res["stages"]])
print("SIGNAL KEYS:", list(res["signals"].keys()))
print("ELAPSED sec:", round(dt, 2))
