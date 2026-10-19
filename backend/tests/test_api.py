"""Automated tests for the cuffless BP backend API.

Run:
    cd backend
    python -m pytest tests/ -q
(or: python tests/test_api.py)
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8000"

passed = 0
failed = 0


def req(method, path, data=None, headers=None, files=None, timeout=60):
    url = BASE + path
    if files:
        import requests  # optional; not used here
    body = None
    if data is not None and not files:
        body = json.dumps(data).encode()
        h = {"Content-Type": "application/json"}
        if headers: h.update(headers)
    else:
        h = headers or {}
    r = urllib.request.Request(url, data=body, headers=h, method=method)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {"detail": str(e)}


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"PASS: {name}")
    else:
        failed += 1
        print(f"FAIL: {name} {detail}")


def main():
    # 0. health
    st, j = req("GET", "/api/health")
    check("health ok", st == 200, str(j))
    check("trained mode", j.get("mode") == "trained", str(j))

    # 1. Valid JSON signal (manual input) - successful prediction
    sig = [0.5 + 0.1 * __import__("math").sin(i) for i in range(500)]
    st, j = req("POST", "/api/predict/json", {"signal": sig, "fs": 125.0})
    check("valid manual signal prediction", st == 200, f"{st} {j}")
    if st == 200:
        check("prediction has sbp/dbp", "sbp" in j.get("prediction", {}) and "dbp" in j.get("prediction", {}), str(j))
        check("mode present", "mode" in j, str(j))
        check("signal_quality present", "signal_quality" in j, str(j))

    # 2. Empty file / empty signal
    st, j = req("POST", "/api/predict/json", {"signal": [], "fs": 125.0})
    check("empty signal rejected", st == 400, f"{st} {j}")

    # 3. Insufficient samples
    st, j = req("POST", "/api/predict/json", {"signal": [0.5] * 10, "fs": 125.0})
    check("insufficient samples rejected", st == 400, f"{st} {j}")

    # 4. Invalid numerical values (non-finite)
    st, j = req("POST", "/api/predict/json", {"signal": [0.5, float("nan")] * 25, "fs": 125.0})
    check("nan values handled", st in (200, 400), f"{st} {j}")

    # 5. Invalid sampling frequency
    st, j = req("POST", "/api/predict/json", {"signal": [0.5] * 100, "fs": 0.0})
    check("invalid fs rejected", st == 400, f"{st} {j}")

    # 6. Missing signal -> Pydantic returns 422 (structured validation error)
    st, j = req("POST", "/api/predict/json", {}, {})
    check("missing signal rejected", st in (400, 422), f"{st} {j}")

    # 7. Assistant response generation
    st, j = req("POST", "/api/assistant", {"message": "explain my result", "sbp": 130, "dbp": 85, "mode": "trained"})
    check("assistant response", st == 200 and bool(j.get("reply")), f"{st} {j}")

    # 8. Dietary preference handling
    st, j = req("POST", "/api/dietary", {"message": "what should I eat for lunch"})
    check("dietary response", st == 200 and bool(j.get("reply")), f"{st} {j}")

    # 9. Evaluation endpoint
    st, j = req("GET", "/api/evaluate")
    check("evaluate endpoint", st == 200, f"{st} {j}")
    if st == 200:
        check("eval has sbp metrics", "sbp" in j and "mae" in j.get("sbp", {}), str(j))
        check("eval has dbp metrics", "dbp" in j and "mae" in j.get("dbp", {}), str(j))

    # 10. Full upload-to-result (manual JSON) end-to-end
    sig2 = [0.6 * __import__("math").sin(i * 0.2) + 0.1 for i in range(1000)]
    st, j = req("POST", "/api/predict/json", {"signal": sig2, "fs": 100.0})
    check("end-to-end prediction", st == 200, f"{st} {j}")
    if st == 200:
        check("has reliability", "reliability" in j.get("prediction", {}), str(j))
        check("has model_status", "model_status" in j.get("prediction", {}), str(j))

    print(f"\n{passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
