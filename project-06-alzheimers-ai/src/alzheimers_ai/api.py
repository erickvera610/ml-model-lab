from __future__ import annotations
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

MODEL_PATH = Path("artifacts/classical_model.joblib")
app = FastAPI(title="Alzheimer Digital Biomarker Research API", version="1.0")

@app.get("/health")
def health():
    return {"status": "ok", "medical_use": "research prototype only"}

@app.post("/predict")
def predict(payload: dict):
    if not MODEL_PATH.exists():
        raise HTTPException(503, "Train the model first: python train.py")
    bundle = joblib.load(MODEL_PATH)
    cols = bundle["feature_names"]
    missing = [c for c in cols if c not in payload]
    if missing:
        raise HTTPException(422, f"Missing {len(missing)} required features")
    row = pd.DataFrame([{c: payload[c] for c in cols}])
    p = float(bundle["model"].predict_proba(row)[:, 1][0])
    return {
        "research_probability": round(p, 6),
        "label_at_0_5": int(p >= 0.5),
        "warning": "Not a diagnosis or medical device. For research/education only.",
    }
