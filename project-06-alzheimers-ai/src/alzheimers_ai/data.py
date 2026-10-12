from __future__ import annotations
import numpy as np
import pandas as pd

TARGET_CANDIDATES = ("class", "Class", "target", "Target", "diagnosis", "Diagnosis")


def normalize_target(y: pd.Series) -> pd.Series:
    s = y.astype(str).str.strip().str.lower()
    positive = {"p", "ad", "alzheimers", "alzheimer", "1", "true", "patient"}
    negative = {"h", "hc", "healthy", "0", "false", "control"}
    mapped = s.map(lambda v: 1 if v in positive else (0 if v in negative else np.nan))
    if mapped.isna().any():
        unique = list(dict.fromkeys(s.tolist()))
        if len(unique) != 2:
            raise ValueError(f"Expected binary target, found: {unique}")
        mapped = s.map({unique[0]: 0, unique[1]: 1})
    return mapped.astype(int)


def clean_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in list(X.columns):
        if str(col).lower() in {"id", "subject_id", "participant_id"}:
            X = X.drop(columns=[col])
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    bad = [c for c in X.columns if X[c].isna().all()]
    if bad:
        X = X.drop(columns=bad)
    return X.astype(float)


def load_darwin() -> tuple[pd.DataFrame, pd.Series, dict]:
    """Load the CC BY 4.0 DARWIN Alzheimer handwriting dataset from UCI."""
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=732)
    X = clean_features(ds.data.features)
    targets = ds.data.targets
    if isinstance(targets, pd.DataFrame):
        y_raw = targets.iloc[:, 0]
    else:
        y_raw = targets
    y = normalize_target(y_raw)
    meta = {
        "dataset": "DARWIN",
        "uci_id": 732,
        "instances": int(len(X)),
        "features": int(X.shape[1]),
        "license": "CC BY 4.0",
        "doi": "10.24432/C55D0K",
    }
    return X, y, meta
