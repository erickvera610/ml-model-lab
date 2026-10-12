from __future__ import annotations
import numpy as np
import pandas as pd


def perturb(X: pd.DataFrame, mode: str, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    Z = X.copy()
    std = Z.std(axis=0).replace(0, 1)
    if mode == "gaussian_noise":
        Z += rng.normal(0, 0.10, size=Z.shape) * std.to_numpy()
    elif mode == "missing_5pct":
        mask = rng.random(Z.shape) < 0.05
        Z = Z.mask(mask).fillna(Z.median())
    elif mode == "feature_dropout_10pct":
        cols = list(rng.choice(Z.columns.to_numpy(), size=max(1, int(0.10 * Z.shape[1])), replace=False))
        for col in cols:
            Z[col] = float(Z[col].median())
    elif mode != "clean":
        raise ValueError(mode)
    return Z
