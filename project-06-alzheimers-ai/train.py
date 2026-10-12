from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import torch
from alzheimers_ai.data import load_darwin
from alzheimers_ai.modeling import (
    split_data, cross_validated_model_selection, classical_candidates,
    fit_calibrated, evaluate, train_torch_mlp, mc_dropout_predict,
)
from alzheimers_ai.robustness import perturb

OUT = Path('artifacts'); OUT.mkdir(exist_ok=True)


def main(seed=42):
    X, y, meta = load_darwin()
    X_train, X_test, y_train, y_test = split_data(X, y, seed)
    k = min(80, X_train.shape[1])
    best_name, cv_scores = cross_validated_model_selection(
        X_train.reset_index(drop=True), y_train.reset_index(drop=True), seed, k
    )
    base = classical_candidates(k, seed)[best_name]
    calibrated = fit_calibrated(base, X_train, y_train)
    p = calibrated.predict_proba(X_test)[:, 1]
    classical_metrics = evaluate(y_test, p)
    joblib.dump(
        {'model': calibrated, 'feature_names': list(X.columns), 'dataset': meta},
        OUT/'classical_model.joblib'
    )

    torch_bundle, val_auc = train_torch_mlp(X_train, y_train, seed=seed)
    p_torch, uncertainty = mc_dropout_predict(torch_bundle, X_test, passes=50)
    torch_metrics = evaluate(y_test, p_torch)
    torch.save({
        'state_dict': torch_bundle.model.state_dict(),
        'mean': torch_bundle.mean,
        'std': torch_bundle.std,
        'feature_names': list(X.columns)
    }, OUT/'torch_mlp.pt')

    robust = {}
    for mode in ['clean', 'gaussian_noise', 'missing_5pct', 'feature_dropout_10pct']:
        Z = perturb(X_test, mode, seed)
        robust[mode] = evaluate(y_test, calibrated.predict_proba(Z)[:, 1])

    report = {
        'seed': seed,
        'dataset': meta,
        'split': {'train': len(X_train), 'test': len(X_test)},
        'selection': {'winner': best_name, 'cv': cv_scores},
        'classical_test': classical_metrics,
        'torch_validation_auc': val_auc,
        'torch_test': torch_metrics,
        'torch_mc_uncertainty_mean': float(np.mean(uncertainty)),
        'robustness': robust,
        'medical_disclaimer': 'Research/education prototype. Not for diagnosis or clinical decision making.'
    }
    (OUT/'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
