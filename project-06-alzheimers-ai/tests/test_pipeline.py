import numpy as np
import pandas as pd
from alzheimers_ai.data import normalize_target, clean_features
from alzheimers_ai.modeling import classical_candidates, evaluate
from alzheimers_ai.robustness import perturb


def test_target_normalization():
    y = normalize_target(pd.Series(['P', 'H', 'P', 'H']))
    assert y.tolist() == [1, 0, 1, 0]


def test_clean_features_removes_id_and_handles_missing():
    X = pd.DataFrame({'ID':['a','b','c'], 'x':[1, np.nan, 3], 'y':[4,5,6]})
    Z = clean_features(X)
    assert 'ID' not in Z.columns
    assert not Z.isna().any().any()


def test_classical_pipeline_predicts_probabilities():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(80, 20)), columns=[f'f{i}' for i in range(20)])
    y = pd.Series(([0, 1] * 40))
    X.loc[y == 1, 'f0'] += 1.5
    model = classical_candidates(k_features=10, seed=2)['logistic'].fit(X, y)
    p = model.predict_proba(X)[:, 1]
    assert np.all((p >= 0) & (p <= 1))
    assert evaluate(y, p)['roc_auc'] > 0.6


def test_robustness_modes_preserve_shape():
    X = pd.DataFrame(np.arange(60, dtype=float).reshape(10,6), columns=list('abcdef'))
    for mode in ['clean', 'gaussian_noise', 'missing_5pct', 'feature_dropout_10pct']:
        assert perturb(X, mode).shape == X.shape
