"""Evaluate the existing sklearn neural model under controlled image corruptions.

This script does not tune on corrupted test data. It trains/selects with neural_train.train,
then measures degradation on the held-out test set under fixed perturbation recipes.
"""
from pathlib import Path
import json
import hashlib
import platform
from datetime import datetime, timezone
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, f1_score
from neural_train import train

ROOT = Path(__file__).parent
CORRUPTION_SEEDS = {'gaussian_noise': 20261012, 'salt_pepper': 20261013,
                    'center_occlusion': 20261014, 'brightness_70pct': 20261015}

def corrupt(x, mode, rng):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 64 or len(x) == 0:
        raise ValueError('Expected a nonempty (n, 64) pixel array')
    if not np.isfinite(x).all() or (x < 0).any() or (x > 16).any():
        raise ValueError('Pixels must be finite and in [0, 16]')
    z = x.copy().reshape(-1, 8, 8)
    if mode == 'gaussian_noise':
        z = np.clip(z + rng.normal(0, 1.5, z.shape), 0, 16)
    elif mode == 'salt_pepper':
        mask = rng.random(z.shape)
        z[mask < .04] = 0
        z[mask > .96] = 16
    elif mode == 'center_occlusion':
        z[:, 3:5, 3:5] = 0
    elif mode == 'brightness_70pct':
        z = np.clip(z * .70, 0, 16)
    else:
        raise ValueError(mode)
    return z.reshape(-1, 64)

def paired_metrics(truth, clean_pred, pred):
    """Count paired failures; a net accuracy drop can hide repaired errors."""
    truth, clean_pred, pred = map(np.asarray, (truth, clean_pred, pred))
    if truth.ndim != 1 or not len(truth) or truth.shape != pred.shape or truth.shape != clean_pred.shape:
        raise ValueError('Labels and predictions must be equal-length nonempty vectors')
    clean_ok, changed_ok = clean_pred == truth, pred == truth
    return {
        'accuracy': float(accuracy_score(truth, pred)),
        'macro_f1': float(f1_score(truth, pred, average='macro', zero_division=0)),
        'correct_to_wrong': int(np.sum(clean_ok & ~changed_ok)),
        'wrong_to_correct': int(np.sum(~clean_ok & changed_ok)),
        'prediction_changes': int(np.sum(clean_pred != pred)),
        'per_class_recall': {str(int(c)): float(np.mean(pred[truth == c] == c))
                             for c in np.unique(truth)},
    }

def main():
    report, artifact, model, data, te = train(42)
    conditions = {'clean': data.data[te]}
    for mode, seed in CORRUPTION_SEEDS.items():
        conditions[mode] = corrupt(data.data[te], mode, np.random.default_rng(seed))
    results = {}
    truth = data.target[te]
    clean_pred = model.predict(conditions['clean'])
    predictions = {}
    for name, x in conditions.items():
        pred = model.predict(x)
        results[name] = paired_metrics(truth, clean_pred, pred)
        results[name]['input_sha256'] = hashlib.sha256(np.asarray(x, dtype='<f8').tobytes()).hexdigest()
        predictions[name] = pred.tolist()
    clean = results['clean']['accuracy']
    for name in results:
        results[name]['accuracy_drop_vs_clean'] = float(clean - results[name]['accuracy'])
    out = {
        'schema_version': 2,
        'executed_at_utc': datetime.now(timezone.utc).isoformat(),
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__},
        'training_seed': 42,
        'corruption_seeds': CORRUPTION_SEEDS,
        'recipes': {'gaussian_noise': 'Add N(0, 1.5^2), clip to [0,16]',
                    'salt_pepper': 'Uniform mask <0.04 -> 0, >0.96 ->16',
                    'center_occlusion': 'Set rows 3:5, columns 3:5 to zero',
                    'brightness_70pct': 'Multiply every pixel by 0.70'},
        'dataset_sha256': report['dataset_sha256'],
        'model_sha256': hashlib.sha256(json.dumps(artifact, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'split_ids': report['split_ids'],
        'selected_hidden_width': report['selected_hidden_width'],
        'validation_macro_f1': report['validation_macro_f1'],
        'protocol': 'Fixed held-out test IDs from seed-42 training run; no corruption-specific tuning.',
        'test_size': int(len(te)),
        'conditions': results,
        'paired_predictions': {'test_ids': te.tolist(), 'truth': truth.tolist(), 'predictions': predictions},
        'version_note': 'v2 uses one seed per condition, so reordering conditions cannot change the random inputs. Do not compare stochastic inputs with v1 as if they were identical.',
        'note': 'Robustness stress test only; corruptions are synthetic and do not represent a deployment distribution.'
    }
    folder = ROOT/'artifacts'; folder.mkdir(exist_ok=True)
    (folder/'robustness-results.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == '__main__': main()
