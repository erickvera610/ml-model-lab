"""Evaluate the existing sklearn neural model under controlled image corruptions.

This script does not tune on corrupted test data. It trains/selects with neural_train.train,
then measures degradation on the held-out test set under fixed perturbation recipes.
"""
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from neural_train import train

ROOT = Path(__file__).parent

def corrupt(x, mode, rng):
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

def main():
    report, artifact, model, data, te = train(42)
    rng = np.random.default_rng(20261012)
    conditions = {'clean': data.data[te]}
    for mode in ('gaussian_noise','salt_pepper','center_occlusion','brightness_70pct'):
        conditions[mode] = corrupt(data.data[te], mode, rng)
    results = {}
    truth = data.target[te]
    for name, x in conditions.items():
        pred = model.predict(x)
        results[name] = {
            'accuracy': float(accuracy_score(truth, pred)),
            'macro_f1': float(f1_score(truth, pred, average='macro')),
        }
    clean = results['clean']['accuracy']
    for name in results:
        results[name]['accuracy_drop_vs_clean'] = float(clean - results[name]['accuracy'])
    out = {
        'protocol': 'Fixed held-out test IDs from seed-42 training run; no corruption-specific tuning.',
        'test_size': int(len(te)),
        'conditions': results,
        'note': 'Robustness stress test only; corruptions are synthetic and do not represent a deployment distribution.'
    }
    folder = ROOT/'artifacts'; folder.mkdir(exist_ok=True)
    (folder/'robustness-results.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

if __name__ == '__main__': main()
