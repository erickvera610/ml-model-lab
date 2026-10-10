"""Validation-selected threshold, ranking metrics and calibration on holdout."""
import random, math
from core import train, sigmoid, metrics

def auc(actual, scores):
    positive = [s for y, s in zip(actual, scores) if y]
    negative = [s for y, s in zip(actual, scores) if not y]
    if not positive or not negative:
        return None
    return sum(((a > b) + 0.5 * (a == b) for a in positive for b in negative)) / (len(positive) * len(negative))

def f1(m):
    p, r = (m['precision'], m['recall'])
    return 2 * p * r / (p + r) if p + r else 0

def run_evaluation(p):
    seed = int(p.get('seed', 42))
    rng = random.Random(seed)
    rows = []
    for _ in range(500):
        x = [rng.uniform(-2, 2), rng.uniform(-2, 2)]
        y = int(rng.random() < sigmoid(2 * x[0] - 1.5 * x[1] + 0.3))
        rows.append((x, y))
    rng.shuffle(rows)
    training, validation, test = (rows[:300], rows[300:400], rows[400:])
    w, loss = train(training)
    prob = lambda row: sigmoid(sum((a * b for a, b in zip(w, [1] + row[0]))))
    vs = [prob(r) for r in validation]
    vy = [r[1] for r in validation]
    thresholds = [i / 20 for i in range(1, 20)]
    trials = [{'threshold': t, 'validation_f1': f1(metrics(vy, [int(s >= t) for s in vs]))} for t in thresholds]
    threshold = max(trials, key=lambda r: r['validation_f1'])['threshold']
    scores = [prob(r) for r in test]
    actual = [r[1] for r in test]
    m = metrics(actual, [int(s >= threshold) for s in scores])
    m['f1'] = f1(m)
    m['roc_auc'] = auc(actual, scores)
    m['brier_score'] = sum(((s - y) ** 2 for s, y in zip(scores, actual))) / len(test)
    majority = int(sum((r[1] for r in training)) >= 150)
    base = metrics(actual, [majority] * 100)
    bins = []
    for k in range(5):
        indices = [i for i, s in enumerate(scores) if k / 5 <= s < (k + 1) / 5]
        bins.append({'range': [k / 5, (k + 1) / 5], 'count': len(indices), 'mean_probability': sum((scores[i] for i in indices)) / len(indices) if indices else None, 'observed_positive_rate': sum((actual[i] for i in indices)) / len(indices) if indices else None})
    return {'metrics': m, 'baseline_accuracy': base['accuracy'], 'selected_threshold': threshold, 'threshold_selection': trials, 'train_samples': 300, 'validation_samples': 100, 'test_samples': 100, 'training_loss': loss, 'weights': w, 'calibration': bins, 'roc_curve': [{'threshold': t, 'tpr': metrics(actual, [int(s >= t) for s in scores])['recall'], 'fpr': sum((y == 0 and s >= t for y, s in zip(actual, scores))) / sum((y == 0 for y in actual))} for t in [0] + thresholds + [1]], 'note': 'Synthetic binary labels. Threshold chosen on validation F1; test held out. Calibration bins are descriptive; no real-world accuracy claim.'}
