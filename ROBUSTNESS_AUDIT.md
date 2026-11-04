# Robustness audit: the same 360 images under controlled corruption

Executed 2026-11-04. AI-assisted extension to ML Model Lab; this improves the **evaluation tooling**, not the trained model's accuracy.

## What changed

The original script reported aggregate accuracy and macro F1. Version 2 adds independent random seeds for each corruption, exact recipes, dataset/model/input hashes, split IDs, software versions, per-class recall, and all paired predictions. Reordering conditions no longer changes the generated random images. Stochastic v2 inputs are not identical to v1; do not interpret that change as a model improvement.

## Measured results

Same seed-42 selected MLP and held-out IDs for every condition. No corruption-specific training or tuning.

| Condition | Accuracy | Drop (percentage points) | Correct → wrong | Wrong → correct |
|---|---:|---:|---:|---:|
| clean | 98.06% | 0.00 | 0 | 0 |
| gaussian_noise | 33.89% | 64.17 | 233 | 2 |
| salt_pepper | 60.56% | 37.50 | 136 | 1 |
| center_occlusion | 85.28% | 12.78 | 48 | 2 |
| brightness_70pct | 95.83% | 2.22 | 11 | 3 |

Averages can hide individual changes: center occlusion breaks 48 previously correct predictions but repairs 2 errors, a net loss of 46/360 = 12.78 percentage points. Example: dataset row **1562**, true digit **9**, is predicted **9** when clean and **7** after zeroing the central 2×2 pixels. The JSON preserves the row ID and both predictions so this example is inspectable.

## Reproduce

```bash
python -m pip install -r requirements-benchmark.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python robustness_benchmark.py
python -m unittest discover -v
```

See [raw report](artifacts/robustness-results.json) and [protocol source](robustness_benchmark.py). This run used Python 3.12.14, NumPy 2.3.5, scikit-learn 1.8.0. Different numerical-library versions may change training; the saved hashes identify this run. The model hash here uses sorted-key JSON and is not directly comparable to the original export's differently serialized hash.

Six added tests cover immutable inputs, output bounds, exact occlusion/brightness behavior, condition-order reproducibility, invalid inputs, and paired-count accounting. All **20 Python tests passed** locally on 2026-11-04; this statement is not a claim about remote CI.

## Limits and interpretation

The data are the public UCI 8×8 digits bundled with scikit-learn. This is one split, one trained model and one fixed realization per stochastic corruption. The model performs particularly poorly under the chosen Gaussian pixel noise. That is a useful failure to expose, not a production robustness score. Corruptions are synthetic; some can change digit readability. No new data or model training improvements are claimed. Do not optimize against these test results and then reuse them as an untouched benchmark.
