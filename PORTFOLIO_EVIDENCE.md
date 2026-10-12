# Portfolio Evidence

This page separates measured evidence from claims. All numbers below come from reproducible project artifacts in this repository.

## Real data

- Dataset: UCI Optical Recognition of Handwritten Digits, accessed through `sklearn.datasets.load_digits`.
- Observations: 1,797 real 8×8 digit images.
- Evaluation: train/validation/test separation with the final test set excluded from model selection.
- Models: standardized logistic regression, random forest, and a small 64→64→10 ReLU/softmax neural network.

## Measured results

- Random-forest benchmark, seed 42: **96.67% test accuracy**, macro F1 **0.9666** on 360 held-out images.
- Neural experiment, seed 42: **98.06% test accuracy** on 360 held-out images.
- Neural three-seed mean accuracy: **96.48%**.
- Logistic reference in the neural experiment: **98.06%** at seed 42.

These numbers demonstrate correct experimentation and reproducibility; they do not claim that the neural model is generally superior.

## Evidence a recruiter can inspect

1. `public_benchmark.py` — public-data model comparison.
2. `neural_train.py` — neural-network training and model selection.
3. `public-results.json` — split indices, versions, confusion matrix, and benchmark metrics.
4. `artifacts/neural-results.json` — neural training report.
5. `build_demo_data.py` plus the public demo — exported parameters used outside Python.
6. GitHub Actions — automated tests and reproducibility checks.

## Engineering skills demonstrated

Python, scikit-learn, numerical optimization, train/validation/test discipline, baseline comparison, classification metrics, calibration concepts, model export, browser/Python parity testing, API design, unit tests, and CI.

## Important limitations

The dataset is small and low-resolution. Digits are much simpler than modern production computer-vision problems. The neural network is intentionally small and is not evidence of large-scale deep learning, transformers, distributed training, CUDA optimization, or production MLOps.

## Next evidence milestone

Add robustness evaluation under controlled corruption (noise, blur, occlusion) with a fixed protocol and report accuracy/F1 degradation relative to the clean held-out test set. The goal is to measure failure behavior, not to tune on the test set.
