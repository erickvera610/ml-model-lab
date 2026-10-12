# Recruiter Quick View

## What this project proves

This repository demonstrates an end-to-end classification workflow on real public data: data provenance, train/validation/test separation, model comparison, a small neural network, exported inference parameters, tests, and reproducible metrics.

## Best evidence to inspect first

1. `artifacts/neural-results.json` — held-out metrics and errors from the scikit-learn neural experiment.
2. `public-results.json` — logistic-regression vs random-forest comparison on real digit images.
3. `pytorch_digits.py` — PyTorch implementation with validation-based architecture selection.
4. `robustness_benchmark.py` — fixed corruption stress tests on the held-out test split.
5. `neural-demo.html` — browser demo using exported trained weights.
6. GitHub Actions — automated tests and reproducibility checks.

## Skills shown

Python, scikit-learn, PyTorch, neural networks, logistic regression, random forests, model selection, held-out evaluation, macro F1, confusion matrices, reproducibility, API/inference parity, unit testing, and CI.

## Important interpretation

The digits benchmark is intentionally small. It is useful evidence of ML engineering fundamentals and PyTorch fluency, but it is not evidence of large-scale transformer training, distributed GPU systems, or production MLOps.

## Recommended interview demo

Run:

```bash
python -m pip install -r requirements-pytorch.txt
python pytorch_digits.py
python robustness_benchmark.py
```

Then compare the clean-test metric against the corruption conditions and explain why the test set is not used for model selection.

## Screenshot/result checklist

For portfolio screenshots, capture:

- the neural browser demo with a held-out example,
- the confusion matrix / error summary from `artifacts/neural-results.json`,
- the robustness table produced by `robustness_benchmark.py`,
- the PyTorch result JSON showing selected hidden width and held-out metrics.

Do not present a screenshot as evidence unless the underlying JSON/artifact can be reproduced.
