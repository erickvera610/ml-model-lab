# Classification Evaluation Lab

![Tests](https://github.com/erickvera610/ml-model-lab/actions/workflows/tests.yml/badge.svg)

> **Recruiter quick view:** real public data, held-out evaluation, neural-network training, PyTorch benchmark, robustness testing, browser inference, unit tests, and CI. Start with [RECRUITER_OVERVIEW.md](RECRUITER_OVERVIEW.md), [PORTFOLIO_EVIDENCE.md](PORTFOLIO_EVIDENCE.md), the [neural demo](https://erickvera610.github.io/ml-model-lab/neural-demo.html), `pytorch_digits.py`, and `robustness_benchmark.py`.

## New: trained neural network with public inference demo

**[Inspect the neural demo](https://erickvera610.github.io/ml-model-lab/neural-demo.html)** · [Reviewer evidence](RECRUITER_GUIDE.md) · [Training report](artifacts/neural-results.json) · [CI tests](https://github.com/erickvera610/ml-model-lab/actions/workflows/tests.yml)

Trained a 64 → 64 → 10 ReLU/softmax neural network on real 8 × 8 digit images. Hidden width is selected using outer validation macro F1; early stopping uses an internal training-only fraction. The test set remains outside selection.

Seed 42 accuracy: **98.06%** on 360 held-out images. Logistic reference: **98.06%**. Three-seed mean: **96.48%**. This experiment shows neural training, not evidence that a neural model is always better. The earlier random-forest report remains separately available.

The public demo loads the actual Python-exported numerical weights and shows held-out examples plus every reported seed-42 error. The local `/neural` page uses the Python inference API with the same artifact. Parameter checksums and cross-language parity tests connect report, artifact and inference.

```bash
python -m pip install -r requirements-benchmark.txt
python neural_train.py
python build_demo_data.py
python -m unittest discover -v
node test_neural.cjs
python app.py
# Open http://127.0.0.1:8000/neural
```

### PyTorch + robustness extension

The repository now also includes `pytorch_digits.py`, a separate PyTorch MLP experiment on the same public digits dataset, and `robustness_benchmark.py`, which measures held-out degradation under fixed noise, salt-and-pepper corruption, center occlusion, and brightness reduction. These extensions are deliberately separate from the original benchmark so a reviewer can compare frameworks and inspect failure behavior without rewriting the original evidence.

```bash
python -m pip install -r requirements-pytorch.txt
python pytorch_digits.py
python robustness_benchmark.py
```

Generated PyTorch and robustness metrics are written under `artifacts/`; do not report them until the scripts have been run in the current environment.

This is a small MLP with one hidden layer, not an LLM or large-scale deep-learning system. Hand-drawn inputs may differ from the measured dataset. Use [neural_train.py](neural_train.py) to reproduce the model; do not repeatedly change settings based on test results.

The original logistic-regression learning demo is preserved. The featured version evaluates classification quality, ranking and probability calibration separately.

**[Open the live demo](https://erickvera610.github.io/ml-model-lab/)** · [Spanish tutorial](TUTORIAL_ES.md) · [Case study](CASE_STUDY.md) · [Measured results](benchmark-results.json)

## Understand it in one minute

300 training observations → 100 validation observations → selected threshold → 100 test observations

Use gradient descent with bounded sigmoid and log-loss. Maximize validation F1 to choose the operating threshold. Freeze it before evaluating the test set. Report ROC AUC, F1, precision, recall, confusion matrix, accuracy and Brier score. Calibration bins expose differences between predicted probability and observed frequency.

## Public-data model comparison

The Python benchmark now uses 1,797 real handwritten-digit images bundled with scikit-learn. It compares standardized logistic regression with a random forest using validation macro F1, then refits the winner and evaluates the untouched test set. Scaling is fitted within the pipeline, never on the complete dataset.

```bash
python -m pip install -r requirements-benchmark.txt
python public_benchmark.py
```

Seed 42: random forest selected; test accuracy **96.67%**, macro F1 **0.9666**, constant-class baseline **10.28%** on 360 held-out images. See [public-results.json](public-results.json) for every split index, versions and confusion matrix. These are digit-recognition results, not student-risk predictions. The browser demo remains the synthetic binary-classification exercise.

Data provenance: [UCI Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), accessed through scikit-learn's bundled `load_digits`. Cite the dataset when reusing it; the repository MIT license applies to project code, not third-party data.

## Measured evidence

Reproduced with Python by `python benchmark.py`. Execution timestamp is recorded in the JSON report. Synthetic JavaScript values may differ because the random generator differs.

| Measurement | Result |
|---|---|
| mean roc auc | 0.9331 |
| mean f1 | 0.8530 |
| mean brier score | 0.1066 |

## Run locally

Python 3.11 or 3.12. Core synthetic exercises use the standard library; the real-data models require the declared Python dependencies. No API keys.

```bash
git clone https://github.com/erickvera610/ml-model-lab.git
cd ml-model-lab
python app.py
```

Open http://127.0.0.1:8000. Stop with Ctrl+C. The public browser demo requires no installation.

## Reproduce and inspect

```bash
python -m unittest discover -v
node test_enhanced.cjs
python benchmark.py
python daily_exercise.py
```

Use `core.run({"mode":"evaluate","seed":42})`. `benchmark.py` repeats seeds 0–4 and reports their mean, rather than showcasing a favorable random seed.

## Read the source

- `core.py`: public algorithm entry point and preserved original exercise.
- `evaluation.py`: featured implementation.
- `enhanced.js`: browser implementation, readable without a build system.
- `app.py`: localhost HTTP API.
- `test_*.py` and `test_enhanced.cjs`: behavior and edge cases.
- `benchmark.py`: evidence generator.
- `pytorch_digits.py`: PyTorch neural baseline on the same public digit data.
- `robustness_benchmark.py`: fixed held-out corruption protocol.

## Limitations

Synthetic binary labels and two features are deliberately simple. Five seeds do not replace cross-validation on real data. Calibration estimates use only 100 test observations. F1 reflects one error tradeoff; production threshold choice should reflect business costs.

## Daily practical exercises

One dated fictional scenario runs daily through GitHub Actions. `exercises/YYYY-MM-DD/result.json` preserves inputs and outputs; `samples/latest.json` holds the latest input. Bot-authored report commits document script execution, not manual daily development. [Daily Spanish guide](EJERCICIOS_DIARIOS_ES.md).

## Authorship and relevance

Independent AI-assisted learning project in Erik Vera's portfolio. No affiliation with FOX, Tesla or BCG. The documentation distinguishes supplied implementation from improvements the learner actually makes. Code is MIT licensed; third-party data provenance is documented separately.

## Three featured portfolio projects

- [ML Model Lab](https://github.com/erickvera610/ml-model-lab): model selection and public-data evaluation.
- [Road Vision Lab](https://github.com/erickvera610/road-vision-lab): image processing and explicit failure analysis.
- [Media Content Lab](https://github.com/erickvera610/media-content-lab): regression and chronological evaluation.
