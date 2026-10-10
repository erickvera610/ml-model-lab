# Classification Evaluation Lab

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

Python 3.11 or 3.12, standard library only. No API keys.

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
