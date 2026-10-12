# Alzheimer Early-Detection AI Lab

> **Portfolio project #6 — Medical AI.** A reproducible research prototype for Alzheimer-related digital biomarker classification using real public data, classical machine learning, PyTorch, calibrated probabilities, uncertainty estimates, robustness testing, an inference API, Docker, tests and CI.

**Medical safety:** This repository is for research, education and portfolio demonstration only. It does **not** diagnose Alzheimer's disease, replace clinical evaluation, or constitute a medical device.

## Recruiter quick view

**Problem:** distinguish Alzheimer patients from healthy controls using digital handwriting biomarkers from the public DARWIN dataset.

**Data:** 174 participants, 451 raw variables, UCI dataset ID 732, CC BY 4.0, DOI `10.24432/C55D0K`.

**Models:** calibrated logistic regression / random forest selected by 5-fold CV, plus a PyTorch MLP benchmark with MC-dropout uncertainty.

**Engineering:** modular Python package, reproducible seed, held-out test set, robustness benchmark, FastAPI endpoint, Docker, pytest and GitHub Actions.

Start with [RECRUITER_OVERVIEW.md](RECRUITER_OVERVIEW.md), `train.py`, `src/alzheimers_ai/modeling.py`, and `tests/test_pipeline.py`.

## Why this medical application

Indiana University School of Medicine conducts active AI/deep-learning research for Alzheimer's disease, including early detection and risk prediction using neuroimaging, whole-genome sequencing and multi-omics. The Indiana Alzheimer's Disease Research Center also integrates imaging, biomarkers and AI for precision approaches to cognitive health.

This project is an **independent portfolio project** inspired by that broad medical-AI problem area. It is not sponsored by, affiliated with, or a reproduction of an IU laboratory's work.

## Scientific question

Can digital handwriting dynamics provide enough signal for a machine-learning model to distinguish Alzheimer patients from healthy participants, while remaining calibrated, uncertainty-aware and robust to moderate input degradation?

The goal is not to maximize one headline metric. The project evaluates discrimination, thresholded classification, calibration and robustness separately.

## Dataset

The DARWIN dataset contains online handwriting-derived measurements from 174 people and 451 variables. It was created for Alzheimer classification research and is distributed by the UCI Machine Learning Repository under CC BY 4.0.

The training script downloads it through `ucimlrepo`; data files are not committed to this repository.

## Architecture

```text
DARWIN public data
      |
      v
schema cleaning + ID removal
      |
      +-------------------------+
      |                         |
      v                         v
classical ML                PyTorch MLP
Logistic / RF               451 -> 128 -> 64 -> 1
5-fold CV selection         dropout + AdamW
      |                         |
calibration                   MC dropout
      |                         |
      +-----------+-------------+
                  v
          untouched test set
                  |
      ROC-AUC / F1 / balanced acc
      Brier score / uncertainty
                  |
          robustness stress tests
                  |
             FastAPI service
```

## Leakage controls

- The test split is created before model selection.
- Classical model choice uses only training data via stratified 5-fold CV.
- Scaling and feature selection live inside sklearn pipelines.
- PyTorch early stopping uses a validation subset drawn only from training data.
- The test set is used once for final evaluation.

## Robustness benchmark

The classical model is re-evaluated on the fixed held-out participants after:

- Gaussian feature noise,
- 5% random missingness with median recovery,
- 10% feature dropout.

This makes failure behavior visible instead of showing only clean-data performance.

## Uncertainty

The PyTorch model keeps dropout active for repeated test-time passes. Mean predicted probability is reported together with prediction standard deviation. This is a research uncertainty signal, **not** a clinically validated confidence interval.

## Run

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=src  # Windows PowerShell: $env:PYTHONPATH="src"
pytest -q
python train.py
```

Outputs are written to `artifacts/results.json`; trained binary model files are intentionally git-ignored.

## API

After training:

```bash
uvicorn alzheimers_ai.api:app --app-dir src --reload
```

- `GET /health` confirms service status.
- `POST /predict` accepts the complete DARWIN feature schema and returns a research probability plus a safety warning.

Docker:

```bash
docker build -t alzheimers-ai-lab .
docker run -p 8000:8000 alzheimers-ai-lab
```

## Metrics reported

- ROC AUC
- Accuracy
- Balanced accuracy
- F1
- Brier score
- mean MC-dropout uncertainty
- clean vs corrupted robustness deltas

No metric is described as clinical sensitivity/specificity validation because this small research dataset is not a prospective clinical study.

## Limitations

DARWIN is small (174 participants), high-dimensional and based on a specific handwriting protocol. External validity is unknown. The project does not use MRI, PET, EHR, genomics or multi-omics. It does not establish clinical utility, causal biomarkers, fairness across demographic groups, or prospective performance. Larger external cohorts and clinical governance would be required before any medical use.

## Next research extensions

1. Nested cross-validation and bootstrap confidence intervals.
2. Grouped feature analysis by handwriting task.
3. External validation on an independent digital-biomarker cohort.
4. Multimodal fusion with imaging or biomarker representations where appropriately licensed data are available.
5. Explainability stability analysis rather than single-sample feature-attribution screenshots.

## Provenance and attribution

DARWIN dataset: Fontanella, F. (2022), UCI Machine Learning Repository, DOI `10.24432/C55D0K`, CC BY 4.0. Dataset license applies to the dataset; repository code is a separate independent implementation.

## Authorship

Independent AI-assisted learning and portfolio project by Erik Vera. No affiliation with Indiana University School of Medicine research laboratories is claimed by this repository. IU is cited only as motivation for the applied medical-AI problem area.
