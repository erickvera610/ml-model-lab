# Recruiter overview

## What this project demonstrates

- Applied medical AI on a real, openly licensed Alzheimer dataset.
- Leakage-aware train/test separation and cross-validated model selection.
- Classical ML and PyTorch deep-learning comparison.
- Probability calibration and Brier score, not accuracy alone.
- Monte Carlo dropout uncertainty estimates.
- Robustness tests under noise, missing values and feature dropout.
- FastAPI inference service, Docker packaging, pytest and CI.
- Explicit medical-safety boundaries and dataset provenance.

## Why Alzheimer / why IU relevance?

Indiana University School of Medicine has active AI/deep-learning research in Alzheimer's disease spanning neuroimaging, genomics, proteomics, metabolomics and precision medicine. This portfolio project does **not** reproduce IU research or claim an IU affiliation. It independently demonstrates the same broad applied-AI problem area using the public DARWIN digital-handwriting biomarker dataset.

## Review in 3 minutes

1. Read `README.md` for the scientific question and architecture.
2. Inspect `train.py` for split/model-selection discipline.
3. Inspect `src/alzheimers_ai/modeling.py` for classical + PyTorch models.
4. Inspect `src/alzheimers_ai/robustness.py` for failure testing.
5. Inspect `tests/test_pipeline.py` and CI for software-engineering evidence.
