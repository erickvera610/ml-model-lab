# Case study: Classification Evaluation Lab

## 1. Problem and intended user

The original logistic-regression learning demo is preserved. The featured version evaluates classification quality, ranking and probability calibration separately.

## 2. Data and information flow

300 training observations → 100 validation observations → selected threshold → 100 test observations

## 3. Why these choices

Use gradient descent with bounded sigmoid and log-loss. Maximize validation F1 to choose the operating threshold. Freeze it before evaluating the test set. Report ROC AUC, F1, precision, recall, confusion matrix, accuracy and Brier score. Calibration bins expose differences between predicted probability and observed frequency.

## 4. Reproducible evidence

| Measurement | Result |
|---|---|
| mean roc auc | 0.9331 |
| mean f1 | 0.8530 |
| mean brier score | 0.1066 |

Generated at 2026-10-10T18:37:13.504044+00:00 by the Python benchmark. Read every run in [benchmark-results.json](benchmark-results.json). These results apply only to the documented benchmark. No production performance claim is made.

## 5. What could go wrong

Synthetic binary labels and two features are deliberately simple. Five seeds do not replace cross-validation on real data. Calibration estimates use only 100 test observations. F1 reflects one error tradeoff; production threshold choice should reflect business costs.

## 6. Next substantive improvement

Implementa un costo distinto para falsos positivos y falsos negativos. Selecciona el umbral en validación según ese costo y compara con F1, manteniendo intacta la prueba final.

Open an issue, create a branch, implement the improvement, run the checks, and explain the change in a pull request. Do not describe automated report commits as personal code development.
