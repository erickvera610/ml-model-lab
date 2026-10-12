from __future__ import annotations
from dataclasses import dataclass
import random
import numpy as np
import torch
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def seed_everything(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def split_data(X, y, seed=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )
    return X_train, X_test, y_train, y_test


def classical_candidates(k_features=80, seed=42):
    logistic = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("select", SelectKBest(f_classif, k=k_features)),
        ("model", LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5)),
    ])
    forest = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("select", SelectKBest(f_classif, k=k_features)),
        ("model", RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3, class_weight="balanced",
            random_state=seed, n_jobs=-1
        )),
    ])
    return {"logistic": logistic, "random_forest": forest}


def evaluate(y_true, p, threshold=0.5):
    pred = (np.asarray(p) >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, p)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "brier": float(brier_score_loss(y_true, p)),
    }


def cross_validated_model_selection(X, y, seed=42, k_features=80):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = {}
    for name, pipe in classical_candidates(k_features, seed).items():
        aucs = []
        for tr, va in folds.split(X, y):
            m = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
            p = m.predict_proba(X.iloc[va])[:, 1]
            aucs.append(roc_auc_score(y.iloc[va], p))
        scores[name] = {"mean_cv_auc": float(np.mean(aucs)), "std_cv_auc": float(np.std(aucs))}
    best = max(scores, key=lambda n: scores[n]["mean_cv_auc"])
    return best, scores


def fit_calibrated(pipe, X, y):
    calibrated = CalibratedClassifierCV(pipe, method="sigmoid", cv=5)
    return calibrated.fit(X, y)


class MLP(torch.nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.30),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


@dataclass
class TorchBundle:
    model: MLP
    mean: np.ndarray
    std: np.ndarray


def train_torch_mlp(X_train, y_train, seed=42, epochs=250, patience=25):
    seed_everything(seed)
    Xtr, Xva, ytr, yva = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=seed)
    mean = Xtr.mean(axis=0).to_numpy(float)
    std = Xtr.std(axis=0).replace(0, 1).to_numpy(float)
    def tx(df): return torch.tensor(((df.to_numpy(float)-mean)/std), dtype=torch.float32)
    xt, xv = tx(Xtr), tx(Xva)
    yt, yv = torch.tensor(ytr.to_numpy(float), dtype=torch.float32), torch.tensor(yva.to_numpy(float), dtype=torch.float32)
    model = MLP(X_train.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best_state, best_auc, stale = None, -1.0, 0
    for _ in range(epochs):
        model.train(); opt.zero_grad(); loss = loss_fn(model(xt), yt); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): p = torch.sigmoid(model(xv)).numpy()
        auc = roc_auc_score(yva, p)
        if auc > best_auc + 1e-4:
            best_auc, stale = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience: break
    model.load_state_dict(best_state)
    return TorchBundle(model=model, mean=mean, std=std), float(best_auc)


def mc_dropout_predict(bundle: TorchBundle, X, passes=50):
    arr = (X.to_numpy(float)-bundle.mean)/bundle.std
    x = torch.tensor(arr, dtype=torch.float32)
    bundle.model.train()
    ps = []
    with torch.no_grad():
        for _ in range(passes): ps.append(torch.sigmoid(bundle.model(x)).numpy())
    p = np.vstack(ps)
    return p.mean(axis=0), p.std(axis=0)
