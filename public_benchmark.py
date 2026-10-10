"""Offline public-data benchmark; no downloaded data or API keys."""
import json
from pathlib import Path
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def benchmark(seed=42):
    data = load_digits()
    indices = list(range(len(data.target)))
    train, rest = train_test_split(indices, test_size=0.4, stratify=data.target, random_state=seed)
    valid, test = train_test_split(rest, test_size=0.5, stratify=data.target[rest], random_state=seed)
    models = {"logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)),
              "forest": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)}
    scores = {}
    for name, model in models.items():
        model.fit(data.data[train], data.target[train])
        scores[name] = f1_score(data.target[valid], model.predict(data.data[valid]), average="macro")
    selected = max(scores, key=scores.get)
    model = models[selected]
    model.fit(data.data[train + valid], data.target[train + valid])
    prediction = model.predict(data.data[test])
    baseline = DummyClassifier(strategy="most_frequent").fit(data.data[train + valid], data.target[train + valid])
    return {"dataset": "scikit-learn bundled UCI Optical Recognition of Handwritten Digits",
            "source": "https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits",
            "seed": seed, "sklearn_version": sklearn.__version__, "split_ids": {"train": train, "validation": valid, "test": test},
            "selection_metric": "validation macro F1", "validation_scores": scores, "selected_model": selected,
            "test_accuracy": accuracy_score(data.target[test], prediction),
            "test_macro_f1": f1_score(data.target[test], prediction, average="macro"),
            "baseline_accuracy": accuracy_score(data.target[test], baseline.predict(data.data[test])),
            "confusion_matrix": confusion_matrix(data.target[test], prediction).tolist(),
            "limitations": "Small bundled benchmark; no production or educational-outcome claim. Test is used only after selection."}
if __name__ == "__main__":
    report = benchmark()
    Path(__file__).with_name("public-results.json").write_text(json.dumps(report, indent=2))
    print({k: report[k] for k in ["selected_model", "test_accuracy", "test_macro_f1", "baseline_accuracy"]})
