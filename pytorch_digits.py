"""PyTorch benchmark on the same real sklearn digits dataset used elsewhere in this repo.

Purpose: add framework-level deep-learning evidence without replacing the existing
scikit-learn experiment. The test split is held out from model selection.
"""
from pathlib import Path
import json
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT = Path(__file__).parent
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

class DigitMLP(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, hidden),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden, 10),
        )
    def forward(self, x):
        return self.net(x)

def split_data(seed=SEED):
    data = load_digits()
    ids = np.arange(len(data.target))
    tr, rest = train_test_split(ids, test_size=.4, stratify=data.target, random_state=seed)
    va, te = train_test_split(rest, test_size=.5, stratify=data.target[rest], random_state=seed)
    mean = data.data[tr].mean(axis=0)
    scale = data.data[tr].std(axis=0)
    scale[scale == 0] = 1.0
    x = ((data.data - mean) / scale).astype('float32')
    y = data.target.astype('int64')
    return x, y, tr, va, te

def train_candidate(x, y, tr, va, hidden):
    model = DigitMLP(hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.tensor(x[tr]), torch.tensor(y[tr])), batch_size=64, shuffle=True)
    best = None
    patience = 15
    stale = 0
    history = []
    for epoch in range(150):
        model.train()
        for xb, yb in loader:
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(x[va])).argmax(1).numpy()
        score = f1_score(y[va], pred, average='macro')
        history.append(float(score))
        if best is None or score > best[0]:
            best = (score, {k: v.detach().clone() for k, v in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
            if stale >= patience: break
    model.load_state_dict(best[1])
    return model, history, float(best[0])

def evaluate(model, x, y, ids):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x[ids]))
        pred = logits.argmax(1).numpy()
    return {
        'accuracy': float(accuracy_score(y[ids], pred)),
        'macro_f1': float(f1_score(y[ids], pred, average='macro')),
        'confusion_matrix': confusion_matrix(y[ids], pred).tolist(),
    }

def main():
    x, y, tr, va, te = split_data()
    candidates = {}
    models = {}
    for hidden in (32, 64, 128):
        torch.manual_seed(SEED)
        model, history, score = train_candidate(x, y, tr, va, hidden)
        candidates[str(hidden)] = {'validation_macro_f1': score, 'epochs': len(history)}
        models[hidden] = model
    winner = max(candidates, key=lambda k: candidates[k]['validation_macro_f1'])
    model = models[int(winner)]
    result = {
        'framework': torch.__version__,
        'dataset': 'sklearn load_digits / UCI Optical Recognition of Handwritten Digits',
        'seed': SEED,
        'split': {'train': len(tr), 'validation': len(va), 'test': len(te)},
        'selection': 'Hidden width selected on validation macro F1; test evaluated once after selection.',
        'candidates': candidates,
        'selected_hidden_width': int(winner),
        'test': evaluate(model, x, y, te),
        'limitations': 'Small 8x8 digit benchmark; CPU-scale MLP. This does not demonstrate transformers, CUDA optimization, distributed training, or production MLOps.'
    }
    out = ROOT/'artifacts'; out.mkdir(exist_ok=True)
    (out/'pytorch-digits-results.json').write_text(json.dumps(result, indent=2))
    torch.save(model.state_dict(), out/'pytorch-digits.pt')
    print(json.dumps(result['test']))

if __name__ == '__main__': main()
