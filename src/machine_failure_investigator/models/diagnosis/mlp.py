"""Multi-layer perceptron fault classifier (PyTorch)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover
    torch = None
    nn = None

from sklearn.preprocessing import LabelEncoder


class _MLP(nn.Module if nn is not None else object):
    def __init__(self, n_in: int, n_classes: int, hidden: Sequence[int], dropout: float) -> None:
        if nn is None:
            raise ImportError("PyTorch required")
        super().__init__()
        layers: List[nn.Module] = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # type: ignore[no-untyped-def]
        return self.net(x)


class MLPClassifier:
    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 40,
        weight_decay: float = 1e-4,
        seed: int = 42,
    ) -> None:
        self.hidden_dims = hidden_dims or [128, 64, 32]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.seed = seed
        self.model: Optional[_MLP] = None
        self.label_encoder = LabelEncoder()
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.feature_names: List[str] = []
        self.is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "MLPClassifier":
        if torch is None:
            raise ImportError("PyTorch required")
        torch.manual_seed(self.seed)
        self.feature_names = list(feature_names or [f"f{i}" for i in range(X.shape[1])])
        y_enc = self.label_encoder.fit_transform(y)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-8
        Xn = (X - self.mean_) / self.std_
        n_classes = len(self.label_encoder.classes_)
        self.model = _MLP(X.shape[1], n_classes, self.hidden_dims, self.dropout)
        opt = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        loss_fn = nn.CrossEntropyLoss()
        ds = TensorDataset(
            torch.tensor(Xn, dtype=torch.float32),
            torch.tensor(y_enc, dtype=torch.long),
        )
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                opt.zero_grad()
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
        self.is_fitted = True
        return self

    def _prepare(self, X: np.ndarray):
        assert self.mean_ is not None and self.std_ is not None and self.model is not None
        Xn = (X - self.mean_) / self.std_
        return torch.tensor(Xn, dtype=torch.float32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if torch is None or self.model is None:
            raise RuntimeError("Not fitted")
        self.model.eval()
        with torch.no_grad():
            logits = self.model(self._prepare(X))
            return torch.softmax(logits, dim=1).numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.label_encoder.inverse_transform(idx)

    def classes(self) -> np.ndarray:
        return self.label_encoder.classes_

    def save(self, path: str | Path) -> None:
        if torch is None or self.model is None:
            raise RuntimeError("Nothing to save")
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
                "mean": self.mean_,
                "std": self.std_,
                "classes": self.label_encoder.classes_,
                "feature_names": self.feature_names,
                "n_features": self.mean_.shape[0] if self.mean_ is not None else 0,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "MLPClassifier":
        if torch is None:
            raise ImportError("PyTorch required")
        obj = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(hidden_dims=obj["hidden_dims"], dropout=obj["dropout"])
        model.mean_ = obj["mean"]
        model.std_ = obj["std"]
        model.feature_names = obj["feature_names"]
        model.label_encoder.classes_ = obj["classes"]
        n_classes = len(obj["classes"])
        model.model = _MLP(obj["n_features"], n_classes, model.hidden_dims, model.dropout)
        model.model.load_state_dict(obj["state_dict"])
        model.is_fitted = True
        return model
