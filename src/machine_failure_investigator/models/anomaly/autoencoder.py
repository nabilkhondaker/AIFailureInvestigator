"""Simple feed-forward autoencoder for reconstruction-based anomaly detection."""

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


class _AE(nn.Module if nn is not None else object):
    def __init__(self, n_features: int, hidden: Sequence[int], latent: int) -> None:
        if nn is None:
            raise ImportError("PyTorch is required for AutoencoderDetector")
        super().__init__()
        dims = [n_features, *hidden, latent]
        enc_layers = []
        for i in range(len(dims) - 1):
            enc_layers += [nn.Linear(dims[i], dims[i + 1]), nn.ReLU()]
        self.encoder = nn.Sequential(*enc_layers)
        dec_dims = [latent, *reversed(hidden), n_features]
        dec_layers = []
        for i in range(len(dec_dims) - 1):
            dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i < len(dec_dims) - 2:
                dec_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):  # type: ignore[no-untyped-def]
        z = self.encoder(x)
        return self.decoder(z)


class AutoencoderDetector:
    def __init__(
        self,
        hidden_dims: Optional[List[int]] = None,
        latent_dim: int = 8,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        epochs: int = 30,
        threshold_quantile: float = 0.95,
        seed: int = 42,
    ) -> None:
        self.hidden_dims = hidden_dims or [64, 32, 16]
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.threshold_quantile = threshold_quantile
        self.seed = seed
        self.model: Optional[_AE] = None
        self.threshold_: Optional[float] = None
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.is_fitted = False

    def fit(self, X_healthy: np.ndarray) -> "AutoencoderDetector":
        if torch is None:
            raise ImportError("PyTorch is required for AutoencoderDetector")
        torch.manual_seed(self.seed)
        self.mean_ = X_healthy.mean(axis=0)
        self.std_ = X_healthy.std(axis=0) + 1e-8
        Xn = (X_healthy - self.mean_) / self.std_
        n_features = Xn.shape[1]
        self.model = _AE(n_features, self.hidden_dims, self.latent_dim)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        ds = TensorDataset(torch.tensor(Xn, dtype=torch.float32))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        self.model.train()
        for _ in range(self.epochs):
            for (batch,) in loader:
                opt.zero_grad()
                recon = self.model(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                opt.step()
        scores = self.anomaly_score(X_healthy)
        self.threshold_ = float(np.quantile(scores, self.threshold_quantile))
        self.is_fitted = True
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        if torch is None or self.model is None or self.mean_ is None:
            raise RuntimeError("Model not fitted")
        Xn = (X - self.mean_) / self.std_
        self.model.eval()
        with torch.no_grad():
            t = torch.tensor(Xn, dtype=torch.float32)
            recon = self.model(t).numpy()
        return np.mean((Xn - recon) ** 2, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.anomaly_score(X)
        thr = self.threshold_ if self.threshold_ is not None else 0.0
        return (scores >= thr).astype(int)

    def save(self, path: str | Path) -> None:
        if torch is None or self.model is None:
            raise RuntimeError("Nothing to save")
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "hidden_dims": self.hidden_dims,
                "latent_dim": self.latent_dim,
                "mean": self.mean_,
                "std": self.std_,
                "threshold": self.threshold_,
                "n_features": self.mean_.shape[0] if self.mean_ is not None else 0,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "AutoencoderDetector":
        if torch is None:
            raise ImportError("PyTorch required")
        obj = torch.load(path, map_location="cpu", weights_only=False)
        det = cls(hidden_dims=obj["hidden_dims"], latent_dim=obj["latent_dim"])
        det.mean_ = obj["mean"]
        det.std_ = obj["std"]
        det.threshold_ = obj["threshold"]
        det.model = _AE(obj["n_features"], det.hidden_dims, det.latent_dim)
        det.model.load_state_dict(obj["state_dict"])
        det.is_fitted = True
        return det
