"""Stroke classifiers: a RandomForest baseline and a 1D-CNN, behind one interface.

Both backends consume the same pose windows and emit the same
``(label, confidence)``, so the pipeline is agnostic to which is loaded. The
RandomForest is the default: it trains in seconds on CPU, needs no GPU, and gives a
verifiable end-to-end path before the temporal model is worth the effort.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from .features import (
    NUM_AGGREGATE_FEATURES,
    NUM_FRAME_FEATURES,
    aggregate_features,
    frame_features,
    impute_frame_features,
    resample_series,
)

#: Fixed timestep count the CNN backend resamples every window to.
CNN_SEQUENCE_LENGTH = 16


class StrokeClassifier(ABC):
    """Common interface for stroke classification backends."""

    labels: list[str]

    @abstractmethod
    def fit(self, windows: list[np.ndarray], labels: list[str]) -> dict[str, float]:
        """Train on labelled pose windows.

        Args:
            windows: ``(T, 17, 3)`` pose windows, one per labelled clip.
            labels: Stroke label for each window.

        Returns:
            Training metrics (e.g. ``{"train_accuracy": 0.93}``).
        """

    @abstractmethod
    def predict(self, window: np.ndarray) -> tuple[str, float]:
        """Classify one pose window.

        Args:
            window: ``(T, 17, 3)`` pose keypoints.

        Returns:
            ``(label, confidence)`` where confidence is in ``[0, 1]``.
        """

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the trained model to ``path``."""

    def score(self, windows: list[np.ndarray], labels: list[str]) -> float:
        """Accuracy over a labelled set.

        Args:
            windows: Pose windows.
            labels: Ground-truth stroke labels.

        Returns:
            Fraction correctly classified, or 0.0 for an empty set.
        """
        if not windows:
            return 0.0
        correct = sum(
            self.predict(window)[0] == label for window, label in zip(windows, labels)
        )
        return correct / len(windows)


class RandomForestStrokeClassifier(StrokeClassifier):
    """RandomForest over aggregate pose-window statistics."""

    def __init__(
        self,
        labels: list[str] | None = None,
        n_estimators: int = 300,
        max_depth: int | None = None,
        random_state: int = 0,
    ) -> None:
        """Construct an untrained classifier.

        Args:
            labels: Known stroke labels. Inferred from the training data if omitted.
            n_estimators: Number of trees.
            max_depth: Maximum tree depth (``None`` grows until pure).
            random_state: Seed, for reproducible training.
        """
        from sklearn.ensemble import RandomForestClassifier

        self.labels = list(labels) if labels else []
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, windows: list[np.ndarray], labels: list[str]) -> dict[str, float]:
        """Train the forest. See :meth:`StrokeClassifier.fit`."""
        if not windows:
            raise ValueError("cannot train on an empty dataset")
        if len(windows) != len(labels):
            raise ValueError(
                f"windows ({len(windows)}) and labels ({len(labels)}) length mismatch"
            )

        features = np.stack([aggregate_features(frame_features(w)) for w in windows])
        self._model.fit(features, labels)
        self.labels = list(self._model.classes_)
        self._fitted = True
        return {"train_accuracy": float(self._model.score(features, labels))}

    def predict(self, window: np.ndarray) -> tuple[str, float]:
        """Classify a pose window. See :meth:`StrokeClassifier.predict`."""
        if not self._fitted:
            raise RuntimeError("classifier is not trained; call fit() or load()")
        features = aggregate_features(frame_features(window)).reshape(1, -1)
        probabilities = self._model.predict_proba(features)[0]
        best = int(np.argmax(probabilities))
        return str(self._model.classes_[best]), float(probabilities[best])

    def save(self, path: str | Path) -> None:
        """Persist the forest with joblib. See :meth:`StrokeClassifier.save`."""
        import joblib

        if not self._fitted:
            raise RuntimeError("refusing to save an untrained classifier")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "backend": "random_forest",
                "model": self._model,
                "labels": self.labels,
                "num_features": NUM_AGGREGATE_FEATURES,
            },
            destination,
        )

    @classmethod
    def load(cls, path: str | Path) -> "RandomForestStrokeClassifier":
        """Load a saved forest.

        Args:
            path: Path written by :meth:`save`.

        Returns:
            The restored classifier.
        """
        import joblib

        payload = joblib.load(Path(path))
        instance = cls(labels=payload["labels"])
        instance._model = payload["model"]
        instance._fitted = True
        return instance


class _CNN1D:
    """Lazily-constructed torch module for the temporal backend."""

    @staticmethod
    def build(num_classes: int) -> Any:
        """Build the 1D-CNN. Kept small deliberately — shot datasets are tiny."""
        from torch import nn

        return nn.Sequential(
            nn.Conv1d(NUM_FRAME_FEATURES, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )


class CNN1DStrokeClassifier(StrokeClassifier):
    """Small 1D convolutional network over the per-frame pose feature sequence."""

    def __init__(
        self,
        labels: list[str] | None = None,
        epochs: int = 60,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        sequence_length: int = CNN_SEQUENCE_LENGTH,
        random_state: int = 0,
    ) -> None:
        """Construct an untrained classifier.

        Args:
            labels: Known stroke labels. Inferred from training data if omitted.
            epochs: Training epochs.
            batch_size: Minibatch size.
            learning_rate: Adam learning rate.
            sequence_length: Timesteps every window is resampled to.
            random_state: Seed, for reproducible training.
        """
        self.labels = list(labels) if labels else []
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.sequence_length = sequence_length
        self.random_state = random_state
        self._model: Any = None

    def _prepare(self, windows: list[np.ndarray]) -> np.ndarray:
        """Turn raw pose windows into a ``(B, F, T)`` tensor-ready array."""
        sequences = [
            resample_series(impute_frame_features(frame_features(w)), self.sequence_length)
            for w in windows
        ]
        # (B, T, F) -> (B, F, T), the layout Conv1d expects.
        return np.stack(sequences).transpose(0, 2, 1)

    def fit(self, windows: list[np.ndarray], labels: list[str]) -> dict[str, float]:
        """Train the network. See :meth:`StrokeClassifier.fit`."""
        import torch
        from torch import nn

        if not windows:
            raise ValueError("cannot train on an empty dataset")
        if len(windows) != len(labels):
            raise ValueError(
                f"windows ({len(windows)}) and labels ({len(labels)}) length mismatch"
            )

        torch.manual_seed(self.random_state)
        self.labels = sorted(set(labels))
        index = {label: i for i, label in enumerate(self.labels)}

        x = torch.from_numpy(self._prepare(windows)).float()
        y = torch.tensor([index[label] for label in labels], dtype=torch.long)

        self._model = _CNN1D.build(len(self.labels))
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        self._model.train()
        for _ in range(self.epochs):
            permutation = torch.randperm(len(x))
            for start in range(0, len(x), self.batch_size):
                batch = permutation[start : start + self.batch_size]
                # BatchNorm needs >1 sample; skip a trailing singleton batch.
                if len(batch) < 2:
                    continue
                optimizer.zero_grad()
                loss = criterion(self._model(x[batch]), y[batch])
                loss.backward()
                optimizer.step()

        self._model.eval()
        with torch.no_grad():
            accuracy = float((self._model(x).argmax(dim=1) == y).float().mean())
        return {"train_accuracy": accuracy}

    def predict(self, window: np.ndarray) -> tuple[str, float]:
        """Classify a pose window. See :meth:`StrokeClassifier.predict`."""
        import torch

        if self._model is None:
            raise RuntimeError("classifier is not trained; call fit() or load()")
        x = torch.from_numpy(self._prepare([window])).float()
        with torch.no_grad():
            probabilities = torch.softmax(self._model(x), dim=1)[0].numpy()
        best = int(np.argmax(probabilities))
        return self.labels[best], float(probabilities[best])

    def save(self, path: str | Path) -> None:
        """Persist weights and label mapping. See :meth:`StrokeClassifier.save`."""
        import torch

        if self._model is None:
            raise RuntimeError("refusing to save an untrained classifier")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "backend": "cnn1d",
                "state_dict": self._model.state_dict(),
                "labels": self.labels,
                "sequence_length": self.sequence_length,
            },
            destination,
        )

    @classmethod
    def load(cls, path: str | Path) -> "CNN1DStrokeClassifier":
        """Load saved weights.

        Args:
            path: Path written by :meth:`save`.

        Returns:
            The restored classifier.
        """
        import torch

        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        instance = cls(
            labels=payload["labels"], sequence_length=payload["sequence_length"]
        )
        instance._model = _CNN1D.build(len(instance.labels))
        instance._model.load_state_dict(payload["state_dict"])
        instance._model.eval()
        return instance


def build_classifier(backend: str, labels: list[str] | None = None) -> StrokeClassifier:
    """Construct an untrained classifier for the named backend.

    Args:
        backend: ``"random_forest"`` or ``"cnn1d"``.
        labels: Known stroke labels.

    Returns:
        An untrained classifier.

    Raises:
        ValueError: If ``backend`` is not recognised.
    """
    if backend == "random_forest":
        return RandomForestStrokeClassifier(labels=labels)
    if backend == "cnn1d":
        return CNN1DStrokeClassifier(labels=labels)
    raise ValueError(f"unknown backend {backend!r}; expected 'random_forest' or 'cnn1d'")


def load_classifier(path: str | Path) -> StrokeClassifier:
    """Load a trained classifier, dispatching on the backend recorded in the file.

    Args:
        path: Path to a saved classifier.

    Returns:
        The restored classifier.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file records an unknown backend.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"stroke classifier not found at {source}. "
            f"Train one with scripts/train_stroke_classifier.py (see README)."
        )

    # The two backends use different serialisation formats; sniff joblib first since
    # it is the default, and fall back to the torch checkpoint.
    try:
        import joblib

        payload = joblib.load(source)
        if isinstance(payload, dict) and payload.get("backend") == "random_forest":
            return RandomForestStrokeClassifier.load(source)
    except Exception:  # noqa: BLE001 - not a joblib file; try torch below
        pass

    import torch

    payload = torch.load(source, map_location="cpu", weights_only=False)
    backend = payload.get("backend") if isinstance(payload, dict) else None
    if backend == "cnn1d":
        return CNN1DStrokeClassifier.load(source)
    raise ValueError(f"unrecognised stroke classifier file: {source} (backend={backend!r})")


def write_label_metadata(path: str | Path, labels: list[str], metrics: dict[str, float]) -> None:
    """Write a sidecar JSON describing a trained classifier.

    Args:
        path: Path to the saved model; the sidecar gets a ``.json`` suffix.
        labels: Stroke labels the model predicts.
        metrics: Training/validation metrics to record.
    """
    sidecar = Path(path).with_suffix(".json")
    sidecar.write_text(json.dumps({"labels": labels, "metrics": metrics}, indent=2))
