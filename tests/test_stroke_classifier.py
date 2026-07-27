"""Tests for the stroke classifier backends.

Synthetic pose windows are generated per stroke class with the geometry that actually
distinguishes those strokes — the racket hand overhead for a serve, swung to one side
for a forehand, across the body for a backhand, compact and in front for a volley.
Both backends must learn that separation and survive a save/load round trip.
"""

from __future__ import annotations

import numpy as np
import pytest

from tennis_analysis.stroke_classification.classifier import (
    CNN1DStrokeClassifier,
    RandomForestStrokeClassifier,
    build_classifier,
    load_classifier,
)
from tennis_analysis.stroke_classification.features import NUM_POSE_KEYPOINTS

#: Racket-hand trajectory templates: (start_xy, end_xy) for the right wrist, in the
#: image frame where smaller y is higher up.
_TEMPLATES = {
    "serve": ((120.0, 60.0), (125.0, 10.0)),
    "forehand": ((60.0, 210.0), (200.0, 165.0)),
    "backhand": ((190.0, 210.0), (35.0, 170.0)),
    "volley": ((115.0, 155.0), (150.0, 150.0)),
}


def synth_window(stroke: str, frames: int = 12, rng: np.random.Generator | None = None):
    """Generate a synthetic pose window for a stroke class.

    Args:
        stroke: One of the keys of ``_TEMPLATES``.
        frames: Number of frames in the window.
        rng: Source of jitter, so samples within a class vary.

    Returns:
        ``(frames, 17, 3)`` pose window.
    """
    rng = rng or np.random.default_rng(0)
    (sx, sy), (ex, ey) = _TEMPLATES[stroke]
    jitter = lambda scale=4.0: rng.normal(0, scale)  # noqa: E731

    window = np.zeros((frames, NUM_POSE_KEYPOINTS, 3))
    for t in range(frames):
        progress = t / max(frames - 1, 1)
        wrist = (sx + (ex - sx) * progress + jitter(), sy + (ey - sy) * progress + jitter())
        elbow = ((wrist[0] + 120.0) / 2 + jitter(2), (wrist[1] + 140.0) / 2 + jitter(2))

        layout = {
            0: (100.0, 100.0),
            5: (80.0, 140.0), 6: (120.0, 140.0),      # shoulders
            7: (70.0, 175.0), 8: elbow,               # elbows
            9: (58.0, 205.0), 10: wrist,              # wrists
            11: (85.0, 220.0), 12: (115.0, 220.0),    # hips
            13: (85.0, 290.0), 14: (115.0, 290.0),    # knees
            15: (85.0, 360.0), 16: (115.0, 360.0),    # ankles
        }
        for index in range(NUM_POSE_KEYPOINTS):
            x, y = layout.get(index, (100.0, 150.0))
            window[t, index] = (x + jitter(1.5), y + jitter(1.5), 0.9)
    return window


@pytest.fixture
def dataset():
    """A small balanced synthetic dataset across all four stroke labels."""
    rng = np.random.default_rng(11)
    windows, labels = [], []
    for stroke in _TEMPLATES:
        for _ in range(25):
            windows.append(synth_window(stroke, rng=rng))
            labels.append(stroke)
    return windows, labels


class TestRandomForestBackend:
    """The default CPU baseline."""

    def test_fit_reports_training_accuracy(self, dataset):
        classifier = RandomForestStrokeClassifier()
        metrics = classifier.fit(*dataset)
        assert metrics["train_accuracy"] > 0.9

    def test_learns_separable_strokes(self, dataset):
        classifier = RandomForestStrokeClassifier()
        classifier.fit(*dataset)

        rng = np.random.default_rng(99)
        held_out = [synth_window(s, rng=rng) for s in _TEMPLATES]
        assert classifier.score(held_out, list(_TEMPLATES)) >= 0.75

    def test_labels_are_populated_after_fit(self, dataset):
        classifier = RandomForestStrokeClassifier()
        classifier.fit(*dataset)
        assert sorted(classifier.labels) == sorted(_TEMPLATES)

    def test_predict_returns_label_and_confidence(self, dataset):
        classifier = RandomForestStrokeClassifier()
        classifier.fit(*dataset)
        label, confidence = classifier.predict(synth_window("serve"))
        assert label in _TEMPLATES
        assert 0.0 <= confidence <= 1.0

    def test_save_load_round_trip(self, dataset, tmp_path):
        classifier = RandomForestStrokeClassifier()
        classifier.fit(*dataset)
        window = synth_window("forehand", rng=np.random.default_rng(5))
        expected = classifier.predict(window)

        path = tmp_path / "stroke.joblib"
        classifier.save(path)
        assert load_classifier(path).predict(window) == expected

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="not trained"):
            RandomForestStrokeClassifier().predict(synth_window("serve"))

    def test_save_before_fit_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="untrained"):
            RandomForestStrokeClassifier().save(tmp_path / "x.joblib")

    def test_empty_dataset_raises(self):
        with pytest.raises(ValueError, match="empty dataset"):
            RandomForestStrokeClassifier().fit([], [])

    def test_length_mismatch_raises(self, dataset):
        windows, labels = dataset
        with pytest.raises(ValueError, match="mismatch"):
            RandomForestStrokeClassifier().fit(windows, labels[:-1])

    def test_handles_variable_length_windows(self, dataset):
        """Windows differ in length near video boundaries; aggregation must absorb it."""
        rng = np.random.default_rng(3)
        windows = [synth_window(s, frames=n, rng=rng)
                   for s in _TEMPLATES for n in (5, 9, 20)]
        labels = [s for s in _TEMPLATES for _ in (5, 9, 20)]
        classifier = RandomForestStrokeClassifier()
        classifier.fit(windows, labels)
        assert classifier.predict(synth_window("volley", frames=7))[0] in _TEMPLATES


class TestCNN1DBackend:
    """The temporal backend, behind the same interface."""

    def test_fit_and_predict(self, dataset):
        classifier = CNN1DStrokeClassifier(epochs=25)
        metrics = classifier.fit(*dataset)
        assert "train_accuracy" in metrics

        label, confidence = classifier.predict(synth_window("serve"))
        assert label in _TEMPLATES
        assert 0.0 <= confidence <= 1.0

    def test_save_load_round_trip(self, dataset, tmp_path):
        classifier = CNN1DStrokeClassifier(epochs=10)
        classifier.fit(*dataset)
        window = synth_window("backhand", rng=np.random.default_rng(8))
        expected = classifier.predict(window)

        path = tmp_path / "stroke.pt"
        classifier.save(path)
        restored = load_classifier(path)
        assert isinstance(restored, CNN1DStrokeClassifier)

        label, confidence = restored.predict(window)
        assert label == expected[0]
        assert confidence == pytest.approx(expected[1], abs=1e-5)

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="not trained"):
            CNN1DStrokeClassifier().predict(synth_window("serve"))

    def test_accepts_variable_length_windows(self, dataset):
        classifier = CNN1DStrokeClassifier(epochs=5)
        classifier.fit(*dataset)
        # Resampling means a 4-frame and a 40-frame window are both valid inputs.
        assert classifier.predict(synth_window("serve", frames=4))[0] in _TEMPLATES
        assert classifier.predict(synth_window("serve", frames=40))[0] in _TEMPLATES


class TestBuildAndLoad:
    """Backend selection and file dispatch."""

    def test_build_random_forest(self):
        assert isinstance(build_classifier("random_forest"), RandomForestStrokeClassifier)

    def test_build_cnn1d(self):
        assert isinstance(build_classifier("cnn1d"), CNN1DStrokeClassifier)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="unknown backend"):
            build_classifier("transformer")

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_classifier(tmp_path / "absent.joblib")
