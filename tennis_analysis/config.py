"""Typed configuration loaded from a single YAML (or JSON) file.

Dataclasses mirror the structure of ``config.yaml``. Unknown keys raise, so a typo in
the config file surfaces immediately instead of silently falling back to a default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml

T = TypeVar("T")


@dataclass
class VideoConfig:
    """Frame sampling and output timing."""

    frame_stride: int = 1
    max_frames: int | None = None
    output_fps: float | None = None


@dataclass
class PlayerConfig:
    """Player detection and tracking."""

    model: str = "yolo11n.pt"
    confidence: float = 0.5
    tracker: str = "bytetrack.yaml"
    court_margin_m: float = 3.0
    min_track_presence: float = 0.1


@dataclass
class KalmanConfig:
    """Ball Kalman filter tuning."""

    process_noise: float = 1.0
    measurement_noise: float = 10.0
    max_age: int = 12
    max_gate_distance: float = 150.0


@dataclass
class BallConfig:
    """Ball detection and trajectory smoothing."""

    model: str = "models/ball_yolo11n.pt"
    confidence: float = 0.15
    imgsz: int = 640
    kalman: KalmanConfig = field(default_factory=KalmanConfig)


@dataclass
class CourtConfig:
    """Court keypoint model and homography quality gates."""

    model: str = "models/court_keypoints_resnet18.pt"
    input_size: int = 224
    min_keypoint_confidence: float = 0.5
    ransac_threshold: float = 5.0
    max_reprojection_error_m: float = 1.0
    refit_interval: int = 30


@dataclass
class AnalyticsConfig:
    """Speed/distance smoothing and plausibility limits."""

    speed_window: int = 5
    max_player_speed_kmh: float = 40.0
    max_ball_speed_kmh: float = 250.0


@dataclass
class StrokeConfig:
    """Pose extraction, shot-moment detection and stroke classification."""

    enabled: bool = True
    pose_model: str = "yolo11n-pose.pt"
    classifier: str = "models/stroke_classifier.joblib"
    backend: Literal["random_forest", "cnn1d"] = "random_forest"
    window_seconds: float = 0.25
    min_direction_change_deg: float = 45.0
    min_speed_change: float = 4.0
    min_shot_interval_s: float = 0.4
    labels: list[str] = field(
        default_factory=lambda: ["serve", "forehand", "backhand", "volley"]
    )


@dataclass
class MiniCourtConfig:
    """Mini-court overlay geometry, in pixels."""

    width: int = 220
    height: int = 460
    margin: int = 24
    position: Literal["top_left", "top_right", "bottom_left", "bottom_right"] = (
        "top_right"
    )


@dataclass
class OutputConfig:
    """Artefact filenames and overlay toggles."""

    video_filename: str = "annotated.mp4"
    shots_csv: str = "shots.csv"
    shots_json: str = "shots.json"
    tracks_csv: str = "tracks.csv"
    draw_mini_court: bool = True
    draw_speeds: bool = True
    draw_stroke_labels: bool = True
    mini_court: MiniCourtConfig = field(default_factory=MiniCourtConfig)


@dataclass
class Config:
    """Root configuration object for the whole pipeline."""

    video: VideoConfig = field(default_factory=VideoConfig)
    players: PlayerConfig = field(default_factory=PlayerConfig)
    ball: BallConfig = field(default_factory=BallConfig)
    court: CourtConfig = field(default_factory=CourtConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    stroke: StrokeConfig = field(default_factory=StrokeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        """Load configuration from a YAML or JSON file.

        Args:
            path: Path to the config file. If ``None``, all defaults are used.

        Returns:
            The populated :class:`Config`.

        Raises:
            FileNotFoundError: If ``path`` is given but does not exist.
            ValueError: If the file contains keys the schema does not define.
        """
        if path is None:
            return cls()

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"config file not found: {path}")

        text = path.read_text()
        raw = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
        return _from_dict(cls, raw or {}, path.name)


def _from_dict(cls: type[T], data: dict[str, Any], context: str) -> T:
    """Recursively build a dataclass from a mapping, rejecting unknown keys."""
    if not isinstance(data, dict):
        raise ValueError(f"{context}: expected a mapping for {cls.__name__}, got {data!r}")

    known = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(
            f"{context}: unknown key(s) for {cls.__name__}: {sorted(unknown)}. "
            f"Valid keys: {sorted(known)}"
        )

    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        field_type = known[name].type
        # Resolve string annotations produced by `from __future__ import annotations`.
        resolved = _resolve(field_type)
        if is_dataclass(resolved) and value is not None:
            kwargs[name] = _from_dict(resolved, value, context)
        else:
            kwargs[name] = value
    return cls(**kwargs)  # type: ignore[return-value]


_NESTED_TYPES = {
    "VideoConfig": VideoConfig,
    "PlayerConfig": PlayerConfig,
    "KalmanConfig": KalmanConfig,
    "BallConfig": BallConfig,
    "CourtConfig": CourtConfig,
    "AnalyticsConfig": AnalyticsConfig,
    "StrokeConfig": StrokeConfig,
    "MiniCourtConfig": MiniCourtConfig,
    "OutputConfig": OutputConfig,
}


def _resolve(annotation: Any) -> Any:
    """Map a (possibly string) annotation to a nested config class, if it is one."""
    if isinstance(annotation, str):
        return _NESTED_TYPES.get(annotation, annotation)
    return annotation
