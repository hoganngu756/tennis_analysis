"""Pipeline orchestration — wires stages 1-7 together.

The pipeline reads the video twice. The first pass runs all inference and collects
per-frame observations; the second pass re-reads the video and renders annotations.
Two passes are necessary because several outputs depend on the *whole* sequence:
picking the two on-court players needs every track's court-time, and detecting a shot
moment needs the ball's trajectory after the impact, not just before it. Rendering in
a single streaming pass would mean drawing labels the pipeline does not yet know.

Each model stage degrades gracefully. Missing ball weights disable ball tracking and
stroke classification but leave player tracking intact; missing court weights disable
real-world units but leave detection intact. A partial result beats a stack trace.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .analytics import (
    Position,
    compute_speed_series,
    cumulative_distance,
    export_shots,
    export_tracks,
    summarise_players,
)
from .config import Config
from .court_keypoints.geometry import COURT_LENGTH, CourtHomography, is_inside_court
from .court_keypoints.mini_court import MiniCourt
from .detection import BallDetector, PlayerTracker
from .types import BallDetection, PlayerDetection, Shot
from .video_io import VideoMetadata, VideoReader, VideoWriter
from .visualization import (
    draw_ball,
    draw_court_keypoints,
    draw_frame_info,
    draw_players,
    draw_stroke_label,
    render_mini_court,
)

logger = logging.getLogger(__name__)

#: Maximum distance (metres) between ball and player for a shot to be attributed.
MAX_STRIKER_DISTANCE_M = 4.0
#: Fallback in pixels, used when no homography is available.
MAX_STRIKER_DISTANCE_PX = 250.0
#: Stroke label used when a shot moment is found but no classifier is loaded.
UNCLASSIFIED_STROKE = "unclassified"


@dataclass
class PipelineResult:
    """Everything the pipeline produced for one video."""

    metadata: VideoMetadata
    effective_fps: float
    frame_indices: list[int] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    player_ids: list[int] = field(default_factory=list)
    player_positions: dict[int, list[Position]] = field(default_factory=dict)
    player_speeds: dict[int, list[float | None]] = field(default_factory=dict)
    player_distances: dict[int, list[float]] = field(default_factory=dict)
    ball_positions: list[Position] = field(default_factory=list)
    ball_speeds: list[float | None] = field(default_factory=list)
    shots: list[Shot] = field(default_factory=list)
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    warnings: list[str] = field(default_factory=list)


@dataclass
class _FrameObservation:
    """Raw per-frame observations collected during the analysis pass."""

    frame_index: int
    timestamp: float
    players: list[PlayerDetection]
    ball: BallDetection | None
    homography: CourtHomography | None
    court_keypoints: np.ndarray | None


class TennisAnalysisPipeline:
    """Runs the full analysis over a video and writes all output artefacts."""

    def __init__(self, config: Config, enable_stroke: bool | None = None) -> None:
        """Construct the pipeline and load whichever models are available.

        Args:
            config: Pipeline configuration.
            enable_stroke: Override for ``config.stroke.enabled``.
        """
        self.config = config
        self.enable_stroke = (
            config.stroke.enabled if enable_stroke is None else enable_stroke
        )
        self.warnings: list[str] = []

        self._player_tracker = PlayerTracker(config.players)
        self._ball_detector = self._try_load_ball()
        self._court_detector, self._homography_tracker = self._try_load_court()
        self._pose_estimator, self._stroke_classifier = self._try_load_stroke()

    # --- model loading ------------------------------------------------------------

    def _warn(self, message: str) -> None:
        """Record a degradation warning and log it."""
        logger.warning(message)
        self.warnings.append(message)

    def _try_load_ball(self) -> BallDetector | None:
        """Load the ball detector, returning ``None`` if its weights are missing."""
        try:
            return BallDetector(self.config.ball)
        except FileNotFoundError as exc:
            self._warn(f"ball tracking disabled: {exc}")
            return None

    def _try_load_court(self):
        """Load the court model and homography tracker, or ``(None, None)``."""
        try:
            from .court_keypoints.detector import CourtKeypointDetector, HomographyTracker

            return CourtKeypointDetector(self.config.court), HomographyTracker(self.config.court)
        except FileNotFoundError as exc:
            self._warn(
                f"court keypoints disabled: {exc} "
                f"Speeds and distances will be unavailable (no pixel-to-metre mapping)."
            )
            return None, None

    def _try_load_stroke(self):
        """Load the pose estimator and stroke classifier, or ``(None, None)``."""
        if not self.enable_stroke:
            return None, None
        try:
            from .stroke_classification.classifier import load_classifier
            from .stroke_classification.pose import PoseEstimator

            classifier = load_classifier(self.config.stroke.classifier)
            return PoseEstimator(self.config.stroke.pose_model), classifier
        except (FileNotFoundError, ValueError) as exc:
            self._warn(f"stroke classification disabled: {exc}")
            self.enable_stroke = False
            return None, None

    # --- main entry point ---------------------------------------------------------

    def run(self, video_path: str | Path, output_dir: str | Path) -> PipelineResult:
        """Analyse a video and write all configured outputs.

        Args:
            video_path: Input video file.
            output_dir: Directory for the annotated video, CSV and JSON artefacts.
                Created if it does not exist.

        Returns:
            The populated :class:`PipelineResult`.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        observations, metadata, effective_fps = self._analysis_pass(video_path)
        if not observations:
            self._warn("no frames were read from the video")
            return PipelineResult(
                metadata=metadata, effective_fps=effective_fps, warnings=self.warnings
            )

        result = self._post_process(observations, metadata, effective_fps)
        self._render_pass(video_path, observations, result, output_dir)
        self._export(result, output_dir)
        return result

    # --- pass 1: inference --------------------------------------------------------

    def _analysis_pass(
        self, video_path: str | Path
    ) -> tuple[list[_FrameObservation], VideoMetadata, float]:
        """Run detection over every processed frame, collecting raw observations."""
        from .tracking import BallKalmanTracker

        self._player_tracker.reset()
        ball_tracker = BallKalmanTracker(self.config.ball.kalman)
        observations: list[_FrameObservation] = []

        with VideoReader(
            video_path,
            stride=self.config.video.frame_stride,
            max_frames=self.config.video.max_frames,
        ) as reader:
            metadata = reader.metadata
            effective_fps = reader.effective_fps

            for frame_index, timestamp, frame in reader:
                players = self._player_tracker.detect(frame)

                if self.enable_stroke and self._pose_estimator is not None:
                    for player in players:
                        player.pose = self._pose_estimator.estimate(frame, player.bbox)

                ball = None
                if self._ball_detector is not None:
                    measurement, confidence = self._ball_detector.detect(frame)
                    ball = ball_tracker.update(measurement, confidence)

                homography, keypoints = self._update_court(frame)

                observations.append(
                    _FrameObservation(
                        frame_index=frame_index,
                        timestamp=timestamp,
                        players=players,
                        ball=ball,
                        homography=homography,
                        court_keypoints=keypoints,
                    )
                )

                if len(observations) % 100 == 0:
                    logger.info("analysed %d frames", len(observations))

        logger.info("analysis pass complete: %d frames", len(observations))
        return observations, metadata, effective_fps

    def _update_court(self, frame: np.ndarray):
        """Refit the court homography if due, and return the current one."""
        if self._court_detector is None or self._homography_tracker is None:
            return None, None

        tracker = self._homography_tracker
        if tracker.should_refit():
            keypoints, confidences = self._court_detector.detect(frame)
            tracker.offer(keypoints, confidences)
        tracker.tick()
        return tracker.homography, tracker.keypoints

    # --- post-processing ----------------------------------------------------------

    def _post_process(
        self,
        observations: list[_FrameObservation],
        metadata: VideoMetadata,
        effective_fps: float,
    ) -> PipelineResult:
        """Select players, project to court space and compute all derived quantities."""
        frame_count = len(observations)
        player_ids = self._select_players(observations)
        if not player_ids:
            self._warn("no players could be selected from the tracked detections")

        player_positions: dict[int, list[Position]] = {pid: [] for pid in player_ids}
        ball_positions: list[Position] = []

        for observation in observations:
            by_id = {p.track_id: p for p in observation.players}
            for player_id in player_ids:
                player = by_id.get(player_id)
                position = None
                if player is not None:
                    position = _project(observation.homography, player.bbox.feet)
                    player.court_xy = position
                player_positions[player_id].append(position)

            ball_position = None
            if observation.ball is not None:
                ball_position = _project(observation.homography, observation.ball.position)
                observation.ball.court_xy = ball_position
            ball_positions.append(ball_position)

        analytics = self.config.analytics
        has_homography = any(o.homography is not None for o in observations)

        player_speeds: dict[int, list[float | None]] = {}
        player_distances: dict[int, list[float]] = {}
        for player_id in player_ids:
            positions = player_positions[player_id]
            if has_homography:
                player_speeds[player_id] = compute_speed_series(
                    positions, effective_fps, analytics.speed_window, analytics.max_player_speed_kmh
                )
                player_distances[player_id] = cumulative_distance(
                    positions, effective_fps, analytics.max_player_speed_kmh
                )
            else:
                player_speeds[player_id] = [None] * frame_count
                player_distances[player_id] = [0.0] * frame_count

        if has_homography:
            ball_speeds = compute_speed_series(
                ball_positions, effective_fps, analytics.speed_window, analytics.max_ball_speed_kmh
            )
        else:
            ball_speeds = [None] * frame_count

        shots = self._detect_and_classify_shots(
            observations, player_ids, player_positions, player_speeds, ball_speeds, effective_fps
        )

        return PipelineResult(
            metadata=metadata,
            effective_fps=effective_fps,
            frame_indices=[o.frame_index for o in observations],
            timestamps=[o.timestamp for o in observations],
            player_ids=player_ids,
            player_positions=player_positions,
            player_speeds=player_speeds,
            player_distances=player_distances,
            ball_positions=ball_positions,
            ball_speeds=ball_speeds,
            shots=shots,
            summary=summarise_players(player_distances, player_speeds),
            warnings=self.warnings,
        )

    def _select_players(self, observations: list[_FrameObservation]) -> list[int]:
        """Pick the two tracks that behave like the players on court.

        With a homography, tracks are scored by how many frames they spend inside the
        court boundary — ball kids stand behind the baseline, the umpire sits to the
        side, and spectators are further out still. Without a homography, the fallback
        is track persistence combined with box size, since the players are the people
        most consistently present and closest to the camera's subject.
        """
        config = self.config.players
        frame_count = len(observations)
        has_homography = any(o.homography is not None for o in observations)

        appearances: dict[int, int] = {}
        scores: dict[int, float] = {}

        for observation in observations:
            for player in observation.players:
                appearances[player.track_id] = appearances.get(player.track_id, 0) + 1
                if has_homography and observation.homography is not None:
                    court_xy = _project(observation.homography, player.bbox.feet)
                    if court_xy is not None and is_inside_court(
                        np.array([court_xy]), margin=config.court_margin_m
                    )[0]:
                        scores[player.track_id] = scores.get(player.track_id, 0.0) + 1.0
                else:
                    # Bigger, more persistent boxes are the players.
                    area = player.bbox.width * player.bbox.height
                    scores[player.track_id] = scores.get(player.track_id, 0.0) + area / 1e4

        min_frames = max(1, int(config.min_track_presence * frame_count))
        eligible = [
            track_id
            for track_id, count in appearances.items()
            if count >= min_frames and scores.get(track_id, 0.0) > 0
        ]
        eligible.sort(key=lambda track_id: scores.get(track_id, 0.0), reverse=True)
        return sorted(eligible[:2])

    def _detect_and_classify_shots(
        self,
        observations: list[_FrameObservation],
        player_ids: list[int],
        player_positions: dict[int, list[Position]],
        player_speeds: dict[int, list[float | None]],
        ball_speeds: list[float | None],
        effective_fps: float,
    ) -> list[Shot]:
        """Find shot moments and, if a classifier is loaded, name the stroke at each.

        Shot *moments* come from the ball trajectory alone, so they are detected even
        without a stroke classifier — they get logged as ``unclassified``. That log is
        what you need to bootstrap a labelled set from your own footage.
        """
        if self._ball_detector is None:
            self._warn("shot detection skipped: ball tracking is unavailable")
            return []

        from .stroke_classification.pose import build_pose_window
        from .stroke_classification.shot_detection import (
            assign_striker,
            detect_shot_moments,
            window_bounds,
        )

        stroke_config = self.config.stroke
        has_homography = any(o.homography is not None for o in observations)

        # Shot moments come from the *pixel* trajectory: it exists even without a
        # homography, and is unaffected by homography dropouts mid-rally.
        pixel_track: list[Position] = [
            None if o.ball is None else o.ball.position for o in observations
        ]
        moments = detect_shot_moments(
            pixel_track,
            effective_fps,
            stroke_config.min_direction_change_deg,
            stroke_config.min_speed_change,
            stroke_config.min_shot_interval_s,
        )
        logger.info("detected %d candidate shot moments", len(moments))

        poses_by_player: dict[int, list[np.ndarray | None]] = {
            player_id: [
                next(
                    (p.pose for p in o.players if p.track_id == player_id),
                    None,
                )
                for o in observations
            ]
            for player_id in player_ids
        }

        shots: list[Shot] = []
        for moment in moments:
            index = moment.index
            observation = observations[index]

            if has_homography:
                ball_reference = observation.ball.court_xy if observation.ball else None
                candidates = {pid: player_positions[pid][index] for pid in player_ids}
                max_distance = MAX_STRIKER_DISTANCE_M
            else:
                ball_reference = observation.ball.position if observation.ball else None
                candidates = {
                    p.track_id: p.bbox.feet
                    for p in observation.players
                    if p.track_id in player_ids
                }
                max_distance = MAX_STRIKER_DISTANCE_PX

            striker = assign_striker(ball_reference, candidates, max_distance)
            if striker is None:
                continue

            stroke, confidence = UNCLASSIFIED_STROKE, 0.0
            if self._stroke_classifier is not None:
                start, end = window_bounds(
                    index, stroke_config.window_seconds, effective_fps, len(observations)
                )
                window = build_pose_window(poses_by_player[striker], start, end)
                if window is None:
                    continue
                stroke, confidence = self._stroke_classifier.predict(window)

            shots.append(
                Shot(
                    frame_index=observation.frame_index,
                    timestamp=observation.timestamp,
                    player_id=striker,
                    stroke=stroke,
                    stroke_confidence=confidence,
                    ball_speed_kmh=ball_speeds[index],
                    player_speed_kmh=player_speeds.get(striker, [None] * len(observations))[index],
                )
            )

        logger.info("classified %d shots", len(shots))
        return shots

    # --- pass 2: rendering --------------------------------------------------------

    def _render_pass(
        self,
        video_path: str | Path,
        observations: list[_FrameObservation],
        result: PipelineResult,
        output_dir: Path,
    ) -> None:
        """Re-read the video and write the annotated output."""
        output_config = self.config.output
        metadata = result.metadata
        fps = self.config.video.output_fps or result.effective_fps

        mini_court = (
            MiniCourt(output_config.mini_court) if output_config.draw_mini_court else None
        )
        # Stroke labels persist for ~0.6s so they are readable at playback speed.
        label_hold = max(1, int(round(0.6 * result.effective_fps)))
        shots_by_index = {
            index: shot
            for shot in result.shots
            for index in [result.frame_indices.index(shot.frame_index)]
        }

        destination = output_dir / output_config.video_filename
        with VideoWriter(destination, fps, metadata.width, metadata.height) as writer:
            with VideoReader(
                video_path,
                stride=self.config.video.frame_stride,
                max_frames=self.config.video.max_frames,
            ) as reader:
                for position, (_, _, frame) in enumerate(reader):
                    if position >= len(observations):
                        break
                    self._annotate(
                        frame, position, observations[position], result, mini_court,
                        shots_by_index, label_hold,
                    )
                    writer.write(frame)

        logger.info("wrote annotated video to %s", destination)

    def _annotate(
        self,
        frame: np.ndarray,
        position: int,
        observation: _FrameObservation,
        result: PipelineResult,
        mini_court: MiniCourt | None,
        shots_by_index: dict[int, Shot],
        label_hold: int,
    ) -> None:
        """Draw all overlays onto one frame, in place."""
        output_config = self.config.output
        player_ids = result.player_ids
        tracked = [p for p in observation.players if p.track_id in player_ids]

        draw_court_keypoints(frame, observation.court_keypoints)

        speeds = None
        distances = None
        if output_config.draw_speeds:
            speeds = {
                pid: value
                for pid in player_ids
                if (value := result.player_speeds[pid][position]) is not None
            }
            distances = {pid: result.player_distances[pid][position] for pid in player_ids}

        draw_players(frame, tracked, player_ids, speeds, distances)
        draw_ball(
            frame,
            observation.ball,
            result.ball_speeds[position] if output_config.draw_speeds else None,
        )

        if output_config.draw_stroke_labels:
            for index, shot in shots_by_index.items():
                if index <= position < index + label_hold:
                    player = next(
                        (p for p in tracked if p.track_id == shot.player_id), None
                    )
                    if player is not None:
                        anchor = (int(player.bbox.x1), int(player.bbox.y1))
                        draw_stroke_label(frame, anchor, shot.stroke, shot.stroke_confidence)

        if mini_court is not None:
            court_players = {
                pid: pos
                for pid in player_ids
                if (pos := result.player_positions[pid][position]) is not None
            }
            render_mini_court(frame, mini_court, court_players, result.ball_positions[position])

        draw_frame_info(frame, observation.frame_index, observation.timestamp)

    # --- outputs ------------------------------------------------------------------

    def _export(self, result: PipelineResult, output_dir: Path) -> None:
        """Write the shot log, track log and player summary."""
        output_config = self.config.output

        export_shots(
            result.shots,
            output_dir / output_config.shots_csv,
            output_dir / output_config.shots_json,
        )
        export_tracks(
            result.frame_indices,
            result.timestamps,
            result.player_positions,
            result.player_speeds,
            result.player_distances,
            result.ball_positions,
            result.ball_speeds,
            output_dir / output_config.tracks_csv,
        )
        result.summary.to_csv(output_dir / "player_summary.csv", index=False)
        logger.info("wrote analytics to %s", output_dir)


def _project(
    homography: CourtHomography | None, point: tuple[float, float]
) -> Position:
    """Project a pixel point into court metres, returning ``None`` if implausible.

    Points far outside the court are almost always bad projections near the horizon,
    where small pixel errors blow up into huge court-space errors.
    """
    if homography is None:
        return None
    court = homography.to_court(np.array(point))
    if not np.isfinite(court).all():
        return None
    if abs(court[0]) > 3 * COURT_LENGTH or abs(court[1]) > 3 * COURT_LENGTH:
        return None
    return (float(court[0]), float(court[1]))
