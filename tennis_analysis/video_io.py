"""Stage 1 — video ingestion, frame extraction and annotated-video writing."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoMetadata:
    """Source video properties needed for real-world timing."""

    path: Path
    fps: float
    width: int
    height: int
    frame_count: int

    @property
    def duration_seconds(self) -> float:
        """Video duration in seconds (0.0 if the frame count is unknown)."""
        return self.frame_count / self.fps if self.fps > 0 else 0.0


class VideoReader:
    """Iterates frames from a video file while preserving FPS metadata.

    Supports striding (process every Nth frame) and a frame cap. Timestamps are always
    computed against the *source* FPS, so speeds stay correct in real-world units
    regardless of the stride.

    Example:
        >>> with VideoReader("match.mp4", stride=2) as reader:  # doctest: +SKIP
        ...     for index, timestamp, frame in reader:
        ...         ...
    """

    def __init__(
        self,
        path: str | Path,
        stride: int = 1,
        max_frames: int | None = None,
    ) -> None:
        """Open a video for reading.

        Args:
            path: Path to the input video.
            stride: Yield every ``stride``-th frame. Must be >= 1.
            max_frames: Stop after yielding this many frames. ``None`` reads to the end.

        Raises:
            FileNotFoundError: If the video file does not exist.
            ValueError: If ``stride`` < 1, or OpenCV cannot open the file.
        """
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"video not found: {self.path}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")

        self.stride = stride
        self.max_frames = max_frames
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise ValueError(f"could not open video (unsupported codec?): {self.path}")

        fps = self._capture.get(cv2.CAP_PROP_FPS)
        # Some containers report 0 or NaN; 30 is the safest fallback for match footage.
        if not fps or not np.isfinite(fps) or fps <= 0:
            fps = 30.0

        self.metadata = VideoMetadata(
            path=self.path,
            fps=float(fps),
            width=int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    @property
    def effective_fps(self) -> float:
        """FPS of the *yielded* frame sequence, accounting for the stride."""
        return self.metadata.fps / self.stride

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        """Yield ``(source_frame_index, timestamp_seconds, frame)`` tuples."""
        source_index = 0
        yielded = 0
        while True:
            ok, frame = self._capture.read()
            if not ok:
                break
            if source_index % self.stride == 0:
                yield source_index, source_index / self.metadata.fps, frame
                yielded += 1
                if self.max_frames is not None and yielded >= self.max_frames:
                    break
            source_index += 1

    def release(self) -> None:
        """Release the underlying capture handle."""
        self._capture.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


class VideoWriter:
    """Writes annotated frames to an MP4 file."""

    def __init__(
        self,
        path: str | Path,
        fps: float,
        width: int,
        height: int,
    ) -> None:
        """Open a video for writing.

        Args:
            path: Output path. Parent directories are created if needed.
            fps: Playback frame rate.
            width: Frame width in pixels.
            height: Frame height in pixels.

        Raises:
            ValueError: If OpenCV cannot open the output writer.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self.path), fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise ValueError(f"could not open video writer for {self.path}")
        self.width = width
        self.height = height

    def write(self, frame: np.ndarray) -> None:
        """Append a frame, resizing it if it does not match the writer's dimensions."""
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        self._writer.write(frame)

    def release(self) -> None:
        """Finalise and close the output file."""
        self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()


def probe(path: str | Path) -> VideoMetadata:
    """Read a video's metadata without iterating its frames.

    Args:
        path: Path to the video file.

    Returns:
        The video's :class:`VideoMetadata`.
    """
    with VideoReader(path) as reader:
        return reader.metadata
