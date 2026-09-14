"""Core data structures for the subject reframe pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


@dataclass(frozen=True)
class Scene:
    index: int
    start_frame: int
    end_frame: int  # exclusive
    fps: float

    @property
    def start_seconds(self) -> float:
        return self.start_frame / self.fps

    @property
    def end_seconds(self) -> float:
        return self.end_frame / self.fps

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame


@dataclass(frozen=True)
class CropConfig:
    output_width: int = 1080
    output_height: int = 1920
    horizontal_margin: float = 0.15
    top_margin: float = 0.10
    bottom_margin: float = 0.12
    smoothing_seconds: float = 0.45
    max_interpolation_seconds: float = 0.60
    low_confidence_threshold: float = 0.35
    padding_mode: str = "blur"  # blur or black
    absent_scene_policy: str = "black"

    @property
    def aspect_ratio(self) -> float:
        return self.output_width / self.output_height
