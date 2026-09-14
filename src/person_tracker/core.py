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
class ProcessingRange:
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
    def duration_seconds(self) -> float:
        return (self.end_frame - self.start_frame) / self.fps

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame