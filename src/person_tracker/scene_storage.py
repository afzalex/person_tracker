"""Portable, validated storage for video scene detection runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .core import VideoInfo


SCENE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SceneRecord:
    """One half-open scene range in source-video frame coordinates."""

    scene_id: str
    scene_index: int
    start_frame: int
    end_frame: int
    fps: float
    detection_score: float | None = None
    boundary_source: str = "automatic"
    enabled: bool = True

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def start_seconds(self) -> float:
        return self.start_frame / self.fps

    @property
    def end_seconds(self) -> float:
        return self.end_frame / self.fps

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update({
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "duration_seconds": self.duration_seconds,
            "frame_count": self.frame_count,
        })
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SceneRecord":
        return cls(
            scene_id=str(payload["scene_id"]),
            scene_index=int(payload["scene_index"]),
            start_frame=int(payload["start_frame"]),
            end_frame=int(payload["end_frame"]),
            fps=float(payload["fps"]),
            detection_score=(
                None if payload.get("detection_score") is None
                else float(payload["detection_score"])
            ),
            boundary_source=str(payload.get("boundary_source", "automatic")),
            enabled=bool(payload.get("enabled", True)),
        )


def build_scene_manifest(
    video_info: VideoInfo,
    *,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a source fingerprint and settings manifest for a scene run."""
    source = Path(video_info.path).expanduser().resolve()
    stat = source.stat()
    return {
        "schema_version": SCENE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_video": str(source),
        "source_size_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "width": int(video_info.width),
        "height": int(video_info.height),
        "fps": float(video_info.fps),
        "frame_count": int(video_info.frame_count),
        "duration_seconds": float(video_info.duration_seconds),
        "settings": dict(settings),
    }


def validate_scenes(
    scenes: Sequence[SceneRecord],
    *,
    frame_count: int,
    require_complete_coverage: bool = True,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> None:
    """Validate IDs, ordering, bounds, and contiguous window coverage."""
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    window_end = frame_count if end_frame is None else int(end_frame)
    if not 0 <= start_frame < window_end <= frame_count:
        raise ValueError(
            f"Invalid scene window {start_frame}:{window_end} for {frame_count} frames"
        )
    if not scenes:
        raise ValueError("At least one scene is required")

    seen_ids: set[str] = set()
    previous_end = start_frame
    for expected_index, scene in enumerate(scenes):
        if scene.scene_id in seen_ids:
            raise ValueError(f"Duplicate scene ID: {scene.scene_id}")
        seen_ids.add(scene.scene_id)
        if scene.scene_index != expected_index:
            raise ValueError(
                f"Scene {scene.scene_id} has index {scene.scene_index}; "
                f"expected {expected_index}"
            )
        if not 0 <= scene.start_frame < scene.end_frame <= frame_count:
            raise ValueError(
                f"Scene {scene.scene_id} has invalid range "
                f"{scene.start_frame}:{scene.end_frame} for {frame_count} frames"
            )
        if require_complete_coverage and scene.start_frame != previous_end:
            raise ValueError(
                f"Scene {scene.scene_id} starts at {scene.start_frame}; "
                f"expected contiguous start {previous_end}"
            )
        previous_end = scene.end_frame

    if require_complete_coverage and previous_end != window_end:
        raise ValueError(
            f"Scenes end at frame {previous_end}; expected window end {window_end}"
        )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, scenes: Sequence[SceneRecord]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for scene in scenes:
            stream.write(json.dumps(scene.to_dict()) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[SceneRecord]:
    scenes: list[SceneRecord] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                scenes.append(SceneRecord.from_dict(json.loads(line)))
    return scenes


def save_scene_run(
    run_directory: str | Path,
    *,
    manifest: Mapping[str, Any],
    raw_scenes: Sequence[SceneRecord],
    scenes: Sequence[SceneRecord],
) -> Path:
    """Atomically save a scene manifest plus raw and accepted scene lists."""
    run_dir = Path(run_directory).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    frame_count = int(manifest["frame_count"])
    start_frame = int(manifest.get("window_start_frame", 0))
    end_frame = int(manifest.get("window_end_frame", frame_count))
    validate_scenes(
        raw_scenes, frame_count=frame_count,
        start_frame=start_frame, end_frame=end_frame,
    )
    validate_scenes(
        scenes, frame_count=frame_count,
        start_frame=start_frame, end_frame=end_frame,
    )

    payload = dict(manifest)
    payload["schema_version"] = SCENE_SCHEMA_VERSION
    payload["raw_scene_count"] = len(raw_scenes)
    payload["scene_count"] = len(scenes)
    payload["enabled_scene_count"] = sum(scene.enabled for scene in scenes)
    _write_json(run_dir / "manifest.json", payload)
    _write_jsonl(run_dir / "raw_scenes.jsonl", raw_scenes)
    _write_jsonl(run_dir / "scenes.jsonl", scenes)
    return run_dir


def load_scene_run(
    run_directory: str | Path,
) -> tuple[dict[str, Any], list[SceneRecord], list[SceneRecord]]:
    """Load and validate a previously saved scene detection run."""
    run_dir = Path(run_directory).expanduser().resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCENE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported scene schema: {manifest.get('schema_version')}")
    raw_scenes = _read_jsonl(run_dir / "raw_scenes.jsonl")
    scenes = _read_jsonl(run_dir / "scenes.jsonl")
    frame_count = int(manifest["frame_count"])
    start_frame = int(manifest.get("window_start_frame", 0))
    end_frame = int(manifest.get("window_end_frame", frame_count))
    validate_scenes(
        raw_scenes, frame_count=frame_count,
        start_frame=start_frame, end_frame=end_frame,
    )
    validate_scenes(
        scenes, frame_count=frame_count,
        start_frame=start_frame, end_frame=end_frame,
    )
    return manifest, raw_scenes, scenes
