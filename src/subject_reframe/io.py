"""File I/O, caching, and reporting operations."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .core import CropConfig, Scene, VideoInfo


def inspect_video(video_path: str | Path) -> VideoInfo:
    """Read basic metadata and fail early if the input cannot be decoded."""
    path = Path(video_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input video does not exist: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open: {path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
        raise RuntimeError(
            "Video metadata is invalid: "
            f"{width=} {height=} {fps=} {frame_count=}"
        )

    return VideoInfo(
        path=str(path),
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_seconds=frame_count / fps,
    )


def read_frame(video_path: str | Path, frame_index: int) -> np.ndarray:
    """Read one BGR frame by zero-based frame index."""
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_index}")
    return frame


def build_tracking_cache_metadata(
    info: VideoInfo,
    scenes: Sequence[Scene],
    *,
    model_name: str,
    tracker_name: str,
    quantize: int | str | None,
    image_size: int,
    frame_stride: int,
    confidence: float,
    iou: float,
    max_detections: int = 300,
) -> dict[str, Any]:
    """Describe every input that can change cached tracking results."""
    source = Path(info.path)
    stat = source.stat()
    return {
        "schema_version": 2,
        "source": {
            "path": str(source),
            "size_bytes": stat.st_size,
            "modified_ns": stat.st_mtime_ns,
            "width": info.width,
            "height": info.height,
            "fps": info.fps,
            "frame_count": info.frame_count,
        },
        "scenes": [[scene.start_frame, scene.end_frame] for scene in scenes],
        "tracking": {
            "model_name": model_name,
            "tracker_name": tracker_name,
            "quantize": quantize,
            "image_size": image_size,
            "frame_stride": frame_stride,
            "confidence": confidence,
            "iou": iou,
            "max_detections": max_detections,
        },
    }


def save_tracking_records(
    records: Mapping[int, Sequence[Mapping[str, Any]]],
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Persist tracking records with metadata for cache validation."""
    serializable = {str(key): list(value) for key, value in records.items()}
    envelope = {
        "metadata": dict(metadata or {}),
        "records": serializable,
    }
    Path(path).write_text(json.dumps(envelope), encoding="utf-8")


def load_tracking_records(
    path: str | Path,
    *,
    expected_metadata: Mapping[str, Any] | None = None,
) -> dict[int, list[dict[str, Any]]]:
    """Load tracking records and validate metadata matches current configuration."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if "records" not in raw or "metadata" not in raw:
        raise ValueError("Tracking cache uses the old schema and must be regenerated")
    if expected_metadata is not None and raw["metadata"] != dict(expected_metadata):
        raise ValueError("Tracking cache configuration does not match the current run")
    return {int(key): value for key, value in raw["records"].items()}


def save_scene_selections(
    selections: Mapping[int, int | Sequence[int] | str | None],
    path: str | Path,
    *,
    metadata: Mapping[str, Any],
) -> None:
    """Persist confirmed scene selections so an interrupted notebook can resume."""
    serializable: dict[str, int | list[int] | str | None] = {}
    for scene_index, selection in selections.items():
        if selection is None or isinstance(selection, (int, str)):
            value = selection
        else:
            value = [int(track_id) for track_id in selection]
        serializable[str(scene_index)] = value
    envelope = {"metadata": dict(metadata), "selections": serializable}
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(envelope, indent=2), encoding="utf-8")


def load_scene_selections(
    path: str | Path,
    *,
    expected_metadata: Mapping[str, Any],
) -> dict[int, int | list[int] | str | None]:
    """Load selections only when they belong to the current input and settings."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if "selections" not in raw or "metadata" not in raw:
        raise ValueError("Scene selections use an unsupported schema")
    if raw["metadata"] != dict(expected_metadata):
        raise ValueError("Saved scene selections do not match the current run")
    return {int(scene_index): value for scene_index, value in raw["selections"].items()}


def source_segments_from_plans(
    plans: Sequence[Mapping[str, Any]], fps: float
) -> list[tuple[float, float]]:
    """Convert retained source frames into contiguous audio time segments."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not plans:
        return []
    frames = [int(plan["frame"]) for plan in plans]
    segments: list[tuple[float, float]] = []
    start = previous = frames[0]
    for frame in frames[1:]:
        if frame != previous + 1:
            segments.append((start / fps, (previous + 1 - start) / fps))
            start = frame
        previous = frame
    segments.append((start / fps, (previous + 1 - start) / fps))
    return segments


def ffmpeg_available() -> bool:
    """Check if FFmpeg is available on the system PATH."""
    return shutil.which("ffmpeg") is not None


def mux_original_audio(
    silent_video_path: str | Path,
    source_video_path: str | Path,
    output_path: str | Path,
    *,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
    copy_video: bool = False,
    source_segments: Sequence[tuple[float, float]] | None = None,
) -> Path:
    """Attach matching audio and encode browser-compatible H.264 video."""
    if not ffmpeg_available():
        raise RuntimeError("FFmpeg was not found on PATH")
    if start_seconds < 0:
        raise ValueError("start_seconds cannot be negative")
    if duration_seconds is not None and duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")

    normalized_segments = None
    if source_segments is not None:
        normalized_segments = [
            (float(segment_start), float(segment_duration))
            for segment_start, segment_duration in source_segments
        ]
        if not normalized_segments:
            raise ValueError("source_segments cannot be empty")
        if any(start < 0 or duration <= 0 for start, duration in normalized_segments):
            raise ValueError("Every source audio segment needs start >= 0 and duration > 0")
        if len(normalized_segments) == 1:
            start_seconds, duration_seconds = normalized_segments[0]
            normalized_segments = None

    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    source_has_audio = True
    if normalized_segments is not None:
        audio_probe = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=index",
                "-of",
                "csv=p=0",
                str(source_video_path),
            ],
            capture_output=True,
            text=True,
        )
        source_has_audio = audio_probe.returncode == 0 and bool(audio_probe.stdout.strip())

    def build_command(video_options: Sequence[str]) -> list[str]:
        command = ["ffmpeg", "-y", "-i", str(silent_video_path)]
        if normalized_segments is None:
            if start_seconds > 0:
                command.extend(["-ss", f"{start_seconds:.9f}"])
            if duration_seconds is not None:
                command.extend(["-t", f"{duration_seconds:.9f}"])
        command.extend(["-i", str(source_video_path)])

        audio_output_options: list[str] = []
        if normalized_segments is not None and source_has_audio:
            source_labels = [f"[src{index}]" for index in range(len(normalized_segments))]
            filter_steps = [
                f"[1:a:0]asplit={len(source_labels)}{''.join(source_labels)}"
            ]
            labels = []
            for index, (segment_start, segment_duration) in enumerate(
                normalized_segments
            ):
                segment_end = segment_start + segment_duration
                label = f"a{index}"
                labels.append(f"[{label}]")
                filter_steps.append(
                    f"[src{index}]atrim=start={segment_start:.9f}:end={segment_end:.9f},"
                    f"asetpts=PTS-STARTPTS[{label}]"
                )
            filter_steps.append(
                "".join(labels)
                + f"concat=n={len(labels)}:v=0:a=1[aout]"
            )
            command.extend(["-filter_complex", ";".join(filter_steps)])
            audio_output_options = ["-map", "[aout]", "-c:a", "aac", "-b:a", "192k"]
        elif normalized_segments is None:
            audio_output_options = ["-map", "1:a?", "-c:a", "aac", "-b:a", "192k"]

        command.extend([
            "-map",
            "0:v:0",
            *audio_output_options,
            *video_options,
            *(
                []
                if video_options == ["-c:v", "copy"]
                else ["-pix_fmt", "yuv420p"]
            ),
            "-shortest",
            "-movflags",
            "+faststart",
            str(output),
        ])
        return command

    # Direct rendering already produces H.264, so the normal notebook path
    # copies video without a second encoding pass.
    attempts = (
        [["-c:v", "copy"]]
        if copy_video
        else [
            ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "21", "-b:v", "0"],
            ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21"],
        ]
    )
    completed = None
    for video_options in attempts:
        completed = subprocess.run(
            build_command(video_options), capture_output=True, text=True
        )
        if completed.returncode == 0:
            break
    if completed is None or completed.returncode != 0:
        error = "" if completed is None else completed.stderr[-4000:]
        raise RuntimeError(f"FFmpeg H.264 encoding failed:\n{error}")
    return output


def create_browser_preview(
    video_path: str | Path,
    output_path: str | Path,
    *,
    width: int = 360,
) -> Path:
    """Create a small H.264 preview suitable for embedding in a notebook."""
    if not ffmpeg_available():
        raise RuntimeError("FFmpeg was not found on PATH")
    if width < 120:
        raise ValueError("Browser preview width must be at least 120 pixels")
    if width % 2:
        width -= 1

    source = Path(video_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-vf",
        f"scale={width}:-2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "27",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-movflags",
        "+faststart",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"FFmpeg could not create the browser preview:\n{completed.stderr[-4000:]}"
        )
    return output


def save_report(
    output_directory: str | Path,
    *,
    info: VideoInfo,
    scenes: Sequence[Scene],
    config: CropConfig,
    warnings: Sequence[Mapping[str, Any]],
    tracking_metadata: Mapping[str, Any] | None = None,
    processing_range: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Save processing report (JSON) and warnings (CSV)."""
    directory = Path(output_directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "processing_report.json"
    csv_path = directory / "warnings.csv"

    report = {
        "video": asdict(info),
        "processing_range": dict(processing_range or {}),
        "config": asdict(config),
        "tracking": dict(tracking_metadata or {}),
        "scenes": [asdict(scene) for scene in scenes],
        "warnings": list(warnings),
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    fieldnames = [
        "type",
        "scene",
        "start_seconds",
        "end_seconds",
        "count",
        "details",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for warning in warnings:
            writer.writerow({key: warning.get(key, "") for key in fieldnames})

    return json_path, csv_path
