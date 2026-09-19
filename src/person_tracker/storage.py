"""Portable storage for tracking, face, and logical-identity observations."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .face import FaceSample


OBSERVATION_SCHEMA_VERSION = 1
LATEST_RUNS_FILENAME = "latest.json"
LATEST_RUN_KEYS = {
    "scene": "latest_scene_run",
    "observation": "latest_observation_run",
    "resolved": "latest_resolved_run",
}


def load_latest_runs(runs_directory: str | Path) -> dict[str, Any]:
    """Load the portable latest-run registry, returning an empty registry if absent."""
    path = Path(runs_directory).expanduser().resolve() / LATEST_RUNS_FILENAME
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Latest-run registry must contain a JSON object: {path}")
    return payload


def update_latest_run(
    runs_directory: str | Path,
    run_directory: str | Path,
    *,
    stage: str,
) -> Path:
    """Atomically update the observation or resolved pointer in ``runs/latest.json``."""
    if stage not in LATEST_RUN_KEYS:
        raise ValueError(f"Unknown run stage {stage!r}; expected one of {sorted(LATEST_RUN_KEYS)}")

    runs_dir = Path(runs_directory).expanduser().resolve()
    run_dir = Path(run_directory).expanduser().resolve()
    try:
        relative_run = run_dir.relative_to(runs_dir)
    except ValueError as error:
        raise ValueError(f"Run directory must be inside {runs_dir}: {run_dir}") from error
    if relative_run == Path("."):
        raise ValueError("Run directory cannot be the runs directory itself")

    runs_dir.mkdir(parents=True, exist_ok=True)
    registry = load_latest_runs(runs_dir)
    registry[LATEST_RUN_KEYS[stage]] = relative_run.as_posix()
    registry["updated_at"] = datetime.now(timezone.utc).isoformat()

    path = runs_dir / LATEST_RUNS_FILENAME
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    _write_json(temporary_path, registry)
    temporary_path.replace(path)
    return path


def resolve_run_directory(
    runs_directory: str | Path,
    *,
    stage: str,
    explicit: str | Path | None = None,
    environment_variable: str = "PERSON_TRACKER_RUN_DIRECTORY",
) -> Path:
    """Resolve an explicit, environment-provided, or latest compatible run."""
    if stage not in LATEST_RUN_KEYS:
        raise ValueError(f"Unknown run stage {stage!r}; expected one of {sorted(LATEST_RUN_KEYS)}")

    runs_dir = Path(runs_directory).expanduser().resolve()
    candidate = explicit
    source = "explicit override"
    if candidate is None:
        candidate = os.environ.get(environment_variable)
        source = f"environment variable {environment_variable}"
    if candidate is None:
        registry = load_latest_runs(runs_dir)
        candidate = registry.get(LATEST_RUN_KEYS[stage])
        source = f"{runs_dir / LATEST_RUNS_FILENAME} ({LATEST_RUN_KEYS[stage]})"
    if candidate is None:
        raise FileNotFoundError(
            f"No {stage} run was provided and no latest pointer exists in "
            f"{runs_dir / LATEST_RUNS_FILENAME}"
        )

    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = runs_dir / path
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Run directory from {source} does not exist: {path}")
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def build_run_manifest(
    *,
    video_path: str | Path,
    width: int,
    height: int,
    fps: float,
    frame_count: int,
    start_frame: int,
    end_frame: int,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the metadata required to interpret a saved observation run."""
    source = Path(video_path).expanduser().resolve()
    stat = source.stat()
    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_video": str(source),
        "source_size_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "width": int(width),
        "height": int(height),
        "fps": float(fps),
        "frame_count": int(frame_count),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "start_seconds": float(start_frame / fps),
        "end_seconds": float(end_frame / fps),
        "settings": dict(settings),
    }


def save_observation_run(
    run_directory: str | Path,
    *,
    manifest: Mapping[str, Any],
    tracking_history: Mapping[int, Sequence[Mapping[str, Any]]],
    face_samples: Sequence[FaceSample],
    save_face_crops: bool = True,
) -> Path:
    """Save tracker observations and face evidence without Python pickles."""
    run_dir = Path(run_directory).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "manifest.json", dict(manifest))

    with (run_dir / "tracks.jsonl").open("w", encoding="utf-8") as stream:
        for frame_no in sorted(tracking_history):
            record = {
                "frame": int(frame_no),
                "tracks": [
                    {
                        **dict(track),
                        "track_id": int(track["track_id"]),
                        "bbox": np.asarray(track["bbox"], dtype=int).tolist(),
                        "confidence": float(track["confidence"]),
                    }
                    for track in tracking_history[frame_no]
                ],
            }
            stream.write(json.dumps(record, default=_json_default) + "\n")

    embeddings = (
        np.stack([np.asarray(sample.embedding, dtype=np.float32) for sample in face_samples])
        if face_samples
        else np.empty((0, 0), dtype=np.float32)
    )
    np.save(run_dir / "face_embeddings.npy", embeddings, allow_pickle=False)

    face_dir = run_dir / "faces"
    if save_face_crops:
        face_dir.mkdir(exist_ok=True)

    with (run_dir / "face_samples.jsonl").open("w", encoding="utf-8") as stream:
        for index, sample in enumerate(face_samples):
            image_path = None
            if save_face_crops and sample.image.size:
                image_path = f"faces/{index:08d}.jpg"
                if not cv2.imwrite(str(run_dir / image_path), sample.image):
                    raise RuntimeError(f"Could not save face crop {image_path}")
            record = {
                "embedding_index": index,
                "frame": int(sample.frame_no),
                "track_id": int(sample.track_id),
                "bbox": [int(value) for value in sample.bbox],
                "confidence": float(sample.confidence),
                "quality": float(sample.quality),
                "image": image_path,
            }
            stream.write(json.dumps(record) + "\n")

    return run_dir


def load_observation_run(
    run_directory: str | Path,
    *,
    load_face_crops: bool = True,
) -> tuple[dict[str, Any], dict[int, list[dict[str, Any]]], list[FaceSample]]:
    """Load a saved observation run and reconstruct ``FaceSample`` objects."""
    run_dir = Path(run_directory).expanduser().resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != OBSERVATION_SCHEMA_VERSION:
        raise ValueError(f"Unsupported observation schema: {manifest.get('schema_version')}")

    tracking_history: dict[int, list[dict[str, Any]]] = {}
    with (run_dir / "tracks.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            tracking_history[int(record["frame"])] = [
                {
                    **track,
                    "track_id": int(track["track_id"]),
                    "bbox": np.asarray(track["bbox"], dtype=int),
                    "confidence": float(track["confidence"]),
                }
                for track in record["tracks"]
            ]

    embeddings = np.load(run_dir / "face_embeddings.npy", allow_pickle=False)
    face_samples: list[FaceSample] = []
    with (run_dir / "face_samples.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            image = np.empty((0, 0, 3), dtype=np.uint8)
            if load_face_crops and record.get("image"):
                loaded = cv2.imread(str(run_dir / record["image"]))
                if loaded is not None:
                    image = loaded
            face_samples.append(
                FaceSample(
                    frame_no=int(record["frame"]),
                    track_id=int(record["track_id"]),
                    bbox=tuple(int(value) for value in record["bbox"]),
                    confidence=float(record["confidence"]),
                    quality=float(record["quality"]),
                    embedding=np.asarray(
                        embeddings[int(record["embedding_index"])],
                        dtype=np.float32,
                    ),
                    image=image,
                )
            )
    return manifest, tracking_history, face_samples


def save_identity_resolution(
    run_directory: str | Path,
    *,
    identity_history: Mapping[int, Mapping[int, Mapping[str, Any]]],
    switch_boundaries: Mapping[int, Sequence[Mapping[str, Any]]],
    summary: Mapping[str, Any],
) -> None:
    """Save resolved logical-person assignments alongside an observation run."""
    run_dir = Path(run_directory).expanduser().resolve()
    with (run_dir / "identities.jsonl").open("w", encoding="utf-8") as stream:
        for frame_no in sorted(identity_history):
            for track_id, identity in sorted(identity_history[frame_no].items()):
                record = {
                    "frame": int(frame_no),
                    "track_id": int(track_id),
                    **dict(identity),
                }
                stream.write(json.dumps(record, default=_json_default) + "\n")
    _write_json(run_dir / "identity_events.json", dict(switch_boundaries))
    _write_json(run_dir / "identity_summary.json", dict(summary))
