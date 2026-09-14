"""Video rendering and frame extraction."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import cv2
import numpy as np

from .core import CropConfig, VideoInfo
from .io import read_frame

try:
    from tqdm.auto import tqdm
except ImportError:
    class _PlainProgress:
        def __init__(self, iterable=None, **_: Any) -> None:
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable if self.iterable is not None else ())

        def update(self, _: int = 1) -> None:
            pass

        def close(self) -> None:
            pass

    def tqdm(iterable=None, **kwargs: Any):
        return _PlainProgress(iterable, **kwargs)


def _extract_crop(
    frame: np.ndarray,
    crop: Sequence[float],
    *,
    padding_mode: str,
) -> np.ndarray:
    """Extract and pad/blur a crop rectangle from a frame."""
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = crop
    ix1, iy1 = math.floor(x1), math.floor(y1)
    ix2, iy2 = math.ceil(x2), math.ceil(y2)
    crop_width = max(1, ix2 - ix1)
    crop_height = max(1, iy2 - iy1)

    source_x1 = max(0, ix1)
    source_y1 = max(0, iy1)
    source_x2 = min(frame_width, ix2)
    source_y2 = min(frame_height, iy2)
    destination_x1 = source_x1 - ix1
    destination_y1 = source_y1 - iy1

    if padding_mode == "blur":
        canvas = cv2.resize(frame, (crop_width, crop_height), interpolation=cv2.INTER_AREA)
        sigma = max(9, int(min(crop_width, crop_height) * 0.025))
        if sigma % 2 == 0:
            sigma += 1
        canvas = cv2.GaussianBlur(canvas, (sigma, sigma), 0)
    elif padding_mode == "black":
        canvas = np.zeros((crop_height, crop_width, 3), dtype=np.uint8)
    else:
        raise ValueError("padding_mode must be 'blur' or 'black'")

    if source_x2 > source_x1 and source_y2 > source_y1:
        roi = frame[source_y1:source_y2, source_x1:source_x2]
        roi_height, roi_width = roi.shape[:2]
        canvas[
            destination_y1 : destination_y1 + roi_height,
            destination_x1 : destination_x1 + roi_width,
        ] = roi
    return canvas


def render_video(
    video_path: str | Path,
    output_path: str | Path,
    plans: Sequence[Mapping[str, Any]],
    info: VideoInfo,
    config: CropConfig,
    *,
    progress: bool = True,
    checkpoint_dir: str | Path | None = None,
    checkpoint_frames: int = 30,
) -> Path:
    """Render a silent fixed-size MP4 from the reviewed crop plan.

    When ``checkpoint_dir`` is set, the video is split into resumable segments.
    Completed segments are saved under the checkpoint directory and reused on the
    next run so a partially rendered output can resume without reprocessing the
    finished frames.
    """
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_count = min(info.frame_count, len(plans))

    if checkpoint_dir is not None:
        checkpoint_root = Path(checkpoint_dir).expanduser().resolve()
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        manifest_path = checkpoint_root / "render_manifest.json"
        manifest = {"segments": []}
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                manifest = {"segments": []}

        checkpointed = set()
        for segment in manifest.get("segments", []):
            if isinstance(segment, dict):
                start = int(segment.get("start", -1))
                end = int(segment.get("end", -1))
                if start >= 0 and end > start:
                    checkpointed.add((start, end))

        segment_size = max(1, int(checkpoint_frames))
        for start in range(0, frame_count, segment_size):
            end = min(frame_count, start + segment_size)
            segment_key = (start, end)
            segment_path = checkpoint_root / f"segment_{start:06d}.mp4"
            if segment_path.exists() and segment_key in checkpointed:
                continue

            segment_writer = cv2.VideoWriter(
                str(segment_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                info.fps,
                (config.output_width, config.output_height),
            )
            if not segment_writer.isOpened():
                raise RuntimeError(f"Could not create checkpoint video: {segment_path}")

            cap = cv2.VideoCapture(str(video_path))
            for frame_index in range(start, end):
                ok, frame = cap.read()
                if not ok:
                    segment_writer.release()
                    cap.release()
                    raise RuntimeError(
                        f"Could not read frame {frame_index} while rendering checkpoint"
                    )
                if frame_index > 0:
                    cap.release()
                    cap = cv2.VideoCapture(str(video_path))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ok, frame = cap.read()
                    if not ok:
                        segment_writer.release()
                        cap.release()
                        raise RuntimeError(
                            f"Could not read frame {frame_index} while preparing checkpoint"
                        )

                plan = plans[frame_index]
                if plan["mode"] == "crop":
                    crop = _extract_crop(
                        frame, plan["crop"], padding_mode=config.padding_mode
                    )
                    interpolation = (
                        cv2.INTER_AREA
                        if crop.shape[1] >= config.output_width
                        and crop.shape[0] >= config.output_height
                        else cv2.INTER_LANCZOS4
                    )
                    rendered = cv2.resize(
                        crop,
                        (config.output_width, config.output_height),
                        interpolation=interpolation,
                    )
                else:
                    rendered = np.zeros(
                        (config.output_height, config.output_width, 3), dtype=np.uint8
                    )
                segment_writer.write(rendered)
            segment_writer.release()
            cap.release()
            checkpointed.add(segment_key)
            manifest.setdefault("segments", []).append({
                "start": start,
                "end": end,
                "path": str(segment_path),
            })
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            info.fps,
            (config.output_width, config.output_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video: {output}")

        for start in range(0, frame_count, segment_size):
            end = min(frame_count, start + segment_size)
            segment_path = checkpoint_root / f"segment_{start:06d}.mp4"
            segment_cap = cv2.VideoCapture(str(segment_path))
            if not segment_cap.isOpened():
                segment_cap.release()
                raise RuntimeError(f"Could not read checkpoint segment: {segment_path}")
            while True:
                ok, chunk_frame = segment_cap.read()
                if not ok:
                    break
                writer.write(chunk_frame)
            segment_cap.release()
        writer.release()
        return output

    cap = cv2.VideoCapture(str(video_path))
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        info.fps,
        (config.output_width, config.output_height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output}")

    iterator = range(frame_count)
    if progress:
        iterator = tqdm(iterator, desc="Rendering video")

    for frame_index in iterator:
        ok, frame = cap.read()
        if not ok:
            writer.release()
            cap.release()
            raise RuntimeError(f"Could not read frame {frame_index} while rendering")

        plan = plans[frame_index]
        if plan["mode"] == "crop":
            crop = _extract_crop(
                frame, plan["crop"], padding_mode=config.padding_mode
            )
            interpolation = (
                cv2.INTER_AREA
                if crop.shape[1] >= config.output_width
                and crop.shape[0] >= config.output_height
                else cv2.INTER_LANCZOS4
            )
            rendered = cv2.resize(
                crop,
                (config.output_width, config.output_height),
                interpolation=interpolation,
            )
        else:
            rendered = np.zeros(
                (config.output_height, config.output_width, 3), dtype=np.uint8
            )

        writer.write(rendered)

    writer.release()
    cap.release()
    return output


def save_preview(
    video_path: str | Path,
    plans: Sequence[Mapping[str, Any]],
    info: VideoInfo,
    config: CropConfig,
    output_path: str | Path,
) -> Path | None:
    """Save the first non-black planned frame as a JPEG preview."""
    selected = next((plan for plan in plans if plan["mode"] == "crop"), None)
    if selected is None:
        return None
    frame = read_frame(video_path, int(selected["frame"]))
    crop = _extract_crop(frame, selected["crop"], padding_mode=config.padding_mode)
    preview = cv2.resize(
        crop,
        (config.output_width, config.output_height),
        interpolation=cv2.INTER_AREA,
    )
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), preview):
        raise RuntimeError(f"Could not save preview: {output}")
    return output
