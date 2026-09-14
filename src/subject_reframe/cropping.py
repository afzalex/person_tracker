"""Crop planning, interpolation, and smoothing."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .core import CropConfig, Scene, VideoInfo


def validate_target_tracks(
    scenes: Sequence[Scene],
    records: Mapping[int, Sequence[Mapping[str, Any]]],
    target_tracks: Mapping[int, int | Sequence[int] | None],
) -> List[str]:
    """Return human-readable selection problems without changing any data."""
    problems: list[str] = []
    for scene in scenes:
        if scene.index not in target_tracks:
            problems.append(f"Scene {scene.index} has no selection entry")
            continue
        selection = target_tracks[scene.index]
        if selection is None:
            continue
        target_ids = {int(selection)} if isinstance(selection, int) else {int(value) for value in selection}
        if not target_ids:
            problems.append(f"Scene {scene.index} has an empty Track ID list")
            continue
        present_ids = {
            int(item["track_id"])
            for frame_index in range(scene.start_frame, scene.end_frame)
            for item in records.get(frame_index, ())
        }
        for target_id in sorted(target_ids - present_ids):
            problems.append(f"Track ID {target_id} does not occur in scene {scene.index}")
    return problems


def _contiguous_false_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Find contiguous segments of False values in a boolean mask."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if not value and start is None:
            start = index
        elif value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _interpolate_short_gaps(
    boxes: np.ndarray, detected: np.ndarray, max_gap_frames: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate box positions for short tracking gaps."""
    filled = boxes.copy()
    usable = detected.copy()
    for start, end in _contiguous_false_runs(detected):
        bounded = start > 0 and end < len(detected)
        if not bounded or end - start > max_gap_frames:
            continue
        left = filled[start - 1]
        right = filled[end]
        length = end - start
        for offset, index in enumerate(range(start, end), start=1):
            weight = offset / (length + 1)
            filled[index] = left * (1.0 - weight) + right * weight
            usable[index] = True
    return filled, usable


def _smooth_segment(values: np.ndarray, window: int) -> np.ndarray:
    """Apply moving-average smoothing to a segment."""
    if window <= 1 or len(values) <= 2:
        return values.copy()
    window = min(window, len(values))
    if window % 2 == 0:
        window -= 1
    if window <= 1:
        return values.copy()
    radius = window // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def _smooth_valid_runs(values: np.ndarray, usable: np.ndarray, window: int) -> np.ndarray:
    """Apply smoothing only to contiguous valid segments."""
    smoothed = values.copy()
    index = 0
    while index < len(usable):
        if not usable[index]:
            index += 1
            continue
        end = index + 1
        while end < len(usable) and usable[end]:
            end += 1
        smoothed[index:end] = _smooth_segment(values[index:end], window)
        index = end
    return smoothed


def _shift_inside_frame(
    cx: float, cy: float, width: float, height: float, frame_width: int, frame_height: int
) -> Tuple[float, float, float, float]:
    """Shift crop rectangle to stay within frame boundaries."""
    x1 = cx - width / 2
    x2 = cx + width / 2
    y1 = cy - height / 2
    y2 = cy + height / 2

    if width <= frame_width:
        if x1 < 0:
            x2 -= x1
            x1 = 0.0
        if x2 > frame_width:
            shift = x2 - frame_width
            x1 -= shift
            x2 = float(frame_width)
    if height <= frame_height:
        if y1 < 0:
            y2 -= y1
            y1 = 0.0
        if y2 > frame_height:
            shift = y2 - frame_height
            y1 -= shift
            y2 = float(frame_height)

    return x1, y1, x2, y2


def build_crop_plan(
    info: VideoInfo,
    scenes: Sequence[Scene],
    records: Mapping[int, Sequence[Mapping[str, Any]]],
    target_tracks: Mapping[int, int | Sequence[int] | None],
    config: CropConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Create a smoothed fixed-ratio crop plan and aggregate quality warnings."""
    selection_problems = validate_target_tracks(scenes, records, target_tracks)
    if selection_problems:
        raise ValueError("Invalid target selection:\n- " + "\n- ".join(selection_problems))

    plans: list[dict[str, Any] | None] = [None] * info.frame_count
    warnings: list[dict[str, Any]] = []
    smoothing_window = max(1, round(config.smoothing_seconds * info.fps))
    max_gap = max(0, round(config.max_interpolation_seconds * info.fps))

    for scene in scenes:
        selection = target_tracks[scene.index]
        n = scene.frame_count

        if selection is None:
            for frame_index in range(scene.start_frame, scene.end_frame):
                plans[frame_index] = {
                    "frame": frame_index,
                    "scene": scene.index,
                    "mode": config.absent_scene_policy,
                    "detected": False,
                }
            warnings.append(
                {
                    "type": "target_absent",
                    "scene": scene.index,
                    "start_seconds": scene.start_seconds,
                    "end_seconds": scene.end_seconds,
                    "count": n,
                    "details": "No target track selected; frames will be black.",
                }
            )
            continue

        target_ids = (
            {int(selection)}
            if isinstance(selection, int)
            else {int(track_id) for track_id in selection}
        )

        boxes = np.full((n, 4), np.nan, dtype=np.float64)
        confidences = np.full(n, np.nan, dtype=np.float64)
        detected = np.zeros(n, dtype=bool)

        for local_index, frame_index in enumerate(
            range(scene.start_frame, scene.end_frame)
        ):
            matches = [
                item
                for item in records.get(frame_index, ())
                if int(item["track_id"]) in target_ids
            ]
            match = max(matches, key=lambda item: float(item["confidence"]), default=None)
            if match is not None:
                boxes[local_index] = np.asarray(match["xyxy"], dtype=np.float64)
                confidences[local_index] = float(match["confidence"])
                detected[local_index] = True

        boxes, usable = _interpolate_short_gaps(boxes, detected, max_gap)

        missing_count = int((~usable).sum())
        if missing_count:
            missing_indices = np.flatnonzero(~usable)
            warnings.append(
                {
                    "type": "tracking_missing",
                    "scene": scene.index,
                    "start_seconds": (
                        scene.start_frame + int(missing_indices[0])
                    )
                    / info.fps,
                    "end_seconds": (
                        scene.start_frame + int(missing_indices[-1]) + 1
                    )
                    / info.fps,
                    "count": missing_count,
                    "details": (
                        "Target was missing beyond the interpolation limit; "
                        "affected frames will be black."
                    ),
                }
            )

        low_confidence = detected & (confidences < config.low_confidence_threshold)
        if low_confidence.any():
            indices = np.flatnonzero(low_confidence)
            warnings.append(
                {
                    "type": "low_confidence",
                    "scene": scene.index,
                    "start_seconds": (scene.start_frame + int(indices[0])) / info.fps,
                    "end_seconds": (scene.start_frame + int(indices[-1]) + 1)
                    / info.fps,
                    "count": int(low_confidence.sum()),
                    "details": f"Detection confidence fell below {config.low_confidence_threshold:.2f}.",
                }
            )

        centers_x = np.full(n, np.nan)
        centers_y = np.full(n, np.nan)
        crop_heights = np.full(n, np.nan)

        valid_indices = np.flatnonzero(usable)
        for local_index in valid_indices:
            x1, y1, x2, y2 = boxes[local_index]
            body_width = max(2.0, x2 - x1)
            body_height = max(2.0, y2 - y1)
            padded_x1 = x1 - body_width * config.horizontal_margin
            padded_x2 = x2 + body_width * config.horizontal_margin
            padded_y1 = y1 - body_height * config.top_margin
            padded_y2 = y2 + body_height * config.bottom_margin
            padded_width = padded_x2 - padded_x1
            padded_height = padded_y2 - padded_y1

            if padded_width / padded_height < config.aspect_ratio:
                crop_height = padded_height
            else:
                crop_height = padded_width / config.aspect_ratio

            centers_x[local_index] = (padded_x1 + padded_x2) / 2
            centers_y[local_index] = (padded_y1 + padded_y2) / 2
            crop_heights[local_index] = crop_height

        centers_x = _smooth_valid_runs(centers_x, usable, smoothing_window)
        centers_y = _smooth_valid_runs(centers_y, usable, smoothing_window)
        log_heights = np.where(usable, np.log(crop_heights), np.nan)
        log_heights = _smooth_valid_runs(log_heights, usable, smoothing_window)
        crop_heights = np.exp(log_heights)

        upscale_factors: list[float] = []
        native_sizes: list[tuple[int, int]] = []
        padded_frames: list[int] = []

        for local_index, frame_index in enumerate(
            range(scene.start_frame, scene.end_frame)
        ):
            if not usable[local_index]:
                plans[frame_index] = {
                    "frame": frame_index,
                    "scene": scene.index,
                    "mode": "black",
                    "detected": False,
                }
                continue

            crop_height = float(crop_heights[local_index])
            crop_width = crop_height * config.aspect_ratio
            x1, y1, x2, y2 = _shift_inside_frame(
                float(centers_x[local_index]),
                float(centers_y[local_index]),
                crop_width,
                crop_height,
                info.width,
                info.height,
            )
            native_width = max(1, int(round(x2 - x1)))
            native_height = max(1, int(round(y2 - y1)))
            upscale = max(
                config.output_width / native_width,
                config.output_height / native_height,
            )
            needs_padding = x1 < 0 or y1 < 0 or x2 > info.width or y2 > info.height

            if upscale > 1.0:
                upscale_factors.append(upscale)
                native_sizes.append((native_width, native_height))
            if needs_padding:
                padded_frames.append(frame_index)

            plans[frame_index] = {
                "frame": frame_index,
                "scene": scene.index,
                "mode": "crop",
                "detected": bool(detected[local_index]),
                "crop": [float(x1), float(y1), float(x2), float(y2)],
                "native_size": [native_width, native_height],
                "requires_upscale": upscale > 1.0,
                "upscale_factor": float(upscale),
                "needs_padding": bool(needs_padding),
            }

        if upscale_factors:
            min_width = min(size[0] for size in native_sizes)
            min_height = min(size[1] for size in native_sizes)
            warnings.append(
                {
                    "type": "upscaling",
                    "scene": scene.index,
                    "start_seconds": scene.start_seconds,
                    "end_seconds": scene.end_seconds,
                    "count": len(upscale_factors),
                    "details": (
                        f"Smallest crop {min_width}x{min_height}; maximum upscale "
                        f"{max(upscale_factors):.2f}x to "
                        f"{config.output_width}x{config.output_height}."
                    ),
                }
            )

        if padded_frames:
            warnings.append(
                {
                    "type": "padding",
                    "scene": scene.index,
                    "start_seconds": padded_frames[0] / info.fps,
                    "end_seconds": (padded_frames[-1] + 1) / info.fps,
                    "count": len(padded_frames),
                    "details": (
                        "The required 9:16 crop extends beyond the source frame; "
                        f"{config.padding_mode} padding will be used."
                    ),
                }
            )

    completed_plans = [
        plan
        if plan is not None
        else {"frame": index, "scene": -1, "mode": "black", "detected": False}
        for index, plan in enumerate(plans)
    ]
    return completed_plans, warnings


def print_warning_summary(warnings: Sequence[Mapping[str, Any]]) -> None:
    """Print a formatted summary of all warnings."""
    if not warnings:
        print("No warnings were generated.")
        return
    for warning in warnings:
        print(
            f"[{warning['type'].upper()}] Scene {warning['scene']} — "
            f"{warning['count']} frame(s): {warning['details']}"
        )
