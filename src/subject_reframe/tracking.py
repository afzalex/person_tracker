"""Person detection, tracking, and ranking."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import cv2
import numpy as np

from .core import Scene, VideoInfo
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


class PersonTrackingSession:
    """Reuse one YOLO model while tracking and resetting scenes independently."""

    def __init__(
        self,
        *,
        model_name: str = "yolo11n.pt",
        tracker_name: str = "strongsort.yaml",
        device: int | str = 0,
        quantize: int | str | None = 16,
        image_size: int = 512,
        frame_stride: int = 2,
        confidence: float = 0.20,
        iou: float = 0.50,
        max_detections: int = 300,
    ) -> None:
        from ultralytics import YOLO

        if frame_stride < 1:
            raise ValueError("frame_stride must be at least 1")
        if image_size < 32:
            raise ValueError("image_size must be at least 32 pixels")
        if max_detections < 1:
            raise ValueError("max_detections must be at least 1")

        self.model_name = model_name
        self.tracker_name = tracker_name
        self.device = device
        self.quantize = quantize
        self.image_size = image_size
        self.frame_stride = frame_stride
        self.confidence = confidence
        self.iou = iou
        self.max_detections = max_detections
        self.model = YOLO(model_name)
        self.closed = False

    def _reset_tracker(self) -> None:
        if self.model.predictor is None:
            return
        for tracker in getattr(self.model.predictor, "trackers", ()):
            tracker.reset()
        if hasattr(self.model.predictor, "_feats"):
            self.model.predictor._feats = None

    def track_scene(
        self,
        video_path: str | Path,
        scene: Scene,
        *,
        progress: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        progress_description: str | None = None,
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Track all people in exactly one scene and return per-frame records."""
        if self.closed:
            raise RuntimeError("This PersonTrackingSession has already been closed")

        self._reset_tracker()
        records: dict[int, list[dict[str, Any]]] = {}
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, scene.start_frame)
        bar = tqdm(
            total=scene.frame_count,
            desc=progress_description or f"Tracking scene {scene.index}",
            disable=not progress,
            unit="frame",
            dynamic_ncols=True,
        )
        callback_interval = max(1, scene.frame_count // 100)

        def update_progress(processed_frames: int) -> None:
            bar.update(1)
            if progress_callback is not None and (
                processed_frames == scene.frame_count
                or processed_frames % callback_interval == 0
            ):
                progress_callback(processed_frames, scene.frame_count)

        if progress_callback is not None:
            progress_callback(0, scene.frame_count)

        for frame_index in range(scene.start_frame, scene.end_frame):
            if not cap.grab():
                bar.close()
                cap.release()
                raise RuntimeError(f"Could not read frame {frame_index} while tracking")

            local_frame = frame_index - scene.start_frame
            should_track = (
                local_frame % self.frame_stride == 0
                or frame_index == scene.end_frame - 1
            )
            if not should_track:
                records[frame_index] = []
                update_progress(local_frame + 1)
                continue

            ok, frame = cap.retrieve()
            if not ok:
                bar.close()
                cap.release()
                raise RuntimeError(f"Could not decode frame {frame_index} while tracking")

            result = self.model.track(
                frame,
                persist=True,
                classes=[0],
                tracker=self.tracker_name,
                device=self.device,
                quantize=self.quantize,
                imgsz=self.image_size,
                conf=self.confidence,
                iou=self.iou,
                max_det=self.max_detections,
                verbose=False,
            )[0]

            detections: list[dict[str, Any]] = []
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                ids = boxes.id.detach().cpu().numpy().astype(int)
                confidences = boxes.conf.detach().cpu().numpy()
                for box, track_id, score in zip(xyxy, ids, confidences):
                    detections.append(
                        {
                            "scene": scene.index,
                            "track_id": int(track_id),
                            "xyxy": [float(value) for value in box],
                            "confidence": float(score),
                        }
                    )
            records[frame_index] = detections
            update_progress(local_frame + 1)

        bar.close()
        cap.release()
        return records

    def close(self) -> None:
        if self.closed:
            return
        self.model.predictor = None
        del self.model
        self.closed = True


def track_all_people(
    video_path: str | Path,
    scenes: Sequence[Scene],
    *,
    model_name: str = "yolo11n.pt",
    tracker_name: str = "strongsort.yaml",
    device: int | str = 0,
    quantize: int | str | None = 16,
    image_size: int = 512,
    frame_stride: int = 2,
    confidence: float = 0.20,
    iou: float = 0.50,
    max_detections: int = 300,
    progress: bool = True,
) -> Dict[int, List[Dict[str, Any]]]:
    """Compatibility wrapper that tracks a sequence of scenes."""
    session = PersonTrackingSession(
        model_name=model_name,
        tracker_name=tracker_name,
        device=device,
        quantize=quantize,
        image_size=image_size,
        frame_stride=frame_stride,
        confidence=confidence,
        iou=iou,
        max_detections=max_detections,
    )
    records: dict[int, list[dict[str, Any]]] = {}
    try:
        for scene in scenes:
            records.update(session.track_scene(video_path, scene, progress=progress))
    finally:
        session.close()
    return records


def summarize_scene_tracks(
    scene: Scene,
    records: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    frame_stride: int,
) -> list[dict[str, Any]]:
    """Summarize sampled-frame coverage and confidence for every scene-local ID."""
    sampled_frames = list(range(scene.start_frame, scene.end_frame, frame_stride))
    if scene.end_frame - 1 not in sampled_frames:
        sampled_frames.append(scene.end_frame - 1)
    expected = max(1, len(sampled_frames))
    by_id: dict[int, list[tuple[int, float]]] = {}
    for frame_index in sampled_frames:
        for detection in records.get(frame_index, ()):
            track_id = int(detection["track_id"])
            by_id.setdefault(track_id, []).append(
                (frame_index, float(detection["confidence"]))
            )

    summary: list[dict[str, Any]] = []
    for track_id, observations in by_id.items():
        frames = sorted(frame for frame, _ in observations)
        confidence_values = [confidence for _, confidence in observations]
        boundaries = [scene.start_frame - frame_stride, *frames, scene.end_frame - 1 + frame_stride]
        maximum_gap_frames = max(
            max(0, right - left - frame_stride)
            for left, right in zip(boundaries[:-1], boundaries[1:])
        )
        summary.append(
            {
                "track_id": track_id,
                "observed_samples": len(frames),
                "expected_samples": expected,
                "coverage_percent": len(frames) / expected * 100.0,
                "mean_confidence": float(np.mean(confidence_values)),
                "maximum_gap_seconds": maximum_gap_frames / scene.fps,
            }
        )
    return sorted(summary, key=lambda item: (-item["coverage_percent"], item["track_id"]))


def print_scene_track_summary(
    scene: Scene,
    records: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    frame_stride: int,
) -> None:
    """Print a formatted table of detected tracks in a scene."""
    summary = summarize_scene_tracks(scene, records, frame_stride=frame_stride)
    if not summary:
        print(f"Scene {scene.index}: no person tracks were found.")
        return
    print(f"{'Track ID':>8}  {'Coverage':>9}  {'Mean conf.':>10}  {'Max gap':>9}")
    for item in summary:
        print(
            f"{item['track_id']:>8}  {item['coverage_percent']:>8.1f}%  "
            f"{item['mean_confidence']:>10.3f}  "
            f"{item['maximum_gap_seconds']:>7.2f}s"
        )


def rank_scene_tracks(
    scene: Scene,
    records: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    frame_stride: int,
    frame_width: int,
    frame_height: int,
) -> list[dict[str, Any]]:
    """Rank scene-local IDs and choose the clearest thumbnail frame per track."""
    sampled_frames = list(range(scene.start_frame, scene.end_frame, frame_stride))
    if scene.end_frame - 1 not in sampled_frames:
        sampled_frames.append(scene.end_frame - 1)
    expected_samples = max(1, len(sampled_frames))
    frame_area = max(1.0, float(frame_width * frame_height))
    observations: dict[int, list[dict[str, Any]]] = {}

    for frame_index in sampled_frames:
        for detection in records.get(frame_index, ()):
            x1, y1, x2, y2 = [float(value) for value in detection["xyxy"]]
            width = max(1.0, x2 - x1)
            height = max(1.0, y2 - y1)
            area_ratio = width * height / frame_area
            edge_margin_x = frame_width * 0.01
            edge_margin_y = frame_height * 0.01
            clipped = (
                x1 <= edge_margin_x
                or y1 <= edge_margin_y
                or x2 >= frame_width - edge_margin_x
                or y2 >= frame_height - edge_margin_y
            )
            confidence = float(detection["confidence"])
            thumbnail_score = confidence * math.sqrt(max(area_ratio, 1e-8))
            if clipped:
                thumbnail_score *= 0.65
            observations.setdefault(int(detection["track_id"]), []).append(
                {
                    "frame": frame_index,
                    "xyxy": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "area_ratio": area_ratio,
                    "clipped": clipped,
                    "thumbnail_score": thumbnail_score,
                }
            )

    ranked: list[dict[str, Any]] = []
    for track_id, track_observations in observations.items():
        coverage = len(track_observations) / expected_samples
        mean_confidence = float(
            np.mean([item["confidence"] for item in track_observations])
        )
        mean_area_ratio = float(
            np.mean([item["area_ratio"] for item in track_observations])
        )
        clipped_fraction = float(
            np.mean([item["clipped"] for item in track_observations])
        )
        size_score = min(1.0, math.sqrt(mean_area_ratio / 0.08))
        ranking_score = (
            coverage * 0.55
            + mean_confidence * 0.25
            + size_score * 0.10
            + (1.0 - clipped_fraction) * 0.10
        )
        best = max(track_observations, key=lambda item: item["thumbnail_score"])
        ranked.append(
            {
                "track_id": track_id,
                "ranking_score": ranking_score,
                "coverage_percent": coverage * 100.0,
                "mean_confidence": mean_confidence,
                "mean_area_percent": mean_area_ratio * 100.0,
                "clipped_percent": clipped_fraction * 100.0,
                "best_frame": int(best["frame"]),
                "best_xyxy": best["xyxy"],
            }
        )
    return sorted(ranked, key=lambda item: (-item["ranking_score"], item["track_id"]))


def choose_candidate_frames(
    scenes: Sequence[Scene],
    records: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    override_seconds: Mapping[int, float] | None = None,
) -> dict[int, int]:
    """Choose an informative frame per scene, preferring frames with people."""
    override_seconds = override_seconds or {}
    chosen: dict[int, int] = {}

    for scene in scenes:
        if scene.index in override_seconds:
            requested = int(round(override_seconds[scene.index] * scene.fps))
            requested = min(
                max(requested, scene.start_frame), scene.end_frame - 1
            )
            frames_with_tracks = [
                frame_index
                for frame_index in range(scene.start_frame, scene.end_frame)
                if records.get(frame_index)
            ]
            chosen[scene.index] = (
                min(frames_with_tracks, key=lambda frame: abs(frame - requested))
                if frames_with_tracks
                else requested
            )
            continue

        best_frame = scene.start_frame
        best_count = -1
        # Prefer the earliest frame with the highest number of visible tracks.
        for frame_index in range(scene.start_frame, scene.end_frame):
            count = len(records.get(frame_index, ()))
            if count > best_count:
                best_count = count
                best_frame = frame_index
        chosen[scene.index] = best_frame

    return chosen


def _annotate_track_ids(
    frame: np.ndarray, detections: Sequence[Mapping[str, Any]]
) -> np.ndarray:
    """Annotate frame with track ID labels and bounding boxes."""
    annotated = frame.copy()
    frame_height, frame_width = frame.shape[:2]
    font_scale = max(0.9, min(2.4, min(frame_width, frame_height) / 720 * 1.4))
    text_thickness = max(2, round(font_scale * 2))
    box_thickness = max(3, round(font_scale * 3))
    padding = max(6, round(font_scale * 7))

    for detection in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in detection["xyxy"]]
        track_id = int(detection["track_id"])
        color = (40, 220, 80)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)
        label = f"ID {track_id}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
        )
        label_height = text_height + baseline + padding * 2
        label_width = text_width + padding * 2
        label_top = y1 - label_height if y1 >= label_height else y1
        label_bottom = min(frame_height, label_top + label_height)
        label_right = min(frame_width, x1 + label_width)
        cv2.rectangle(
            annotated,
            (max(0, x1), max(0, label_top)),
            (label_right, label_bottom),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (max(0, x1) + padding, label_top + padding + text_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            text_thickness,
            cv2.LINE_AA,
        )
    return annotated
