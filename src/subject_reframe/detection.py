"""Scene detection using histogram and pixel-difference analysis."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from .core import Scene, VideoInfo
from .io import inspect_video, read_frame

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


def _scene_signature(frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Create inexpensive grayscale and color-histogram scene signatures."""
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist(
        [small], [0, 1], None, [32, 32], [0, 256, 0, 256]
    )
    cv2.normalize(histogram, histogram)
    return gray, histogram


def _scene_change_metrics(
    previous_gray: np.ndarray,
    previous_histogram: np.ndarray,
    frame: np.ndarray,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Compute correlation and mean pixel difference between frames."""
    gray, histogram = _scene_signature(frame)
    correlation = float(
        cv2.compareHist(previous_histogram, histogram, cv2.HISTCMP_CORREL)
    )
    mean_difference = float(cv2.absdiff(previous_gray, gray).mean())
    return correlation, mean_difference, gray, histogram


def _is_scene_cut(
    correlation: float,
    mean_difference: float,
    *,
    histogram_correlation_threshold: float,
    mean_difference_threshold: float,
    extreme_difference_threshold: float,
) -> bool:
    """Determine if frame pair indicates a hard cut."""
    return (
        correlation < histogram_correlation_threshold
        and mean_difference > mean_difference_threshold
    ) or mean_difference > extreme_difference_threshold


def _refine_scene_cut(
    video_path: str | Path,
    start_frame: int,
    end_frame: int,
    *,
    histogram_correlation_threshold: float,
    mean_difference_threshold: float,
    extreme_difference_threshold: float,
) -> Optional[int]:
    """Find the exact hard-cut frame inside a coarse sampled interval."""
    if end_frame < start_frame:
        return None

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame - 1))
    ok, previous = cap.read()
    if not ok:
        cap.release()
        return None
    previous_gray, previous_histogram = _scene_signature(previous)

    best_frame: int | None = None
    best_score = float("-inf")
    for frame_index in range(start_frame, end_frame + 1):
        ok, frame = cap.read()
        if not ok:
            break
        correlation, mean_difference, gray, histogram = _scene_change_metrics(
            previous_gray, previous_histogram, frame
        )
        if _is_scene_cut(
            correlation,
            mean_difference,
            histogram_correlation_threshold=histogram_correlation_threshold,
            mean_difference_threshold=mean_difference_threshold,
            extreme_difference_threshold=extreme_difference_threshold,
        ):
            score = mean_difference + max(0.0, 1.0 - correlation) * 50.0
            if score > best_score:
                best_score = score
                best_frame = frame_index
        previous_gray = gray
        previous_histogram = histogram

    cap.release()
    return best_frame


def _input_md5(video_path: str | Path) -> str:
    """Compute a stable MD5 digest for the input video file."""
    digest = hashlib.md5()
    with Path(video_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_scene_cache(cache_path: str | Path, expected_md5: str) -> list[Scene] | None:
    """Load cached scene data when the stored input hash matches the current video."""
    cache_file = Path(cache_path).expanduser().resolve()
    if not cache_file.exists():
        return None

    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None

    stored_md5 = payload.get("md5")
    stored_scenes = payload.get("scenes")
    if stored_md5 != expected_md5 or not isinstance(stored_scenes, list):
        return None

    try:
        return [Scene(**scene) for scene in stored_scenes]
    except (TypeError, ValueError, KeyError):
        return None


def _save_scene_cache(cache_path: str | Path, input_md5: str, scenes: Sequence[Scene]) -> None:
    """Persist detected scene boundaries alongside the source-video digest."""
    cache_file = Path(cache_path).expanduser().resolve()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "md5": input_md5,
        "scenes": [asdict(scene) for scene in scenes],
    }
    cache_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def detect_scenes(
    video_path: str | Path,
    *,
    histogram_correlation_threshold: float = 0.35,
    mean_difference_threshold: float = 30.0,
    extreme_difference_threshold: float = 68.0,
    min_scene_seconds: float = 1.0,
    scan_stride: int = 5,
    manual_cut_seconds: Iterable[float] = (),
    cache_path: str | Path | None = None,
    progress: bool = True,
) -> List[Scene]:
    """Detect hard cuts on sampled frames, then refine each cut exactly.

    Automatic scene detection is intentionally reviewable. Manual cuts are
    merged into the result so unusual fades or missed edits can be corrected.
    Cached scene boundaries are reused when the input MD5 matches the stored file.
    """
    if scan_stride < 1:
        raise ValueError("scan_stride must be at least 1")

    info = inspect_video(video_path)
    input_md5 = _input_md5(info.path)
    if cache_path is not None:
        cached = _load_scene_cache(cache_path, input_md5)
        if cached is not None:
            return cached

    cap = cv2.VideoCapture(info.path)
    min_scene_frames = max(1, round(min_scene_seconds * info.fps))

    ok, previous = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Could not read the first video frame")

    previous_gray, previous_histogram = _scene_signature(previous)
    previous_sample_frame = 0

    cuts = [0]
    last_cut = 0
    sample_count = math.ceil(max(0, info.frame_count - 1) / scan_stride)
    bar = tqdm(total=sample_count, desc="Detecting scenes", disable=not progress)

    for frame_index in range(1, info.frame_count):
        if not cap.grab():
            break
        should_sample = frame_index % scan_stride == 0 or frame_index == info.frame_count - 1
        if not should_sample:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break

        correlation, mean_difference, gray, histogram = _scene_change_metrics(
            previous_gray, previous_histogram, frame
        )
        likely_cut = _is_scene_cut(
            correlation,
            mean_difference,
            histogram_correlation_threshold=histogram_correlation_threshold,
            mean_difference_threshold=mean_difference_threshold,
            extreme_difference_threshold=extreme_difference_threshold,
        )

        if likely_cut:
            refined_cut = _refine_scene_cut(
                info.path,
                previous_sample_frame + 1,
                frame_index,
                histogram_correlation_threshold=histogram_correlation_threshold,
                mean_difference_threshold=mean_difference_threshold,
                extreme_difference_threshold=extreme_difference_threshold,
            )
            if refined_cut is not None and refined_cut - last_cut >= min_scene_frames:
                cuts.append(refined_cut)
                last_cut = refined_cut

        previous_gray = gray
        previous_histogram = histogram
        previous_sample_frame = frame_index
        bar.update(1)

    bar.close()
    cap.release()

    for seconds in manual_cut_seconds:
        frame_index = int(round(float(seconds) * info.fps))
        if 0 < frame_index < info.frame_count:
            cuts.append(frame_index)

    cuts.append(info.frame_count)
    cuts = sorted(set(cuts))
    scenes = [
        Scene(index=i, start_frame=start, end_frame=end, fps=info.fps)
        for i, (start, end) in enumerate(zip(cuts[:-1], cuts[1:]))
        if end > start
    ]
    if cache_path is not None:
        _save_scene_cache(cache_path, input_md5, scenes)
    return scenes


def print_scene_table(scenes: Sequence[Scene]) -> None:
    """Print a formatted table of detected scenes."""
    print(f"{'Scene':>5}  {'Start':>10}  {'End':>10}  {'Frames':>8}")
    for scene in scenes:
        print(
            f"{scene.index:>5}  {scene.start_seconds:>10.2f}  "
            f"{scene.end_seconds:>10.2f}  {scene.frame_count:>8}"
        )
