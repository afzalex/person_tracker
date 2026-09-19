"""Reviewable hard-cut scene detection and optional clip extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Callable, Iterable, Sequence

import cv2
import numpy as np
from .core import VideoInfo
from .io import inspect_video
from .scene_storage import SceneRecord


@dataclass(frozen=True)
class SceneCut:
    """A detected or manually supplied boundary at the start of a new scene."""

    frame: int
    score: float | None
    correlation: float | None
    mean_difference: float | None
    source: str = "automatic"


def _signature(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([small], [0, 1], None, [32, 32], [0, 256, 0, 256])
    cv2.normalize(histogram, histogram)
    return gray, histogram


def _metrics(
    previous_gray: np.ndarray,
    previous_histogram: np.ndarray,
    frame: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    gray, histogram = _signature(frame)
    correlation = float(cv2.compareHist(previous_histogram, histogram, cv2.HISTCMP_CORREL))
    mean_difference = float(cv2.absdiff(previous_gray, gray).mean())
    return correlation, mean_difference, gray, histogram


def _cut_score(correlation: float, mean_difference: float) -> float:
    return mean_difference + max(0.0, 1.0 - correlation) * 50.0


def _is_cut(
    correlation: float,
    mean_difference: float,
    *,
    histogram_correlation_threshold: float,
    mean_difference_threshold: float,
    extreme_difference_threshold: float,
) -> bool:
    return (
        correlation < histogram_correlation_threshold
        and mean_difference > mean_difference_threshold
    ) or mean_difference > extreme_difference_threshold


def _refine_cut(
    video_path: str | Path,
    start_frame: int,
    end_frame: int,
    **thresholds,
) -> SceneCut | None:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame - 1))
    ok, previous = cap.read()
    if not ok:
        cap.release()
        return None
    previous_gray, previous_histogram = _signature(previous)
    best: SceneCut | None = None
    for frame_index in range(start_frame, end_frame + 1):
        ok, frame = cap.read()
        if not ok:
            break
        correlation, mean_difference, gray, histogram = _metrics(
            previous_gray, previous_histogram, frame
        )
        if _is_cut(correlation, mean_difference, **thresholds):
            candidate = SceneCut(
                frame=frame_index,
                score=_cut_score(correlation, mean_difference),
                correlation=correlation,
                mean_difference=mean_difference,
            )
            if best is None or candidate.score > best.score:
                best = candidate
        previous_gray, previous_histogram = gray, histogram
    cap.release()
    return best


def resolve_scene_window(
    video_info: VideoInfo,
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> tuple[int, int]:
    """Resolve one half-open source-video window from frame or time values."""
    if start_frame is not None and start_seconds is not None:
        raise ValueError("Provide start_frame or start_seconds, not both")
    if end_frame is not None and end_seconds is not None:
        raise ValueError("Provide end_frame or end_seconds, not both")
    window_start = (
        int(start_frame) if start_frame is not None
        else int(round(float(start_seconds) * video_info.fps))
        if start_seconds is not None else 0
    )
    window_end = (
        int(end_frame) if end_frame is not None
        else int(round(float(end_seconds) * video_info.fps))
        if end_seconds is not None else video_info.frame_count
    )
    if not 0 <= window_start < window_end <= video_info.frame_count:
        raise ValueError(
            "Scene window must satisfy "
            f"0 <= start < end <= {video_info.frame_count}; "
            f"received {window_start}:{window_end}"
        )
    return window_start, window_end


def detect_scene_cuts(
    video_path: str | Path,
    *,
    histogram_correlation_threshold: float = 0.35,
    mean_difference_threshold: float = 30.0,
    extreme_difference_threshold: float = 68.0,
    min_scene_seconds: float = 1.0,
    scan_stride: int = 5,
    start_frame: int | None = None,
    end_frame: int | None = None,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[VideoInfo, list[SceneCut]]:
    """Detect hard cuts in an optional half-open frame/time window.

    ``end_frame`` and ``end_seconds`` are exclusive. Frame and time values cannot
    both be supplied for the same boundary. Progress belongs to the caller and is
    reported as ``progress_callback(done, total)`` when a callback is provided.
    """
    if scan_stride < 1:
        raise ValueError("scan_stride must be at least 1")
    if min_scene_seconds < 0:
        raise ValueError("min_scene_seconds cannot be negative")

    info = inspect_video(video_path)
    window_start, window_end = resolve_scene_window(
        info,
        start_frame=start_frame,
        end_frame=end_frame,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )
    thresholds = {
        "histogram_correlation_threshold": histogram_correlation_threshold,
        "mean_difference_threshold": mean_difference_threshold,
        "extreme_difference_threshold": extreme_difference_threshold,
    }
    cap = cv2.VideoCapture(info.path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, window_start)
    ok, previous = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Could not read detection start frame {window_start}")
    previous_gray, previous_histogram = _signature(previous)
    previous_sample_frame = window_start
    minimum_frames = max(1, round(min_scene_seconds * info.fps))
    last_cut = window_start
    cuts: list[SceneCut] = []
    total_frames = window_end - window_start - 1
    if progress_callback is not None:
        progress_callback(0, total_frames)

    for done, frame_index in enumerate(
        range(window_start + 1, window_end), start=1
    ):
        if not cap.grab():
            break
        if progress_callback is not None:
            progress_callback(done, total_frames)
        should_sample = (
            (frame_index - window_start) % scan_stride == 0
            or frame_index == window_end - 1
        )
        if not should_sample:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        correlation, mean_difference, gray, histogram = _metrics(
            previous_gray, previous_histogram, frame
        )
        if _is_cut(correlation, mean_difference, **thresholds):
            refined = _refine_cut(
                info.path, previous_sample_frame + 1, frame_index, **thresholds
            )
            if refined is not None and refined.frame - last_cut >= minimum_frames:
                cuts.append(refined)
                last_cut = refined.frame
        previous_gray, previous_histogram = gray, histogram
        previous_sample_frame = frame_index

    cap.release()
    return info, cuts


def combine_scene_cuts(
    automatic_cuts: Sequence[SceneCut],
    *,
    fps: float,
    frame_count: int,
    manual_cut_seconds: Iterable[float] = (),
    remove_cut_frames: Iterable[int] = (),
) -> list[SceneCut]:
    """Apply manual additions/removals while preserving automatic cut evidence."""
    removed = {int(frame) for frame in remove_cut_frames}
    by_frame = {
        cut.frame: cut
        for cut in automatic_cuts
        if 0 < cut.frame < frame_count and cut.frame not in removed
    }
    for seconds in manual_cut_seconds:
        frame = int(round(float(seconds) * fps))
        if 0 < frame < frame_count and frame not in removed:
            by_frame[frame] = SceneCut(
                frame=frame,
                score=None,
                correlation=None,
                mean_difference=None,
                source="manual",
            )
    return [by_frame[frame] for frame in sorted(by_frame)]


def build_scene_records(
    video_info: VideoInfo,
    cuts: Sequence[SceneCut],
    *,
    disabled_scene_ids: Iterable[str] = (),
    start_frame: int = 0,
    end_frame: int | None = None,
) -> list[SceneRecord]:
    """Turn cut frames into contiguous records covering one source window."""
    window_end = video_info.frame_count if end_frame is None else int(end_frame)
    if not 0 <= start_frame < window_end <= video_info.frame_count:
        raise ValueError(
            f"Invalid scene window {start_frame}:{window_end} for "
            f"{video_info.frame_count} frames"
        )
    disabled = set(disabled_scene_ids)
    cuts_by_frame = {
        cut.frame: cut for cut in cuts if start_frame < cut.frame < window_end
    }
    boundaries = [start_frame, *sorted(cuts_by_frame), window_end]
    scenes: list[SceneRecord] = []
    for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        scene_id = f"scene-{index + 1:06d}"
        boundary = cuts_by_frame.get(start)
        scenes.append(SceneRecord(
            scene_id=scene_id,
            scene_index=index,
            start_frame=start,
            end_frame=end,
            fps=video_info.fps,
            detection_score=None if boundary is None else boundary.score,
            boundary_source=(
                "video_start" if boundary is None and start == 0
                else "window_start" if boundary is None
                else boundary.source
            ),
            enabled=scene_id not in disabled,
        ))
    return scenes


def extract_scene_clips(
    source_video: str | Path,
    scenes: Sequence[SceneRecord],
    output_directory: str | Path,
    *,
    mode: str = "copy",
    overwrite: bool = False,
) -> list[Path]:
    """Extract enabled scenes with FFmpeg using fast copy or frame-accurate re-encoding."""
    if mode not in {"copy", "reencode"}:
        raise ValueError("mode must be 'copy' or 'reencode'")
    source = Path(source_video).expanduser().resolve()
    output_dir = Path(output_directory).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for scene in scenes:
        if not scene.enabled:
            continue
        output = output_dir / f"{scene.scene_id}.mp4"
        if output.exists() and not overwrite:
            outputs.append(output)
            continue
        command = [
            "ffmpeg", "-y" if overwrite else "-n", "-loglevel", "error",
            "-ss", f"{scene.start_seconds:.9f}",
            "-i", str(source),
            "-t", f"{scene.duration_seconds:.9f}",
            "-map", "0:v:0", "-map", "0:a?",
        ]
        if mode == "copy":
            command += ["-c", "copy"]
        else:
            command += [
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p",
            ]
        command.append(str(output))
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(
                f"FFmpeg failed for {scene.scene_id}: {completed.stderr.strip()}"
            )
        outputs.append(output)
    return outputs


def _subtitle_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"

def create_scene_review_video(
    source_video: str | Path,
    scenes: Sequence[SceneRecord],
    output_path: str | Path,
    *,
    clean_output_path: str | Path | None = None,
    width: int = 960,
    quality: int = 32,
    progress_callback: Callable[[float, float], None] | None = None,
) -> Path:
    """Create labelled and optional clean review videos in one traversal."""

    if width < 160:
        raise ValueError("width must be at least 160 pixels")

    if not scenes:
        raise ValueError("At least one scene is required for a review video")

    source = Path(source_video).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    clean_output = (
        Path(clean_output_path).expanduser().resolve()
        if clean_output_path is not None
        else None
    )

    if clean_output == output:
        raise ValueError(
            "clean_output_path must be different from output_path"
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    if clean_output is not None:
        clean_output.parent.mkdir(parents=True, exist_ok=True)

    review_start = scenes[0].start_seconds
    review_end = scenes[-1].end_seconds
    review_duration = review_end - review_start
    expected_frame_count = (
        scenes[-1].end_frame - scenes[0].start_frame
    )

    if review_duration <= 0:
        raise ValueError("Review-video duration must be greater than zero")

    subtitle_path = output.with_suffix(".srt")
    subtitle_lines: list[str] = []

    for subtitle_index, scene in enumerate(scenes, start=1):
        status = "" if scene.enabled else " | SKIPPED"

        subtitle_lines.extend([
            str(subtitle_index),
            f"{_subtitle_timestamp(scene.start_seconds - review_start)} --> "
            f"{_subtitle_timestamp(scene.end_seconds - review_start)}",
            f"{{\\an8}}Scene {scene.scene_index + 1:06d} | "
            f"{scene.start_seconds:.2f}s–{scene.end_seconds:.2f}s | "
            f"{scene.duration_seconds:.2f}s{status}",
            "",
        ])

    subtitle_path.write_text(
        "\n".join(subtitle_lines),
        encoding="utf-8",
    )

    escaped_subtitles = str(subtitle_path).replace("\\", "\\\\")
    escaped_subtitles = escaped_subtitles.replace(":", "\\:")
    escaped_subtitles = escaped_subtitles.replace("'", "\\'")

    scale_filter = (
        "setpts=PTS-STARTPTS,"
        f"scale={width}:-2:force_original_aspect_ratio=decrease,"
        "pad=ceil(iw/2)*2:ceil(ih/2)*2"
    )
    subtitle_filter = (
        f"subtitles=filename='{escaped_subtitles}':"
        "force_style='Alignment=8,FontSize=18,Outline=2,Shadow=1,"
        "MarginL=0,MarginR=0,MarginV=24'"
    )

    command = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-nostats",
        "-progress", "pipe:1",
        "-ss", f"{review_start:.9f}",
        "-i", str(source),
    ]

    if clean_output is not None:
        filter_complex = (
            f"[0:v]{scale_filter},split=2[clean][label_source];"
            f"[label_source]{subtitle_filter}[labelled]"
        )

        command.extend([
            "-filter_complex", filter_complex,

            "-map", "[clean]",
            "-map", "0:a?",
            "-t", f"{review_duration:.9f}",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", str(quality),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "96k",
            "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart",
            str(clean_output),

            "-map", "[labelled]",
            "-map", "0:a?",
            "-t", f"{review_duration:.9f}",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", str(quality),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "96k",
            "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart",
            str(output),
        ])
    else:
        video_filter = (
            f"{scale_filter},"
            f"{subtitle_filter}"
        )

        command.extend([
            "-map", "0:v:0",
            "-map", "0:a?",
            "-t", f"{review_duration:.9f}",
            "-vf", video_filter,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-crf", str(quality),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "96k",
            "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart",
            str(output),
        ])

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    last_processed_seconds = 0.0

    if progress_callback is not None:
        progress_callback(0.0, review_duration)

    try:
        assert process.stdout is not None

        for line in process.stdout:
            key, separator, value = line.strip().partition("=")

            if not separator or key != "out_time_us":
                continue

            try:
                processed_seconds = min(
                    int(value) / 1_000_000,
                    review_duration,
                )
            except ValueError:
                continue

            if (
                progress_callback is not None
                and processed_seconds > last_processed_seconds
            ):
                progress_callback(
                    processed_seconds,
                    review_duration,
                )

            last_processed_seconds = processed_seconds

        stderr = (
            process.stderr.read()
            if process.stderr is not None
            else ""
        )
        return_code = process.wait()
    except BaseException:
        process.terminate()
        process.wait()
        raise

    if return_code != 0:
        raise RuntimeError(
            f"Could not create scene review videos: "
            f"{stderr.strip()}"
        )

    review_outputs = [output]

    if clean_output is not None:
        review_outputs.append(clean_output)

    duration_tolerance = max(
        0.25,
        2 / scenes[0].fps,
    )

    for review_output in review_outputs:
        review_info = inspect_video(review_output)
        frame_difference = abs(
            review_info.frame_count - expected_frame_count
        )
        duration_difference = abs(
            review_info.duration_seconds - review_duration
        )

        if frame_difference > 2:
            raise RuntimeError(
                f"{review_output.name} has "
                f"{review_info.frame_count} frames; expected approximately "
                f"{expected_frame_count}"
            )

        if duration_difference > duration_tolerance:
            raise RuntimeError(
                f"{review_output.name} duration is "
                f"{review_info.duration_seconds:.3f}s; "
                f"expected approximately {review_duration:.3f}s"
            )

    if progress_callback is not None:
        progress_callback(
            review_duration,
            review_duration,
        )

    return output

def create_selected_scene_review_video(
    review_video: str | Path,
    scenes: Sequence[SceneRecord],
    output_path: str | Path,
    *,
    review_start_frame: int | None = None,
    quality: int = 32,
    progress_callback: Callable[[float, float], None] | None = None,
) -> Path:
    """Create a compact video containing only enabled review-video scenes."""

    if not scenes:
        raise ValueError("At least one scene is required")

    selected_scenes = [
        scene
        for scene in scenes
        if scene.enabled
    ]

    if not selected_scenes:
        raise ValueError("At least one scene must be enabled")

    source = Path(review_video).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if not source.exists():
        raise FileNotFoundError(
            f"Review video does not exist: {source}"
        )

    if review_start_frame is None:
        review_start_frame = scenes[0].start_frame

    fps = scenes[0].fps
    selected_frame_ranges: list[list[int]] = []

    for scene in selected_scenes:
        range_start = scene.start_frame - review_start_frame
        range_end = scene.end_frame - review_start_frame

        if range_start < 0 or range_end <= range_start:
            raise ValueError(
                f"Invalid review-video range for {scene.scene_id}: "
                f"{range_start}:{range_end}"
            )

        if (
            selected_frame_ranges
            and selected_frame_ranges[-1][1] == range_start
        ):
            selected_frame_ranges[-1][1] = range_end
        else:
            selected_frame_ranges.append([
                range_start,
                range_end,
            ])

    selected_frame_count = sum(
        range_end - range_start
        for range_start, range_end in selected_frame_ranges
    )
    selected_duration = selected_frame_count / fps

    video_selection = "+".join(
        f"between(n\\,{range_start}\\,{range_end - 1})"
        for range_start, range_end in selected_frame_ranges
    )
    audio_selection = "+".join(
        f"between(t\\,{range_start / fps:.9f}\\,"
        f"{range_end / fps:.9f})"
        for range_start, range_end in selected_frame_ranges
    )

    probe = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            str(source),
        ],
        capture_output=True,
        text=True,
    )
    has_audio = (
        probe.returncode == 0
        and bool(probe.stdout.strip())
    )

    filter_parts = [
        f"[0:v]select={video_selection},"
        f"setpts=N/({fps:.12f}*TB)[video]"
    ]

    if has_audio:
        filter_parts.append(
            f"[0:a]aselect={audio_selection},"
            f"asetpts=N/SR/TB[audio]"
        )

    command = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-nostats",
        "-progress", "pipe:1",
        "-t", f"{selected_frame_ranges[-1][1] / fps:.9f}",
        "-i", str(source),
        "-filter_complex", ";".join(filter_parts),
        "-map", "[video]",
    ]

    if has_audio:
        command.extend([
            "-map", "[audio]",
        ])

    command.extend([
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", str(quality),
        "-pix_fmt", "yuv420p",
    ])

    if has_audio:
        command.extend([
            "-c:a", "aac",
            "-b:a", "96k",
        ])

    command.extend([
        "-movflags", "+faststart",
        str(output),
    ])

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    last_processed_seconds = 0.0

    if progress_callback is not None:
        progress_callback(0.0, selected_duration)

    try:
        assert process.stdout is not None

        for line in process.stdout:
            key, separator, value = line.strip().partition("=")

            if not separator or key != "out_time_us":
                continue

            try:
                processed_seconds = min(
                    int(value) / 1_000_000,
                    selected_duration,
                )
            except ValueError:
                continue

            if (
                progress_callback is not None
                and processed_seconds > last_processed_seconds
            ):
                progress_callback(
                    processed_seconds,
                    selected_duration,
                )

            last_processed_seconds = processed_seconds

        stderr = (
            process.stderr.read()
            if process.stderr is not None
            else ""
        )
        return_code = process.wait()
    except BaseException:
        process.terminate()
        process.wait()
        raise

    if return_code != 0:
        raise RuntimeError(
            f"Could not create selected-scene review video: "
            f"{stderr.strip()}"
        )

    selected_info = inspect_video(output)
    frame_difference = abs(
        selected_info.frame_count - selected_frame_count
    )
    duration_difference = abs(
        selected_info.duration_seconds - selected_duration
    )
    duration_tolerance = max(
        0.25,
        2 / fps,
    )

    if frame_difference > 2:
        raise RuntimeError(
            f"Selected review has {selected_info.frame_count} frames; "
            f"expected approximately {selected_frame_count}"
        )

    if duration_difference > duration_tolerance:
        raise RuntimeError(
            f"Selected review duration is "
            f"{selected_info.duration_seconds:.3f}s; "
            f"expected approximately {selected_duration:.3f}s"
        )

    if progress_callback is not None:
        progress_callback(
            selected_duration,
            selected_duration,
        )

    return output

def nvenc_available(
    source_video: str | Path | None = None,
) -> bool:
    """Check whether NVIDIA NVENC is usable by FFmpeg."""

    if source_video is None:
        command = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-f", "lavfi",
            "-i", "color=size=128x128:rate=1",
            "-frames:v", "1",
            "-an",
            "-c:v", "h264_nvenc",
            "-f", "null",
            "-",
        ]
    else:
        source = Path(source_video).expanduser().resolve()
        command = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
            "-i", str(source),
            "-frames:v", "1",
            "-an",
            "-c:v", "h264_nvenc",
            "-f", "null",
            "-",
        ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    return completed.returncode == 0

def create_selected_scene_video(
    source_video: str | Path,
    scenes: Sequence[SceneRecord],
    output_path: str | Path,
    *,
    encoder: str = "auto",
    crf: int = 14,
    cpu_preset: str = "medium",
    nvenc_cq: int = 14,
    nvenc_preset: str = "p7",
    audio_bitrate: str = "320k",
    threads: int = 0,
    progress_callback: Callable[[float, float], None] | None = None,
) -> Path:
    """Create a full-resolution, high-quality video of enabled scenes."""

    if not scenes:
        raise ValueError("At least one scene is required")

    if encoder not in {"auto", "nvenc", "x264"}:
        raise ValueError(
            "encoder must be 'auto', 'nvenc', or 'x264'"
        )

    if not 0 <= crf <= 51:
        raise ValueError("crf must be between 0 and 51")

    if not 0 <= nvenc_cq <= 51:
        raise ValueError("nvenc_cq must be between 0 and 51")

    if threads < 0:
        raise ValueError("threads cannot be negative")

    selected_scenes = [
        scene
        for scene in scenes
        if scene.enabled
    ]

    if not selected_scenes:
        raise ValueError("At least one scene must be enabled")

    source = Path(source_video).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if source == output:
        raise ValueError(
            "The output path must be different from the source video"
        )

    if encoder == "auto":
        resolved_encoder = (
            "nvenc"
            if nvenc_available(source)
            else "x264"
        )
    else:
        resolved_encoder = encoder

    if (
        resolved_encoder == "nvenc"
        and not nvenc_available(source)
    ):
        raise RuntimeError(
            "NVENC was requested, but CUDA/NVENC is not available"
        )

    fps = scenes[0].fps
    selected_frame_ranges: list[list[int]] = []

    for scene in selected_scenes:
        if (
            selected_frame_ranges
            and selected_frame_ranges[-1][1] == scene.start_frame
        ):
            selected_frame_ranges[-1][1] = scene.end_frame
        else:
            selected_frame_ranges.append([
                scene.start_frame,
                scene.end_frame,
            ])

    selected_frame_count = sum(
        range_end - range_start
        for range_start, range_end in selected_frame_ranges
    )
    selected_duration = selected_frame_count / fps

    video_selection = "+".join(
        f"between(n\\,{range_start}\\,{range_end - 1})"
        for range_start, range_end in selected_frame_ranges
    )
    audio_selection = "+".join(
        f"between(t\\,{range_start / fps:.9f}\\,"
        f"{range_end / fps:.9f})"
        for range_start, range_end in selected_frame_ranges
    )

    probe = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            str(source),
        ],
        capture_output=True,
        text=True,
    )
    has_audio = (
        probe.returncode == 0
        and bool(probe.stdout.strip())
    )

    filter_parts = [
        f"[0:v]select={video_selection},"
        f"setpts=N/({fps:.12f}*TB)[video]"
    ]

    if has_audio:
        filter_parts.append(
            f"[0:a]aselect={audio_selection},"
            f"asetpts=N/SR/TB[audio]"
        )

    command = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-nostats",
        "-progress", "pipe:1",
    ]

    if resolved_encoder == "nvenc":
        command.extend([
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
        ])

    command.extend([
        "-t", f"{selected_frame_ranges[-1][1] / fps:.9f}",
        "-i", str(source),
        "-filter_complex", ";".join(filter_parts),
        "-map", "[video]",
    ])

    if has_audio:
        command.extend([
            "-map", "[audio]",
        ])

    command.extend([
        "-map_metadata", "0",
        "-map_chapters", "-1",
    ])

    if resolved_encoder == "nvenc":
        command.extend([
            "-c:v", "h264_nvenc",
            "-preset", nvenc_preset,
            "-tune", "hq",
            "-rc", "vbr",
            "-cq", str(nvenc_cq),
            "-b:v", "0",
            "-multipass", "fullres",
            "-spatial-aq", "1",
            "-temporal-aq", "1",
            "-aq-strength", "8",
            "-rc-lookahead", "32",
            "-bf", "3",
            "-b_ref_mode", "middle",
            "-fps_mode", "passthrough",
        ])
    else:
        command.extend([
            "-c:v", "libx264",
            "-preset", cpu_preset,
            "-crf", str(crf),
            "-threads", str(threads),
            "-pix_fmt", "yuv420p",
            "-fps_mode", "passthrough",
        ])

    if has_audio:
        command.extend([
            "-c:a", "aac",
            "-b:a", audio_bitrate,
        ])

    command.extend([
        "-movflags", "+faststart",
        str(output),
    ])

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    last_processed_seconds = 0.0

    if progress_callback is not None:
        progress_callback(0.0, selected_duration)

    try:
        assert process.stdout is not None

        for line in process.stdout:
            key, separator, value = line.strip().partition("=")

            if not separator or key != "out_time_us":
                continue

            try:
                processed_seconds = min(
                    int(value) / 1_000_000,
                    selected_duration,
                )
            except ValueError:
                continue

            if (
                progress_callback is not None
                and processed_seconds > last_processed_seconds
            ):
                progress_callback(
                    processed_seconds,
                    selected_duration,
                )

            last_processed_seconds = processed_seconds

        stderr = (
            process.stderr.read()
            if process.stderr is not None
            else ""
        )
        return_code = process.wait()
    except BaseException:
        process.terminate()
        process.wait()
        raise

    if return_code != 0:
        raise RuntimeError(
            f"Could not create selected full-resolution video "
            f"using {resolved_encoder}: {stderr.strip()}"
        )

    output_info = inspect_video(output)
    frame_difference = abs(
        output_info.frame_count - selected_frame_count
    )
    duration_difference = abs(
        output_info.duration_seconds - selected_duration
    )
    duration_tolerance = max(
        0.25,
        2 / fps,
    )

    if frame_difference > 2:
        raise RuntimeError(
            f"Final video has {output_info.frame_count} frames; "
            f"expected approximately {selected_frame_count}"
        )

    if duration_difference > duration_tolerance:
        raise RuntimeError(
            f"Final video duration is "
            f"{output_info.duration_seconds:.3f}s; "
            f"expected approximately {selected_duration:.3f}s"
        )

    if progress_callback is not None:
        progress_callback(
            selected_duration,
            selected_duration,
        )

    return output

