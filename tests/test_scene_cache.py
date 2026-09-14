import json
from pathlib import Path

import cv2
import numpy as np

from src.subject_reframe.core import CropConfig, VideoInfo
from src.subject_reframe.detection import detect_scenes
from src.subject_reframe.rendering import render_video


def _write_test_video(video_path: Path, frame_count: int = 40) -> None:
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (64, 64),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create test video: {video_path}")

    for index in range(frame_count):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        if index < frame_count // 2:
            frame[:, :, 0] = 255
        else:
            frame[:, :, 2] = 255
        writer.write(frame)
    writer.release()


def test_detect_scenes_caches_by_input_md5(tmp_path):
    video_path = tmp_path / "synthetic.mp4"
    cache_path = tmp_path / "scene_cache.json"
    _write_test_video(video_path)

    scenes = detect_scenes(
        video_path,
        cache_path=cache_path,
        scan_stride=1,
        min_scene_seconds=0.1,
        histogram_correlation_threshold=0.1,
        mean_difference_threshold=1.0,
        extreme_difference_threshold=10.0,
    )

    assert scenes
    assert cache_path.exists()

    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert payload["md5"]
    assert payload["scenes"]

    cached_again = detect_scenes(
        video_path,
        cache_path=cache_path,
        scan_stride=1,
        min_scene_seconds=0.1,
        histogram_correlation_threshold=0.1,
        mean_difference_threshold=1.0,
        extreme_difference_threshold=10.0,
    )

    assert cached_again == scenes


def test_render_video_checkpointing_resumes_by_scene(tmp_path):
    video_path = tmp_path / "synthetic.mp4"
    output_path = tmp_path / "rendered.mp4"
    checkpoint_dir = tmp_path / "checkpoints"
    _write_test_video(video_path, frame_count=30)

    info = VideoInfo(
        path=str(video_path),
        width=64,
        height=64,
        fps=10.0,
        frame_count=30,
        duration_seconds=3.0,
    )
    config = CropConfig(output_width=32, output_height=32)
    plans = []
    for frame_index in range(30):
        plans.append({
            "mode": "crop",
            "frame": frame_index,
            "crop": (0.0, 0.0, 64.0, 64.0),
        })

    rendered = render_video(
        video_path,
        output_path,
        plans,
        info,
        config,
        checkpoint_dir=checkpoint_dir,
    )

    assert rendered == output_path
    assert output_path.exists()
    assert checkpoint_dir.exists()
    assert any(checkpoint_dir.iterdir())

    second_render = render_video(
        video_path,
        output_path,
        plans,
        info,
        config,
        checkpoint_dir=checkpoint_dir,
    )

    assert second_render == output_path
    assert output_path.exists()
