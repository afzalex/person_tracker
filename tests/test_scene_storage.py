from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from src.person_tracker.core import VideoInfo
from src.person_tracker.scene_detection import (
    SceneCut,
    build_scene_records,
    combine_scene_cuts,
    create_scene_review_video,
    create_selected_scene_review_video,
    detect_scene_cuts,
    resolve_scene_window,
)
from src.person_tracker.scene_storage import (
    build_scene_manifest,
    load_scene_run,
    save_scene_run,
)


def _write_scene_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64)
    )
    assert writer.isOpened()
    for color in ((255, 0, 0), (0, 0, 255), (0, 255, 0)):
        for _ in range(10):
            writer.write(np.full((64, 64, 3), color, dtype=np.uint8))
    writer.release()


def test_scene_detection_and_storage_round_trip(tmp_path):
    video_path = tmp_path / "source.mp4"
    _write_scene_video(video_path)
    info, cuts = detect_scene_cuts(
        video_path,
        scan_stride=1,
        min_scene_seconds=0.1,
        histogram_correlation_threshold=0.2,
        mean_difference_threshold=1.0,
        extreme_difference_threshold=10.0,
    )
    assert [cut.frame for cut in cuts] == [10, 20]

    raw_scenes = build_scene_records(info, cuts)
    accepted_cuts = combine_scene_cuts(
        cuts,
        fps=info.fps,
        frame_count=info.frame_count,
        manual_cut_seconds=(0.5,),
        remove_cut_frames=(20,),
    )
    scenes = build_scene_records(info, accepted_cuts, disabled_scene_ids=("scene-000002",))
    manifest = build_scene_manifest(info, settings={"scan_stride": 1})
    run_directory = tmp_path / "scene-run"
    save_scene_run(
        run_directory,
        manifest=manifest,
        raw_scenes=raw_scenes,
        scenes=scenes,
    )

    loaded_manifest, loaded_raw, loaded_scenes = load_scene_run(run_directory)
    assert loaded_manifest["source_video"] == str(video_path.resolve())
    assert [scene.start_frame for scene in loaded_raw] == [0, 10, 20]
    assert [scene.start_frame for scene in loaded_scenes] == [0, 5, 10]
    assert loaded_scenes[1].enabled is False


def test_scene_detection_window_and_progress_callback(tmp_path):
    video_path = tmp_path / "source.mp4"
    _write_scene_video(video_path)
    progress = []

    _, cuts = detect_scene_cuts(
        video_path,
        start_frame=11,
        end_frame=25,
        scan_stride=1,
        min_scene_seconds=0.1,
        histogram_correlation_threshold=0.2,
        mean_difference_threshold=1.0,
        extreme_difference_threshold=10.0,
        progress_callback=lambda done, total: progress.append((done, total)),
    )

    assert [cut.frame for cut in cuts] == [20]
    assert progress[0] == (0, 13)
    assert progress[-1] == (13, 13)


def test_scene_detection_rejects_conflicting_window_units(tmp_path):
    video_path = tmp_path / "source.mp4"
    _write_scene_video(video_path)

    try:
        detect_scene_cuts(video_path, start_frame=0, start_seconds=0.0)
    except ValueError as error:
        assert "start_frame or start_seconds" in str(error)
    else:
        raise AssertionError("Expected conflicting start boundary values to fail")


def test_scene_records_cover_entire_video():
    info = VideoInfo(
        path="unused.mp4",
        width=1920,
        height=1080,
        fps=30.0,
        frame_count=300,
        duration_seconds=10.0,
    )
    scenes = build_scene_records(
        info,
        [SceneCut(75, 80.0, 0.1, 45.0), SceneCut(210, 90.0, 0.0, 50.0)],
    )
    assert [(scene.start_frame, scene.end_frame) for scene in scenes] == [
        (0, 75), (75, 210), (210, 300)
    ]


def test_scene_window_applies_to_records_storage_and_review(tmp_path):
    video_path = tmp_path / "source.mp4"
    _write_scene_video(video_path)
    info, cuts = detect_scene_cuts(
        video_path,
        start_seconds=1.0,
        end_seconds=3.0,
        scan_stride=1,
        min_scene_seconds=0.1,
        histogram_correlation_threshold=0.2,
        mean_difference_threshold=1.0,
        extreme_difference_threshold=10.0,
    )
    start_frame, end_frame = resolve_scene_window(
        info, start_seconds=1.0, end_seconds=3.0
    )
    scenes = build_scene_records(
        info, cuts, start_frame=start_frame, end_frame=end_frame
    )
    assert [(scene.start_frame, scene.end_frame) for scene in scenes] == [
        (10, 20), (20, 30)
    ]

    manifest = build_scene_manifest(info, settings={})
    manifest["window_start_frame"] = start_frame
    manifest["window_end_frame"] = end_frame
    run_directory = tmp_path / "windowed-run"
    save_scene_run(
        run_directory, manifest=manifest, raw_scenes=scenes, scenes=scenes
    )
    _, _, loaded = load_scene_run(run_directory)
    assert loaded[0].start_frame == 10
    assert loaded[-1].end_frame == 30

    review_path = create_scene_review_video(
        video_path, scenes, tmp_path / "review.mp4", width=320
    )
    capture = cv2.VideoCapture(str(review_path))
    review_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    review_fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    assert abs(review_frames / review_fps - 2.0) < 0.2
    subtitles = (tmp_path / "review.srt").read_text()
    assert "00:00:00,000 --> 00:00:01,000" in subtitles


def test_selected_scene_review_omits_disabled_scenes(tmp_path):
    video_path = tmp_path / "source.mp4"
    _write_scene_video(video_path)
    info, cuts = detect_scene_cuts(
        video_path,
        scan_stride=1,
        min_scene_seconds=0.1,
        histogram_correlation_threshold=0.2,
        mean_difference_threshold=1.0,
        extreme_difference_threshold=10.0,
    )
    scenes = build_scene_records(info, cuts)
    scenes[1] = replace(scenes[1], enabled=False)

    review_path = create_selected_scene_review_video(
        video_path, scenes, tmp_path / "selected-review.mp4", width=320
    )
    capture = cv2.VideoCapture(str(review_path))
    review_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    review_fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.set(cv2.CAP_PROP_POS_MSEC, 1500)
    ok, frame = capture.read()
    capture.release()

    assert abs(review_frames / review_fps - 2.0) < 0.2
    assert ok
    assert frame[:, :, 1].mean() > frame[:, :, 0].mean()
    assert frame[:, :, 1].mean() > frame[:, :, 2].mean()
    subtitles = (tmp_path / "selected-review.srt").read_text()
    assert "00:00:00,000 --> 00:00:01,000" in subtitles
    assert "00:00:01,000 --> 00:00:02,000" in subtitles
    assert "Scene 000002" not in subtitles
