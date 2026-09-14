import json

import numpy as np

from src.person_tracker.face import FaceSample
from src.person_tracker.storage import (
    build_run_manifest,
    load_observation_run,
    save_identity_resolution,
    save_observation_run,
)


def test_observation_and_identity_storage_round_trip(tmp_path):
    video_path = tmp_path / "source.mp4"
    video_path.write_bytes(b"source fingerprint fixture")
    run_directory = tmp_path / "run"

    manifest = build_run_manifest(
        video_path=video_path,
        width=1920,
        height=1080,
        fps=30.0,
        frame_count=300,
        start_frame=30,
        end_frame=90,
        settings={"tracker": "deepocsort.yaml", "complete": True},
    )
    tracking_history = {
        30: [{
            "track_id": 7,
            "bbox": np.array([10, 20, 110, 220]),
            "confidence": 0.91,
        }],
        31: [],
    }
    face_samples = [FaceSample(
        frame_no=30,
        track_id=7,
        bbox=(20, 30, 60, 80),
        confidence=0.95,
        quality=0.83,
        embedding=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        image=np.full((8, 8, 3), 127, dtype=np.uint8),
    )]

    save_observation_run(
        run_directory,
        manifest=manifest,
        tracking_history=tracking_history,
        face_samples=face_samples,
        save_face_crops=True,
    )
    loaded_manifest, loaded_tracks, loaded_faces = load_observation_run(run_directory)

    assert loaded_manifest["start_frame"] == 30
    assert loaded_manifest["end_frame"] == 90
    assert loaded_tracks[30][0]["track_id"] == 7
    np.testing.assert_array_equal(loaded_tracks[30][0]["bbox"], [10, 20, 110, 220])
    assert len(loaded_faces) == 1
    assert loaded_faces[0].frame_no == 30
    assert loaded_faces[0].image.size > 0
    np.testing.assert_allclose(loaded_faces[0].embedding, [0.1, 0.2, 0.3])

    save_identity_resolution(
        run_directory,
        identity_history={30: {7: {"person_id": 1, "confidence": 0.9}}},
        switch_boundaries={},
        summary={"person_count": 1},
    )
    identity = json.loads((run_directory / "identities.jsonl").read_text())
    assert identity["frame"] == 30
    assert identity["track_id"] == 7
    assert identity["person_id"] == 1
