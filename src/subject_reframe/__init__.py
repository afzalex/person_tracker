"""Subject Reframe video tracking and cropping package."""

# Core data structures
from .core import CropConfig, Scene, VideoInfo

# File I/O and caching
from .io import (
    build_tracking_cache_metadata,
    ffmpeg_available,
    inspect_video,
    load_scene_selections,
    load_tracking_records,
    mux_original_audio,
    read_frame,
    save_report,
    save_scene_selections,
    save_tracking_records,
)

# Scene detection
from .detection import detect_scenes, print_scene_table

# Person tracking and ranking
from .tracking import (
    PersonTrackingSession,
    choose_candidate_frames,
    print_scene_track_summary,
    rank_scene_tracks,
    summarize_scene_tracks,
    track_all_people,
)

# Crop planning and smoothing
from .cropping import build_crop_plan, print_warning_summary, validate_target_tracks

# Video rendering
from .rendering import render_video, save_preview

# Interactive widgets and visualization
from .widgets import (
    select_scene_tracks_widget,
    show_scene_candidates,
    show_scene_midpoints,
)

__all__ = [
    "CropConfig",
    "PersonTrackingSession",
    "Scene",
    "VideoInfo",
    "build_tracking_cache_metadata",
    "build_crop_plan",
    "choose_candidate_frames",
    "detect_scenes",
    "ffmpeg_available",
    "inspect_video",
    "load_tracking_records",
    "load_scene_selections",
    "mux_original_audio",
    "print_scene_table",
    "print_scene_track_summary",
    "print_warning_summary",
    "render_video",
    "save_preview",
    "save_report",
    "save_tracking_records",
    "save_scene_selections",
    "show_scene_midpoints",
    "show_scene_candidates",
    "rank_scene_tracks",
    "select_scene_tracks_widget",
    "track_all_people",
    "summarize_scene_tracks",
    "validate_target_tracks",
]
