"""Interactive Jupyter widgets and visualizations."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .core import Scene
from .io import inspect_video, read_frame
from .tracking import rank_scene_tracks, choose_candidate_frames, _annotate_track_ids


def _encode_jpeg(frame: np.ndarray, *, quality: int = 88) -> bytes:
    """Encode a frame as JPEG bytes for Jupyter display."""
    ok, encoded = cv2.imencode(
        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        raise RuntimeError("Could not encode a Jupyter preview image")
    return encoded.tobytes()


def _track_thumbnail(
    video_path: str | Path,
    ranked_track: Mapping[str, Any],
    *,
    padding: float = 0.14,
) -> bytes:
    """Extract and annotate a track thumbnail with green body box."""
    frame = read_frame(video_path, int(ranked_track["best_frame"]))
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = [float(value) for value in ranked_track["best_xyxy"]]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    ix1 = max(0, math.floor(x1 - width * padding))
    iy1 = max(0, math.floor(y1 - height * padding))
    ix2 = min(frame_width, math.ceil(x2 + width * padding))
    iy2 = min(frame_height, math.ceil(y2 + height * padding))
    crop = frame[iy1:iy2, ix1:ix2]
    if crop.size == 0:
        crop = frame
        ix1 = 0
        iy1 = 0

    # A crop can contain adjacent people. Mark the exact body associated with
    # this Track ID so the selection card cannot be mistaken for a bystander.
    crop = crop.copy()
    crop_height, crop_width = crop.shape[:2]
    target_x1 = min(crop_width - 1, max(0, round(x1 - ix1)))
    target_y1 = min(crop_height - 1, max(0, round(y1 - iy1)))
    target_x2 = min(crop_width - 1, max(0, round(x2 - ix1)))
    target_y2 = min(crop_height - 1, max(0, round(y2 - iy1)))
    thickness = max(3, round(min(crop_width, crop_height) / 80))
    color = (40, 230, 70)
    cv2.rectangle(
        crop,
        (target_x1, target_y1),
        (target_x2, target_y2),
        color,
        thickness,
    )
    label = f"TRACK {int(ranked_track['track_id'])}"
    font_scale = max(0.65, min(1.6, crop_width / 360))
    text_thickness = max(2, round(font_scale * 2))
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
    )
    label_x = max(0, min(target_x1, crop_width - text_width - 12))
    label_y = max(text_height + baseline + 10, target_y1)
    cv2.rectangle(
        crop,
        (label_x, label_y - text_height - baseline - 10),
        (min(crop_width - 1, label_x + text_width + 12), label_y + 4),
        color,
        -1,
    )
    cv2.putText(
        crop,
        label,
        (label_x + 6, label_y - baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        text_thickness,
        cv2.LINE_AA,
    )
    return _encode_jpeg(crop)


async def select_scene_tracks_widget(
    video_path: str | Path,
    scene: Scene,
    records: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    frame_stride: int,
    top_n: int = 5,
    columns: int = 3,
    allow_multiple: bool = True,
    override_seconds: Optional[Mapping[int, float]] = None,
    initial_selection: Optional[int | Sequence[int]] = None,
    output_widget: Optional[Any] = None,
) -> Dict[str, Any]:
    """Wait asynchronously for a clickable Jupyter track-card selection."""
    import asyncio

    import ipywidgets as widgets
    from IPython.display import display

    try:
        from ipyevents import Event
    except ImportError:
        Event = None

    if top_n < 1:
        raise ValueError("top_n must be at least 1")
    if columns < 1:
        raise ValueError("columns must be at least 1")

    info = inspect_video(video_path)
    ranked = rank_scene_tracks(
        scene,
        records,
        frame_stride=frame_stride,
        frame_width=info.width,
        frame_height=info.height,
    )
    initial_ids = (
        set()
        if initial_selection is None
        else {int(initial_selection)}
        if isinstance(initial_selection, int)
        else {int(value) for value in initial_selection}
    )

    candidate_frame = choose_candidate_frames(
        [scene], records, override_seconds=override_seconds
    )[scene.index]
    overview_frame = read_frame(video_path, candidate_frame)
    overview_frame = _annotate_track_ids(
        overview_frame, records.get(candidate_frame, ())
    )
    overview = widgets.Image(
        value=_encode_jpeg(overview_frame),
        format="jpeg",
        layout=widgets.Layout(width="100%", max_width="1100px", height="auto"),
    )

    header = widgets.HTML(
        value=(
            f"<h2>Scene {scene.index}: select the target</h2>"
            f"<p>{scene.start_seconds:.2f}s–{scene.end_seconds:.2f}s. "
            f"Showing the top {min(top_n, len(ranked))} of {len(ranked)} tracks. "
            f"Each card uses that track's clearest frame from this scene; "
            f"the green box marks the tracked body.</p>"
        )
    )
    status = widgets.HTML(value="<b>Click a person image or its Track ID button.</b>")
    grid = widgets.GridBox(
        layout=widgets.Layout(
            width="100%",
            grid_template_columns=f"repeat({columns}, minmax(220px, 280px))",
            grid_gap="14px",
        )
    )
    card_by_id: dict[int, widgets.VBox] = {}
    toggle_by_id: dict[int, widgets.ToggleButton] = {}
    image_events: list[Any] = []
    updating = False

    def refresh_visual_state() -> None:
        selected = [track_id for track_id, toggle in toggle_by_id.items() if toggle.value]
        for track_id, card in card_by_id.items():
            toggle_by_id[track_id].button_style = (
                "success" if toggle_by_id[track_id].value else ""
            )
            card.layout.border = (
                "4px solid #2e7d32" if toggle_by_id[track_id].value else "1px solid #bdbdbd"
            )
        status.value = (
            f"<b>Selected Track IDs: {', '.join(map(str, sorted(selected)))}</b>"
            if selected
            else "<b>No track selected.</b> Use 'Target absent' if appropriate."
        )

    def on_toggle(change: Mapping[str, Any], track_id: int) -> None:
        nonlocal updating
        if updating:
            return
        if change.get("new") and not allow_multiple:
            updating = True
            for other_id, other_toggle in toggle_by_id.items():
                if other_id != track_id:
                    other_toggle.value = False
            updating = False
        refresh_visual_state()

    def create_card(
        item: Mapping[str, Any], thumbnail: bytes | None = None
    ) -> widgets.VBox:
        track_id = int(item["track_id"])
        image_widget = widgets.Image(
            value=thumbnail if thumbnail is not None else _track_thumbnail(video_path, item),
            format="jpeg",
            layout=widgets.Layout(
                width="240px",
                height="320px",
                object_fit="contain",
                cursor="pointer",
            ),
        )
        toggle = widgets.ToggleButton(
            value=track_id in initial_ids,
            description=f"Track ID {track_id}",
            tooltip=f"Select Track ID {track_id}",
            button_style="success" if track_id in initial_ids else "",
            layout=widgets.Layout(width="240px", height="42px"),
        )
        details = widgets.HTML(
            value=(
                f"<div style='text-align:center'>"
                f"Video time <b>{item['best_frame'] / scene.fps:.2f}s</b><br>"
                f"Coverage <b>{item['coverage_percent']:.1f}%</b><br>"
                f"Confidence <b>{item['mean_confidence']:.3f}</b><br>"
                f"Ranking <b>{item['ranking_score']:.3f}</b>"
                f"</div>"
            )
        )
        card = widgets.VBox(
            [image_widget, toggle, details],
            layout=widgets.Layout(
                align_items="center",
                padding="8px",
                border="1px solid #bdbdbd",
                width="260px",
            ),
        )
        toggle_by_id[track_id] = toggle
        card_by_id[track_id] = card
        toggle.observe(lambda change, value=track_id: on_toggle(change, value), names="value")
        if Event is not None:
            event = Event(source=image_widget, watched_events=["click"])

            def image_clicked(_: Mapping[str, Any], value: int = track_id) -> None:
                toggle_widget = toggle_by_id[value]
                if not toggle_widget.disabled:
                    toggle_widget.value = not toggle_widget.value

            event.on_dom_event(image_clicked)
            image_events.append(event)
        return card

    visible_ids = [int(item["track_id"]) for item in ranked[:top_n]]
    visible_ids.extend(sorted(initial_ids - set(visible_ids)))
    visible_ids = list(dict.fromkeys(visible_ids))
    ranked_by_id = {int(item["track_id"]): item for item in ranked}
    for track_id in visible_ids:
        if track_id in ranked_by_id:
            create_card(ranked_by_id[track_id])
    grid.children = tuple(card_by_id[track_id] for track_id in visible_ids if track_id in card_by_id)

    show_all_button = widgets.Button(
        description=f"Show all {len(ranked)} tracks",
        icon="th",
        disabled=len(ranked) <= len(visible_ids),
    )
    clear_button = widgets.Button(description="Clear selection", icon="eraser")
    confirm_button = widgets.Button(
        description="Confirm selection", button_style="success", icon="check"
    )
    absent_button = widgets.Button(
        description="Target absent", button_style="warning", icon="eye-slash"
    )
    retrack_button = widgets.Button(
        description="Retrack scene", button_style="danger", icon="refresh"
    )
    controls = widgets.HBox(
        [show_all_button, clear_button, confirm_button, absent_button, retrack_button],
        layout=widgets.Layout(flex_flow="row wrap"),
    )
    container = widgets.VBox([header, overview, status, grid, controls])

    loop = asyncio.get_running_loop()
    result_future: asyncio.Future[dict[str, Any]] = loop.create_future()
    show_all_task: asyncio.Task[None] | None = None

    def finish(result: dict[str, Any], message: str) -> None:
        if result_future.done():
            return
        for toggle in toggle_by_id.values():
            toggle.disabled = True
        for button in (
            show_all_button,
            clear_button,
            confirm_button,
            absent_button,
            retrack_button,
        ):
            button.disabled = True
        for event in image_events:
            event.watched_events = []
        overview.layout.display = "none"
        grid.layout.display = "none"
        controls.layout.display = "none"
        header.value = f"<h3>Scene {scene.index} completed</h3>"
        status.value = f"<b>{message}</b>"
        result_future.set_result(result)

    async def expand_all_tracks() -> None:
        remaining = [
            item for item in ranked if int(item["track_id"]) not in card_by_id
        ]
        loaded = len(card_by_id)
        try:
            for item in remaining:
                if result_future.done():
                    return
                show_all_button.description = f"Loading {loaded}/{len(ranked)}"
                show_all_button.icon = "spinner"
                status.value = (
                    f"<b>Loading all track previews: {loaded} of {len(ranked)}…</b>"
                )
                await asyncio.sleep(0.05)
                thumbnail = await asyncio.to_thread(_track_thumbnail, video_path, item)
                if result_future.done():
                    return
                card = create_card(item, thumbnail)
                grid.children = (*grid.children, card)
                loaded += 1
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            return
        except Exception as error:
            status.value = (
                f"<b style='color:#b71c1c'>Could not load all previews: {error}</b>"
            )
            show_all_button.disabled = False
            show_all_button.description = f"Show all {len(ranked)} tracks"
            show_all_button.icon = "th"
            return
        refresh_visual_state()
        show_all_button.description = f"All {len(ranked)} shown"
        show_all_button.icon = "check"

    def show_all(_: widgets.Button) -> None:
        nonlocal show_all_task
        show_all_button.disabled = True
        show_all_task = loop.create_task(expand_all_tracks())

    def clear_selection(_: widgets.Button) -> None:
        for toggle in toggle_by_id.values():
            toggle.value = False
        refresh_visual_state()

    def confirm(_: widgets.Button) -> None:
        selected = sorted(
            track_id for track_id, toggle in toggle_by_id.items() if toggle.value
        )
        if not selected:
            status.value = "<b style='color:#b71c1c'>Select a track or use 'Target absent'.</b>"
            return
        finish(
            {"action": "confirm", "track_ids": selected},
            f"Confirmed Track IDs: {', '.join(map(str, selected))}",
        )

    show_all_button.on_click(show_all)
    clear_button.on_click(clear_selection)
    confirm_button.on_click(confirm)
    absent_button.on_click(
        lambda _: finish({"action": "absent", "track_ids": []}, "Confirmed: target absent")
    )
    retrack_button.on_click(
        lambda _: finish({"action": "retrack", "track_ids": []}, "Retracking requested")
    )

    refresh_visual_state()
    if output_widget is None:
        display(container)
    else:
        with output_widget:
            output_widget.clear_output(wait=True)
            display(container)
    return await result_future


def show_scene_midpoints(
    video_path: str | Path,
    scenes: Sequence[Scene],
    *,
    scenes_per_page: int = 9,
    columns: int = 3,
) -> None:
    """Show representative midpoint frames in paginated scene-review grids."""
    if scenes_per_page < 1:
        raise ValueError("scenes_per_page must be at least 1")
    if columns < 1:
        raise ValueError("columns must be at least 1")

    for page_start in range(0, len(scenes), scenes_per_page):
        page = scenes[page_start : page_start + scenes_per_page]
        rows = math.ceil(len(page) / columns)
        figure, axes = plt.subplots(rows, columns, figsize=(5.5 * columns, 4 * rows))
        axes_array = np.atleast_1d(axes).reshape(-1)
        for axis, scene in zip(axes_array, page):
            midpoint = scene.start_frame + max(0, scene.frame_count - 1) // 2
            frame = read_frame(video_path, midpoint)
            axis.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            axis.set_title(
                f"Scene {scene.index}\n"
                f"{scene.start_seconds:.2f}s–{scene.end_seconds:.2f}s "
                f"({scene.end_seconds - scene.start_seconds:.2f}s)",
                fontsize=13,
                fontweight="bold",
            )
            axis.axis("off")
        for axis in axes_array[len(page) :]:
            axis.axis("off")
        figure.suptitle(
            f"Scene review — {page_start + 1} to {page_start + len(page)} of {len(scenes)}",
            fontsize=18,
            fontweight="bold",
        )
        figure.tight_layout()
        plt.show()


def show_scene_candidates(
    video_path: str | Path,
    scenes: Sequence[Scene],
    records: Mapping[int, Sequence[Mapping[str, Any]]],
    candidate_frames: Mapping[int, int],
    *,
    person_columns: int = 3,
    crop_padding: float = 0.12,
) -> None:
    """Display a large scene view and readable person close-ups per scene."""
    if not scenes:
        print("No scenes to display")
        return
    if person_columns < 1:
        raise ValueError("person_columns must be at least 1")
    if crop_padding < 0:
        raise ValueError("crop_padding cannot be negative")

    for scene in scenes:
        frame_index = candidate_frames[scene.index]
        frame = read_frame(video_path, frame_index)
        detections = sorted(
            records.get(frame_index, ()), key=lambda item: float(item["xyxy"][0])
        )
        annotated = _annotate_track_ids(frame, detections)

        overview, overview_axis = plt.subplots(figsize=(16, 9))
        overview_axis.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        overview_axis.set_title(
            f"Scene {scene.index} — {frame_index / scene.fps:.2f}s "
            f"(frame {frame_index})",
            fontsize=18,
            fontweight="bold",
        )
        overview_axis.axis("off")
        overview.tight_layout()
        plt.show()

        if not detections:
            print(f"Scene {scene.index}: no tracked people in this candidate frame.")
            continue

        rows = math.ceil(len(detections) / person_columns)
        closeups, axes = plt.subplots(
            rows,
            person_columns,
            figsize=(5 * person_columns, 5 * rows),
        )
        axes_array = np.atleast_1d(axes).reshape(-1)
        frame_height, frame_width = frame.shape[:2]

        for axis, detection in zip(axes_array, detections):
            x1, y1, x2, y2 = [float(value) for value in detection["xyxy"]]
            width = x2 - x1
            height = y2 - y1
            padded_x1 = max(0, math.floor(x1 - width * crop_padding))
            padded_y1 = max(0, math.floor(y1 - height * crop_padding))
            padded_x2 = min(frame_width, math.ceil(x2 + width * crop_padding))
            padded_y2 = min(frame_height, math.ceil(y2 + height * crop_padding))
            crop = frame[padded_y1:padded_y2, padded_x1:padded_x2]
            axis.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            axis.set_title(
                f"Track ID {int(detection['track_id'])}",
                fontsize=18,
                fontweight="bold",
                color="darkgreen",
            )
            axis.axis("off")

        for axis in axes_array[len(detections) :]:
            axis.axis("off")
        closeups.suptitle(
            f"Scene {scene.index}: choose the target Track ID",
            fontsize=20,
            fontweight="bold",
        )
        closeups.tight_layout()
        plt.show()
