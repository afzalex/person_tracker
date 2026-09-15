import asyncio
import json
from pathlib import Path

import cv2
import ipywidgets as widgets
from ipycanvas import Canvas
from IPython.display import display

from person_tracker.io import read_nth_frame


def load_selected_person(selection_path):
    """Return a persisted person ID, or ``None`` when no selection exists."""
    path = Path(selection_path).expanduser().resolve()
    if not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    person_id = payload.get("person_id")
    if person_id is None:
        raise ValueError(f"Selection file has no person_id: {path}")
    return int(person_id)


def save_selected_person(selection_path, person_id):
    """Persist a selected logical person ID for later/headless notebooks."""
    path = Path(selection_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"person_id": int(person_id)}
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)
    return path


def create_person_selection_widget(
    person_galleries,
    person_stats,
    *,
    selection_path=None,
    initial_person_id=None,
    fps=None,
    columns=5,
    card_width=180,
    image_height=140,
):
    """Create reusable person cards and persist clicks when a path is supplied."""
    available_ids = sorted(set(person_galleries) & set(person_stats))
    if not available_ids:
        raise ValueError("No resolved people with face galleries are available")

    if initial_person_id is None and selection_path is not None:
        initial_person_id = load_selected_person(selection_path)
    if initial_person_id is not None:
        initial_person_id = int(initial_person_id)
        if initial_person_id not in available_ids:
            initial_person_id = None

    state = {"person_id": initial_person_id}
    buttons = {}
    status = widgets.HTML()

    def format_duration(seconds):
        minutes, seconds = divmod(int(seconds), 60)
        return f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"

    def update_status(person_id):
        suffix = ""
        if selection_path is not None:
            suffix = f"<br><small>Saved to {Path(selection_path)}</small>"
        status.value = f"<b>Selected Person {person_id}</b>{suffix}"

    def on_select(person_id):
        state["person_id"] = int(person_id)
        for candidate_id, button in buttons.items():
            button.button_style = "success" if candidate_id == person_id else ""
        if selection_path is not None:
            save_selected_person(selection_path, person_id)
        update_status(person_id)

    cards = []
    for person_id in available_ids:
        samples = person_galleries[person_id]
        best = max(samples, key=lambda sample: sample.quality)
        stats = person_stats[person_id]
        duration = stats.get("duration")
        if duration is None:
            if fps is None:
                raise ValueError("fps is required when person_stats has no duration")
            duration = len(stats.get("frames", ())) / float(fps)
        track_count = stats.get("track_count")
        if track_count is None:
            track_count = len(stats.get("tracks", ()))
        h, w = best.image.shape[:2]
        scale = image_height / max(h, 1)
        image = cv2.resize(best.image, (max(1, int(w * scale)), image_height))
        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            raise RuntimeError(f"Could not encode preview for Person {person_id}")

        preview = widgets.Box(
            [widgets.Image(value=encoded.tobytes(), format="jpg", height=image_height)],
            layout=widgets.Layout(
                width="100%", height=f"{image_height}px", display="flex",
                justify_content="center", align_items="center", overflow="hidden",
            ),
        )
        info = widgets.HTML(
            f"<div style='text-align:left; line-height:1.5'>"
            f"<b>Person {person_id}</b><br>"
            f"Visible: {format_duration(duration)}<br>"
            f"Tracks: {track_count}<br>"
            f"Faces: {len(samples)}<br>Quality: {best.quality:.2f}</div>"
        )
        button = widgets.Button(
            description=f"Select Person {person_id}",
            layout=widgets.Layout(width="100%"),
        )
        button.on_click(lambda _, pid=person_id: on_select(pid))
        buttons[person_id] = button
        cards.append(widgets.VBox(
            [preview, info, button],
            layout=widgets.Layout(
                width=f"{card_width}px", border="1px solid #aaa", padding="8px",
                margin="4px", overflow="hidden", box_sizing="border-box",
            ),
        ))

    grid = widgets.GridBox(
        cards,
        layout=widgets.Layout(
            grid_template_columns=f"repeat({columns}, {card_width}px)",
            grid_gap="10px",
        ),
    )
    if initial_person_id is None:
        status.value = "<b>Select the target person</b>"
    else:
        buttons[initial_person_id].button_style = "success"
        update_status(initial_person_id)

    return widgets.VBox([grid, status]), state

def create_frame_canvas(video_path, frame_no):
    frame = read_nth_frame(video_path, frame_no)
    height, width = frame.shape[:2]

    canvas = Canvas(width=width, height=height)
    canvas.put_image_data(frame, 0, 0)

    coords = widgets.Label(value="x=-, y=-")

    state = {
        "mouse_point": None,
        "selected_point": None,
    }

    def on_mouse_move(x, y):
        x = int(x)
        y = int(y)

        state["mouse_point"] = (x, y)
        coords.value = f"x={x}, y={y}"

    def on_mouse_down(x, y):
        state["selected_point"] = (int(x), int(y))

    canvas.on_mouse_move(on_mouse_move)
    canvas.on_mouse_down(on_mouse_down)

    return canvas, coords, state

async def select_person_widget(
    person_galleries,
    person_stats,
    columns=5,
    card_width=180,
    image_height=140,
):
    loop = asyncio.get_running_loop()
    future = loop.create_future()
    buttons = {}

    def format_duration(seconds):
        minutes, seconds = divmod(int(seconds), 60)
        return f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"

    def on_select(person_id):
        if future.done():
            return

        for pid, button in buttons.items():
            button.button_style = "success" if pid == person_id else ""

        status.value = f"<b>Selected Person {person_id}</b>"
        future.set_result(person_id)

    cards = []

    for person_id in sorted(person_galleries):
        samples = person_galleries[person_id]
        best = max(samples, key=lambda s: s.quality)
        stats = person_stats[person_id]

        h, w = best.image.shape[:2]
        scale = image_height / h

        image = cv2.resize(
            best.image,
            (max(1, int(w * scale)), image_height),
        )

        _, encoded = cv2.imencode(".jpg", image)

        preview = widgets.Box(
            [
                widgets.Image(
                    value=encoded.tobytes(),
                    format="jpg",
                    height=image_height,
                )
            ],
            layout=widgets.Layout(
                width="100%",
                height=f"{image_height}px",
                display="flex",
                justify_content="center",
                align_items="center",
                overflow="hidden",
            ),
        )

        info = widgets.HTML(
            f"""
            <div style="text-align:left; line-height:1.5">
                <b>Person {person_id}</b><br>
                Visible: {format_duration(stats["duration"])}<br>
                Tracks: {stats["track_count"]}<br>
                Faces: {len(samples)}<br>
                Quality: {best.quality:.2f}
            </div>
            """
        )

        button = widgets.Button(
            description=f"Select Person {person_id}",
            layout=widgets.Layout(width="100%"),
        )

        button.on_click(
            lambda _, pid=person_id: on_select(pid)
        )

        buttons[person_id] = button

        cards.append(
            widgets.VBox(
                [preview, info, button],
                layout=widgets.Layout(
                    width=f"{card_width}px",
                    border="1px solid #aaa",
                    padding="8px",
                    margin="4px",
                    overflow="hidden",
                    box_sizing="border-box",
                ),
            )
        )

    grid = widgets.GridBox(
        cards,
        layout=widgets.Layout(
            grid_template_columns=f"repeat({columns}, {card_width}px)",
            grid_gap="10px",
        ),
    )

    status = widgets.HTML("<b>Select the target person</b>")

    display(grid, status)

    return await future
