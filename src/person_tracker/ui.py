import asyncio

import cv2
import ipywidgets as widgets
from ipycanvas import Canvas
from IPython.display import display

from person_tracker.io import read_nth_frame

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


