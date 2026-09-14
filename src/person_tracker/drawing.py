# src/person_tracker/drawing.py

import cv2

TRACK_COL_WIDTH = 72
OTHER_COL_WIDTH = 58

TRACKING_COLUMNS = [
    ("Track", TRACK_COL_WIDTH),
    ("Person", OTHER_COL_WIDTH),
    ("Identity", OTHER_COL_WIDTH),
    ("Face", OTHER_COL_WIDTH),
]

def build_face_samples_by_track(face_samples):
    samples_by_track = {}

    for sample in face_samples:
        samples_by_track.setdefault(
            sample.track_id,
            [],
        ).append(
            (sample.frame_no, sample.confidence)
        )

    for samples in samples_by_track.values():
        samples.sort()

    return samples_by_track

def get_last_face_confidence(
    face_samples_by_track,
    track_id,
    frame_no,
    max_age_frames=None,
):
    samples = face_samples_by_track.get(track_id)

    if not samples:
        return None, None

    last_frame = None
    last_confidence = None

    for sample_frame, confidence in samples:
        if sample_frame > frame_no:
            break

        last_frame = sample_frame
        last_confidence = confidence

    if last_frame is None:
        return None, None

    age = frame_no - last_frame

    if (
        max_age_frames is not None
        and age > max_age_frames
    ):
        return None, None

    return last_confidence, age

def put_text(
    frame,
    text,
    x,
    y,
    font_scale=1,
    thickness=2,
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_h, frame_w = frame.shape[:2]

    (w, h), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    x = max(2, min(x, frame_w - w - 12))
    y = max(h + 10, min(y, frame_h - baseline - 6))

    cv2.rectangle(
        frame,
        (x, y - h - 10),
        (x + w + 10, y + baseline + 5),
        (20, 20, 20),
        -1,
    )

    cv2.rectangle(
        frame,
        (x, y - h - 10),
        (x + w + 10, y + baseline + 5),
        (0, 255, 0),
        1,
    )

    cv2.putText(
        frame,
        text,
        (x + 5, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

def get_tracking_label_width():
    return sum(
        width
        for _, width in TRACKING_COLUMNS
    )

def put_tracking_label(
    frame,
    x,
    y,
    values,
    font_scale=0.75,
    thickness=2,
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_h, frame_w = frame.shape[:2]

    (_, text_height), baseline = cv2.getTextSize(
        "999",
        font,
        font_scale,
        thickness,
    )

    box_width = get_tracking_label_width()
    box_height = text_height + 16

    x = max(
        2,
        min(
            x,
            frame_w - box_width - 2,
        ),
    )

    if y - box_height < 2:
        y = box_height + 2

    y = min(
        y,
        frame_h - baseline - 5,
    )

    cv2.rectangle(
        frame,
        (x, y - box_height),
        (x + box_width, y + 4),
        (20, 20, 20),
        -1,
    )

    cv2.rectangle(
        frame,
        (x, y - box_height),
        (x + box_width, y + 4),
        (0, 255, 0),
        1,
    )

    offset = 0

    for index, ((_, width), value) in enumerate(
        zip(TRACKING_COLUMNS, values)
    ):
        column_x = x + offset

        cv2.putText(
            frame,
            str(value),
            (column_x + 7, y - 4),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        if index < len(values) - 1:
            pipe_x = column_x + width - 10

            cv2.putText(
                frame,
                "|",
                (pipe_x, y - 4),
                font,
                font_scale,
                (130, 130, 130),
                1,
                cv2.LINE_AA,
            )

        offset += width

def put_tracking_legend(
    frame,
    margin=15,
    margin_top=15,
    font_scale=0.82,
    thickness=2,
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    _, frame_w = frame.shape[:2]

    labels = [
        "Track",
        "Person",
        "Identity",
        "Face",
    ]

    column_widths = [
        90,
        95,
        110,
        80,
    ]

    box_width = sum(column_widths)

    (_, text_height), _ = cv2.getTextSize(
        "Identity",
        font,
        font_scale,
        thickness,
    )

    box_height = text_height + 18

    x = max(
        0,
        frame_w - box_width - margin,
    )

    y = margin_top + box_height

    cv2.rectangle(
        frame,
        (x, y - box_height),
        (x + box_width, y + 5),
        (20, 20, 20),
        -1,
    )

    cv2.rectangle(
        frame,
        (x, y - box_height),
        (x + box_width, y + 5),
        (0, 255, 0),
        2,
    )

    offset = 0

    for index, (label, width) in enumerate(
        zip(labels, column_widths)
    ):
        column_x = x + offset

        cv2.putText(
            frame,
            label,
            (column_x + 8, y - 5),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        if index < len(labels) - 1:
            pipe_x = column_x + width - 12

            cv2.putText(
                frame,
                "|",
                (pipe_x, y - 5),
                font,
                font_scale,
                (160, 160, 160),
                thickness,
                cv2.LINE_AA,
            )

        offset += width