def fit_aspect_box(
    bbox,
    aspect_ratio=(9, 16),
    horizontal_padding=0.0,
    vertical_padding=0.0,
):
    """
    Return the smallest box with the requested aspect ratio containing
    the tracked person plus optional horizontal and vertical padding.

    Args:
        bbox (tuple[int, int, int, int]):
            Tracking box as ``(x1, y1, x2, y2)``.

        aspect_ratio (tuple[int, int], optional):
            Required output aspect ratio as ``(width, height)``.
            Defaults to ``(9, 16)``.

        horizontal_padding (float, optional):
            Additional horizontal space on each side as a proportion of
            the tracking-box width. For example, 0.20 adds 20% of the
            tracking width to both the left and right sides.
            Defaults to 0.0.

        vertical_padding (float, optional):
            Additional vertical space on each side as a proportion of
            the tracking-box height. For example, 0.20 adds 20% of the
            tracking height above and below the person.
            Defaults to 0.0.

    Returns:
        tuple[int, int, int, int]:
            Aspect-ratio box as ``(x1, y1, x2, y2)``.
    """
    x1, y1, x2, y2 = bbox
    track_w = x2 - x1
    track_h = y2 - y1

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    padded_w = track_w * (1 + 2 * horizontal_padding)
    padded_h = track_h * (1 + 2 * vertical_padding)

    target_ratio = aspect_ratio[0] / aspect_ratio[1]
    padded_ratio = padded_w / padded_h

    if padded_ratio >= target_ratio:
        outer_w = padded_w
        outer_h = outer_w / target_ratio
    else:
        outer_h = padded_h
        outer_w = outer_h * target_ratio

    return (
        round(cx - outer_w / 2),
        round(cy - outer_h / 2),
        round(cx + outer_w / 2),
        round(cy + outer_h / 2),
    )
