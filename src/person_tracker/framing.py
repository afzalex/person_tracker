from person_tracker.geometry import fit_aspect_box

import cv2
import numpy as np

class SmoothContainmentBox:
    """
    Smooth framing using velocity- and acceleration-limited camera motion.

    Args:
        fps (float):
            Video frame rate used to make camera motion time-based.
        aspect_ratio (tuple[int, int], optional):
            Required framing aspect ratio as ``(width, height)``.
        response_time (float, optional):
            Approximate time in seconds for the camera to respond to target
            movement. Higher values produce slower, smoother movement.
        visible_expansion (float, optional):
            Extra framing space around the visible target.
        transition_expansion (float, optional):
            Temporary framing expansion during disappearance/reappearance.
        containment_priority (float, optional):
            How strongly the camera favors keeping the current target inside
            the crop. ``0.0`` favors smoothness, ``1.0`` strongly favors
            containment. Motion limits are always respected.
        max_position_speed (float, optional):
            Maximum camera center speed in crop-widths/heights per second.
        max_zoom_speed (float, optional):
            Maximum crop-width change per second as a fraction of current width.
    """

    _CONTAINMENT_TOLERANCE = 0.02

    def __init__(
        self,
        fps,
        aspect_ratio=(9, 16),
        response_time=0.35,
        visible_expansion=0.10,
        transition_expansion=0.20,
        containment_priority=0.50,
        max_position_speed=1.25,
        max_zoom_speed=0.60,
    ):
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if response_time <= 0:
            raise ValueError(f"response_time must be positive, got {response_time}")

        self._ratio = aspect_ratio[0] / aspect_ratio[1]
        self._dt = 1.0 / fps
        self._response_time = response_time
        self._visible_expansion = visible_expansion
        self._transition_expansion = transition_expansion
        self._containment_priority = max(0.0, min(1.0, containment_priority))
        self._max_position_speed = max(0.0, max_position_speed)
        self._max_zoom_speed = max(0.0, max_zoom_speed)
        self.reset()

    def _target_state(self, bbox):
        """Return center and minimum aspect-ratio width required for a box."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        width = max(x2 - x1, (y2 - y1) * self._ratio)
        return cx, cy, width

    def _box(self):
        """Return the current framing box."""
        if self._cx is None:
            return None

        height = self._width / self._ratio

        return (
            round(self._cx - self._width / 2),
            round(self._cy - height / 2),
            round(self._cx + self._width / 2),
            round(self._cy + height / 2),
        )

    def _step_axis(self, value, velocity, target, max_speed, max_acceleration):
        """Move one camera axis toward a target with speed and acceleration limits."""
        error = target - value
        desired_velocity = error / self._response_time
        desired_velocity = max(-max_speed, min(max_speed, desired_velocity))

        max_velocity_change = max_acceleration * self._dt
        velocity_change = desired_velocity - velocity
        velocity_change = max(
            -max_velocity_change,
            min(max_velocity_change, velocity_change),
        )

        velocity += velocity_change
        new_value = value + velocity * self._dt

        # Prevent numerical overshoot when extremely close to the target.
        if error != 0 and (target - new_value) * error < 0:
            new_value = target
            velocity = 0.0

        return new_value, velocity

    def _advance(self, target_cx, target_cy, target_width):
        """Advance the camera state toward a target without allowing jerky motion."""
        height = self._width / self._ratio

        max_speed_x = self._max_position_speed * self._width
        max_speed_y = self._max_position_speed * height

        # Acceleration is derived from response time so another tuning
        # parameter is not required.
        max_accel_x = max_speed_x / self._response_time
        max_accel_y = max_speed_y / self._response_time

        self._cx, self._vx = self._step_axis(
            self._cx,
            self._vx,
            target_cx,
            max_speed_x,
            max_accel_x,
        )

        self._cy, self._vy = self._step_axis(
            self._cy,
            self._vy,
            target_cy,
            max_speed_y,
            max_accel_y,
        )

        max_zoom_speed = self._max_zoom_speed * self._width
        max_zoom_acceleration = max_zoom_speed / self._response_time

        self._width, self._vw = self._step_axis(
            self._width,
            self._vw,
            target_width,
            max_zoom_speed,
            max_zoom_acceleration,
        )

        self._width = max(1.0, self._width)
        return self._box()

    def _containment_target(self, container_bbox):
        """
        Return the nearest fully-contained camera state only when the target
        has actually moved outside the tolerated framing boundary.
        """
        x1, y1, x2, y2 = container_bbox
        height = self._width / self._ratio

        crop_x1 = self._cx - self._width / 2
        crop_y1 = self._cy - height / 2
        crop_x2 = self._cx + self._width / 2
        crop_y2 = self._cy + height / 2

        tolerance_x = self._width * self._CONTAINMENT_TOLERANCE
        tolerance_y = height * self._CONTAINMENT_TOLERANCE

        violated = (
            x1 < crop_x1 - tolerance_x
            or y1 < crop_y1 - tolerance_y
            or x2 > crop_x2 + tolerance_x
            or y2 > crop_y2 + tolerance_y
        )

        if not violated:
            return None

        _, _, minimum_width = self._target_state(container_bbox)
        width = max(self._width, minimum_width)

        half_width = width / 2
        half_height = half_width / self._ratio

        min_cx = x2 - half_width
        max_cx = x1 + half_width
        min_cy = y2 - half_height
        max_cy = y1 + half_height

        cx = min(max(self._cx, min_cx), max_cx)
        cy = min(max(self._cy, min_cy), max_cy)

        return cx, cy, width

    def update(self, container_bbox, target_bbox=None):
        """
        Move smoothly toward the future-aware framing target while optionally
        biasing the destination toward containing the current person.
        """
        self._transition_start = None
        self._transition_target = None

        if target_bbox is None:
            target_bbox = container_bbox

        target_cx, target_cy, target_width = self._target_state(target_bbox)
        _, _, minimum_width = self._target_state(container_bbox)

        target_width *= 1 + self._visible_expansion
        target_width = max(target_width, minimum_width)

        if self._cx is None:
            self._cx = target_cx
            self._cy = target_cy
            self._width = target_width
            self._vx = self._vy = self._vw = 0.0
            return self._box()

        containment = self._containment_target(container_bbox)

        if containment is not None:
            contained_cx, contained_cy, contained_width = containment
            priority = self._containment_priority

            target_cx += (contained_cx - target_cx) * priority
            target_cy += (contained_cy - target_cy) * priority

            containment_width = (
                self._width
                + (contained_width - self._width) * priority
            )
            target_width = max(target_width, containment_width)

        return self._advance(target_cx, target_cy, target_width)

    def move_toward(self, target_bbox, progress):
        """
        Move smoothly toward a future reappearance target.

        The transition itself defines the desired path, while the same camera
        velocity and acceleration limits prevent abrupt movement.
        """
        if self._cx is None:
            return None

        target_state = self._target_state(target_bbox)

        if self._transition_start is None or self._transition_target != target_state:
            self._transition_start = (self._cx, self._cy, self._width)
            self._transition_target = target_state

        t = max(0.0, min(1.0, progress))
        eased = t * t * (3 - 2 * t)

        start_cx, start_cy, start_width = self._transition_start
        target_cx, target_cy, target_width = self._transition_target

        planned_cx = start_cx + (target_cx - start_cx) * eased
        planned_cy = start_cy + (target_cy - start_cy) * eased

        planned_width = start_width + (target_width - start_width) * eased
        expansion = 1 + self._transition_expansion * 4 * t * (1 - t)
        planned_width *= expansion

        return self._advance(planned_cx, planned_cy, planned_width)

    def current_box(self):
        """Return the current framing box."""
        return self._box()

    def reset(self):
        """Reset all camera position, velocity and transition state."""
        self._cx = None
        self._cy = None
        self._width = None

        self._vx = 0.0
        self._vy = 0.0
        self._vw = 0.0

        self._transition_start = None
        self._transition_target = None

def build_target_frames(
    tracking_history,
    identity_history,
    target_person_id,
    start_frame,
    end_frame,
    aspect_ratio=(9, 16),
    horizontal_padding=0.0,
    vertical_padding=0.0,
):
    """
    Build the best visible target observation for each frame.

    If multiple tracks resolve to the target person in the same frame,
    the track with the highest identity confidence is selected.

    Args:
        tracking_history (dict[int, list[dict]]):
            Tracking results keyed by frame number.
        identity_history (dict[int, dict[int, dict]]):
            Resolved identities keyed by frame number and track ID.
        target_person_id (int):
            Logical Person ID to follow.
        start_frame (int):
            First frame to process, inclusive.
        end_frame (int):
            Last frame boundary, exclusive.
        aspect_ratio (tuple[int, int], optional):
            Target framing aspect ratio as ``(width, height)``.
        horizontal_padding (float, optional):
            Horizontal padding around the tracking box.
        vertical_padding (float, optional):
            Vertical padding around the tracking box.

    Returns:
        dict[int, dict]:
            Visible target information keyed by frame number.
    """
    target_frames = {}

    for frame_no in range(start_frame, end_frame):
        identities = identity_history.get(frame_no, {})
        best_track = None
        best_identity = None

        for track in tracking_history.get(frame_no, []):
            identity = identities.get(track["track_id"])

            if not identity or identity["person_id"] != target_person_id:
                continue

            if best_identity is None or identity["confidence"] > best_identity["confidence"]:
                best_track = track
                best_identity = identity

        if best_track is None:
            continue

        target_frames[frame_no] = {
            "track": best_track,
            "identity": best_identity,
            "container_box": fit_aspect_box(
                best_track["bbox"],
                aspect_ratio=aspect_ratio,
                horizontal_padding=horizontal_padding,
                vertical_padding=vertical_padding,
            ),
        }

    return target_frames

def build_reappearance_transitions(target_frames, start_frame, end_frame):
    """
    Build future-target transitions for frames where the target is invisible.

    Each invisible frame between two visible periods receives the future
    reappearance box and normalized transition progress. Progress reaches
    exactly 1.0 on the final invisible frame before reappearance.

    Args:
        target_frames (dict[int, dict]):
            Visible target observations keyed by frame number.
        start_frame (int):
            First frame to process, inclusive.
        end_frame (int):
            Last frame boundary, exclusive.

    Returns:
        dict[int, dict]:
            Invisible-frame transition information containing
            ``reappearance_frame``, ``target_box`` and ``progress``.
    """
    next_visible = {}
    future = None

    for frame_no in range(end_frame - 1, start_frame - 1, -1):
        next_visible[frame_no] = future

        if frame_no in target_frames:
            future = (
                frame_no,
                target_frames[frame_no]["container_box"],
            )

    transitions = {}
    last_visible_frame = None

    for frame_no in range(start_frame, end_frame):
        if frame_no in target_frames:
            last_visible_frame = frame_no
            continue

        reappearance = next_visible.get(frame_no)

        if last_visible_frame is None or reappearance is None:
            continue

        reappearance_frame, target_box = reappearance
        gap_start = last_visible_frame + 1
        invisible_frames = reappearance_frame - gap_start

        progress = (frame_no - gap_start + 1) / max(1, invisible_frames)
        progress = min(1.0, progress)

        transitions[frame_no] = {
            "reappearance_frame": reappearance_frame,
            "target_box": target_box,
            "progress": progress,
        }

    return transitions

def build_smooth_boxes(
    target_frames,
    reappearance_transitions,
    start_frame,
    end_frame,
    smoother,
    lookahead_boxes=None,
):
    """
    Build the final smooth framing box for every frame.

    Args:
        target_frames (dict[int, dict]):
            Visible target information keyed by frame number.
        reappearance_transitions (dict[int, dict]):
            Future movement information for invisible frames.
        start_frame (int):
            First frame to process, inclusive.
        end_frame (int):
            Last frame boundary, exclusive.
        smoother (SmoothContainmentBox):
            Framing smoother instance.
        lookahead_boxes (dict[int, tuple] | None, optional):
            Future-aware target boxes for visible frames.

    Returns:
        dict[int, tuple[int, int, int, int]]:
            Final smooth box keyed by frame number.
    """
    smoother.reset()
    smooth_boxes = {}

    for frame_no in range(start_frame, end_frame):
        target = target_frames.get(frame_no)

        if target:
            target_bbox = (
                lookahead_boxes.get(frame_no)
                if lookahead_boxes
                else target["container_box"]
            )

            smooth_bbox = smoother.update(
                target["container_box"],
                target_bbox,
            )
        else:
            transition = reappearance_transitions.get(frame_no)

            if transition:
                smooth_bbox = smoother.move_toward(
                    transition["target_box"],
                    transition["progress"],
                )
            else:
                smooth_bbox = smoother.current_box()

        if smooth_bbox is not None:
            smooth_boxes[frame_no] = smooth_bbox

    if smooth_boxes:
        first_frame = min(smooth_boxes)
        first_box = smooth_boxes[first_frame]

        for frame_no in range(start_frame, first_frame):
            smooth_boxes[frame_no] = first_box

    return smooth_boxes

def crop_frame(frame, bbox, output_size):
    """
    Crop a frame using the framing box and resize it to the output size.

    Areas outside the source frame are filled with black.

    Args:
        frame (np.ndarray):
            Source BGR video frame.
        bbox (tuple[int, int, int, int]):
            Crop box as ``(x1, y1, x2, y2)``.
        output_size (tuple[int, int]):
            Output size as ``(width, height)``.

    Returns:
        np.ndarray:
            Cropped and resized frame.
    """
    x1, y1, x2, y2 = bbox
    crop_w = max(1, x2 - x1)
    crop_h = max(1, y2 - y1)

    frame_h, frame_w = frame.shape[:2]

    src_x1 = max(0, x1)
    src_y1 = max(0, y1)
    src_x2 = min(frame_w, x2)
    src_y2 = min(frame_h, y2)

    crop = np.zeros((crop_h, crop_w, 3), dtype=frame.dtype)

    if src_x2 > src_x1 and src_y2 > src_y1:
        dst_x1 = src_x1 - x1
        dst_y1 = src_y1 - y1
        dst_x2 = dst_x1 + src_x2 - src_x1
        dst_y2 = dst_y1 + src_y2 - src_y1

        crop[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]

    interpolation = (
        cv2.INTER_AREA
        if crop_w > output_size[0] or crop_h > output_size[1]
        else cv2.INTER_LINEAR
    )

    return cv2.resize(crop, output_size, interpolation=interpolation)

def build_lookahead_boxes(
    target_frames,
    start_frame,
    end_frame,
    fps,
    aspect_ratio=(9, 16),
    lookahead_seconds=0.75,
):
    """
    Build future-aware framing targets while the target remains visible.

    Args:
        target_frames (dict[int, dict]):
            Visible target observations keyed by frame number.
        start_frame (int):
            First frame to process, inclusive.
        end_frame (int):
            Last frame boundary, exclusive.
        fps (float):
            Video frame rate.
        aspect_ratio (tuple[int, int], optional):
            Required framing aspect ratio.
        lookahead_seconds (float, optional):
            Amount of future video to inspect.

    Returns:
        dict[int, tuple[int, int, int, int]]:
            Future-aware framing target for each visible frame.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if lookahead_seconds < 0:
        raise ValueError(
            f"lookahead_seconds must be non-negative, got {lookahead_seconds}"
        )

    lookahead_frames = round(lookahead_seconds * fps)
    lookahead_boxes = {}

    for frame_no in range(start_frame, end_frame):
        target = target_frames.get(frame_no)

        if target is None:
            continue

        current = target["container_box"]
        px1, py1, px2, py2 = current

        for offset in range(1, lookahead_frames + 1):
            future = target_frames.get(frame_no + offset)

            # Disappearance/reappearance is handled separately.
            if future is None:
                break

            fx1, fy1, fx2, fy2 = future["container_box"]

            time_ahead = offset / fps
            weight = 1.0 - time_ahead / lookahead_seconds
            weight = max(0.0, min(1.0, weight))
            weight *= weight

            ax1 = current[0] + (fx1 - current[0]) * weight
            ay1 = current[1] + (fy1 - current[1]) * weight
            ax2 = current[2] + (fx2 - current[2]) * weight
            ay2 = current[3] + (fy2 - current[3]) * weight

            px1 = min(px1, ax1)
            py1 = min(py1, ay1)
            px2 = max(px2, ax2)
            py2 = max(py2, ay2)

        lookahead_boxes[frame_no] = fit_aspect_box(
            (px1, py1, px2, py2),
            aspect_ratio=aspect_ratio,
        )

    return lookahead_boxes
