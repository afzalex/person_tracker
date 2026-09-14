from dataclasses import dataclass, replace

import cv2
import numpy as np
from insightface.app import FaceAnalysis


@dataclass
class FaceSample:
    frame_no: int
    track_id: int
    bbox: tuple[int, int, int, int]
    confidence: float
    quality: float
    embedding: np.ndarray
    image: np.ndarray



def calculate_face_quality(frame, face):
    """
        Calculate the quality of a detected face based on its size, sharpness, and detection confidence. <br/>
        For now, use confidence + resolution + sharpness:
        Args:
            frame (np.ndarray): The image frame containing the face.
            face: The detected face object from InsightFace.

        Returns:
            float: The calculated quality score of the face.
    """
    x1, y1, x2, y2 = face.bbox.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0

    h, w = crop.shape[:2]
    size_score = min(1.0, min(w, h) / 160.0)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(1.0, sharpness / 250.0)

    return (
        0.5 * float(face.det_score)
        + 0.3 * size_score
        + 0.2 * sharpness_score
    )

def find_face_track(face_bbox, boxes, ids):
    """
        Find the best matching track ID for a given face bounding box based on proximity to the head region.

        Args:
            face_bbox (tuple[int, int, int, int]): The bounding box of the detected face.
            boxes (list[tuple[int, int, int, int]]): List of bounding boxes for tracked objects.
            ids (list[int]): List of track IDs corresponding to the boxes.

        Returns:
            int | None: The best matching track ID, or None if no match is found.
    """
    fx1, fy1, fx2, fy2 = face_bbox
    fx = (fx1 + fx2) / 2
    fy = (fy1 + fy2) / 2

    best_track_id = None
    best_distance = float("inf")

    for box, track_id in zip(boxes, ids):
        x1, y1, x2, y2 = box
        if not (x1 <= fx <= x2 and y1 <= fy <= y2):
            continue

        width = x2 - x1
        height = y2 - y1

        head_x = (x1 + x2) / 2
        head_y = y1 + height * 0.15

        distance = np.hypot(
            (fx - head_x) / max(width, 1),
            (fy - head_y) / max(height, 1),
        )

        if distance < best_distance:
            best_distance = distance
            best_track_id = int(track_id)

    return best_track_id

def extract_track_faces(
    face_app,
    frame,
    frame_no,
    boxes,
    ids,
    max_faces=5,
):
    """
    Extract up to max_faces highest-confidence faces associated
    with tracked people in the frame.

    No minimum confidence or minimum face-size filtering is applied.

    ``max_faces`` is forwarded to InsightFace so the recognition and landmark
    models do not process faces that would only be discarded afterward.
    """
    candidates = []

    for face in face_app.get(frame, max_num=max_faces):
        x1, y1, x2, y2 = face.bbox.astype(int)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            continue

        track_id = find_face_track(
            (x1, y1, x2, y2),
            boxes,
            ids,
        )

        # Ignore faces that cannot be associated with a tracked person.
        if track_id is None:
            continue

        candidates.append(
            (
                float(face.det_score),
                face,
                track_id,
                (x1, y1, x2, y2),
            )
        )

    # Highest face-detection confidence first.
    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    samples = []

    for confidence, face, track_id, bbox in candidates[:max_faces]:
        x1, y1, x2, y2 = bbox

        samples.append(
            FaceSample(
                frame_no=frame_no,
                track_id=track_id,
                bbox=bbox,
                confidence=confidence,
                quality=calculate_face_quality(
                    frame,
                    face,
                ),
                embedding=face.normed_embedding.copy(),
                image=frame[
                    y1:y2,
                    x1:x2,
                ].copy(),
            )
        )

    return samples
