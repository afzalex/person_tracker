from collections import Counter, defaultdict
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment
import numpy as np
from sklearn.cluster import DBSCAN
from .face import FaceSample

@dataclass
class PersonIdentity:
    person_id: int
    face_samples: list[FaceSample] = field(default_factory=list)
    track_ids: set[int] = field(default_factory=set)

    @property
    def embeddings(self):
        return [sample.embedding for sample in self.face_samples]

def _normalize_embedding(embedding):
    """Return an L2-normalized embedding."""
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding

def _select_track_representatives(face_samples, max_per_track):
    """Select high-quality, timeline-distributed face samples per tracker ID."""
    samples_by_track = defaultdict(list)

    for sample in face_samples:
        samples_by_track[sample.track_id].append(sample)

    representatives = []

    for samples in samples_by_track.values():
        samples.sort(key=lambda sample: sample.frame_no)

        if len(samples) <= max_per_track:
            representatives.extend(samples)
            continue

        for i in range(max_per_track):
            start = round(i * len(samples) / max_per_track)
            end = round((i + 1) * len(samples) / max_per_track)
            bucket = samples[start:end]

            if bucket:
                representatives.append(max(bucket, key=lambda sample: sample.quality))

    return representatives

def cluster_face_samples(
    face_samples,
    min_quality=0.6,
    eps=0.28,
    min_samples=3,
    max_representatives_per_track=12,
    assignment_similarity=0.60,
):
    """
    Cluster face samples into logical people using representative embeddings.

    A limited number of timeline-distributed, high-quality samples from each
    tracker ID are clustered first. All valid face samples are then assigned
    to the nearest resulting identity prototype.

    Args:
        face_samples (list[FaceSample]):
            All collected face observations.
        min_quality (float, optional):
            Minimum face quality required for clustering and assignment.
        eps (float, optional):
            Maximum cosine distance between neighboring representative
            embeddings used by DBSCAN.
        min_samples (int, optional):
            Minimum number of neighboring representative samples required
            by DBSCAN to form a cluster.
        max_representatives_per_track (int, optional):
            Maximum representative samples selected from each tracker ID.
            Samples are distributed across the track timeline.
        assignment_similarity (float, optional):
            Minimum cosine similarity required to assign a valid face sample
            to the nearest identity prototype.

    Returns:
        tuple[list[FaceSample], dict[int, int]]:
            Valid face samples and mapping from sample object ID to Person ID.
    """
    valid_samples = [
        sample
        for sample in face_samples
        if sample.quality >= min_quality
    ]

    if not valid_samples:
        return [], {}

    representatives = _select_track_representatives(
        valid_samples,
        max_representatives_per_track,
    )

    embeddings = np.stack([
        _normalize_embedding(sample.embedding)
        for sample in representatives
    ])

    labels = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="cosine",
        n_jobs=-1,
    ).fit_predict(embeddings)

    cluster_labels = sorted(set(labels) - {-1})

    if not cluster_labels:
        return valid_samples, {}

    label_to_person = {
        label: person_id
        for person_id, label in enumerate(cluster_labels, start=1)
    }

    prototypes = {}

    for label in cluster_labels:
        cluster_embeddings = embeddings[labels == label]
        prototypes[label_to_person[label]] = _normalize_embedding(
            np.mean(cluster_embeddings, axis=0)
        )

    person_ids = list(prototypes)
    prototype_matrix = np.stack([prototypes[person_id] for person_id in person_ids])

    assignments = {}

    for sample in valid_samples:
        embedding = _normalize_embedding(sample.embedding)
        similarities = prototype_matrix @ embedding
        best_index = int(np.argmax(similarities))
        best_similarity = float(similarities[best_index])

        if best_similarity >= assignment_similarity:
            assignments[id(sample)] = person_ids[best_index]

    return valid_samples, assignments

def merge_face_clusters(
    face_samples,
    assignments,
    similarity_threshold=0.65,
    top_k=5,
):
    """
    Merge face clusters whose best face matches strongly indicate
    that they belong to the same person.
    """
    clusters = defaultdict(list)

    for sample in face_samples:
        person_id = assignments.get(id(sample))
        if person_id is not None:
            clusters[person_id].append(sample)

    # Compare best-quality samples first
    for person_id in clusters:
        clusters[person_id].sort(
            key=lambda sample: sample.quality,
            reverse=True,
        )

    parent = {person_id: person_id for person_id in clusters}

    def find(person_id):
        while parent[person_id] != person_id:
            parent[person_id] = parent[parent[person_id]]
            person_id = parent[person_id]
        return person_id

    def union(a, b):
        a = find(a)
        b = find(b)

        if a != b:
            parent[max(a, b)] = min(a, b)

    def cluster_similarity(samples_a, samples_b):
        similarities = [
            float(np.dot(a.embedding, b.embedding))
            for a in samples_a[:top_k]
            for b in samples_b[:top_k]
        ]

        similarities.sort(reverse=True)

        return float(np.mean(
            similarities[:min(top_k, len(similarities))]
        ))

    person_ids = sorted(clusters)

    for i, person_a in enumerate(person_ids):
        for person_b in person_ids[i + 1:]:
            similarity = cluster_similarity(
                clusters[person_a],
                clusters[person_b],
            )

            if similarity >= similarity_threshold:
                union(person_a, person_b)

    merged_ids = {}
    next_person_id = 1

    for person_id in person_ids:
        root = find(person_id)

        if root not in merged_ids:
            merged_ids[root] = next_person_id
            next_person_id += 1

    return {
        sample_id: merged_ids[find(person_id)]
        for sample_id, person_id in assignments.items()
    }

def build_track_identity_anchors(face_samples, assignments):
    """
    Build per-track identity observations from clustered face samples.

    Returns:
        dict[int, list[tuple[int, int]]]:
            Mapping of track_id to sorted (frame_no, person_id) anchors.
    """
    anchors = defaultdict(list)

    for sample in face_samples:
        person_id = assignments.get(id(sample))
        if person_id is None:
            continue

        anchors[sample.track_id].append(
            (sample.frame_no, person_id)
        )

    for track_id in anchors:
        anchors[track_id].sort()

    return dict(anchors)

def smooth_identity_anchors(anchors, window=5):
    """
    Smooth noisy person assignments using a local majority vote.

    Args:
        anchors: Sorted list of (frame_no, person_id) observations.
        window: Number of neighboring observations used for smoothing.

    Returns:
        Smoothed list of (frame_no, person_id) anchors.
    """
    if len(anchors) < 3:
        return anchors

    half = window // 2
    smoothed = []

    for i, (frame_no, _) in enumerate(anchors):
        start = max(0, i - half)
        end = min(len(anchors), i + half + 1)

        nearby_ids = [
            person_id
            for _, person_id in anchors[start:end]
        ]

        person_id = Counter(nearby_ids).most_common(1)[0][0]
        smoothed.append((frame_no, person_id))

    return smoothed

def split_track_identity_anchors(
    track_identity_anchors,
    min_segment_anchors=2,
):
    """
    Split tracker IDs into identity-consistent tracklets whenever
    the resolved person ID changes.

    Returns:
        dict[int, list[dict]]:
            Mapping of track_id to tracklet definitions.
    """
    tracklets = {}

    for track_id, anchors in track_identity_anchors.items():
        if not anchors:
            continue

        segments = []
        current_person = anchors[0][1]
        current_anchors = [anchors[0]]

        for anchor in anchors[1:]:
            frame_no, person_id = anchor

            if person_id == current_person:
                current_anchors.append(anchor)
                continue

            if len(current_anchors) >= min_segment_anchors:
                segments.append({
                    "person_id": current_person,
                    "anchors": current_anchors,
                })

            current_person = person_id
            current_anchors = [anchor]

        if len(current_anchors) >= min_segment_anchors:
            segments.append({
                "person_id": current_person,
                "anchors": current_anchors,
            })

        tracklets[track_id] = segments

    return tracklets

def detect_switch_boundaries(
    tracklets,
    tracking_history,
    min_jump_score=0.25,
):
    """
    Estimate identity-switch frames between adjacent tracklets.

    Uses bounding-box motion discontinuity between the last anchor of
    one identity and the first anchor of the next identity.

    Returns:
        dict[int, list[dict]]:
            Mapping of track_id to detected identity switch boundaries.
    """
    boundaries = {}

    for track_id, segments in tracklets.items():
        if len(segments) < 2:
            continue

        track_boundaries = []

        for before, after in zip(segments, segments[1:]):
            start_frame = before["anchors"][-1][0]
            end_frame = after["anchors"][0][0]

            switch_frame, score = _find_track_switch(
                track_id,
                start_frame,
                end_frame,
                tracking_history,
            )

            if score < min_jump_score:
                switch_frame = (start_frame + end_frame) // 2

            track_boundaries.append({
                "from_person": before["person_id"],
                "to_person": after["person_id"],
                "start_frame": start_frame,
                "end_frame": end_frame,
                "switch_frame": switch_frame,
                "score": score,
            })

        boundaries[track_id] = track_boundaries

    return boundaries

def _find_track_switch(
    track_id,
    start_frame,
    end_frame,
    tracking_history,
):
    previous_box = None
    best_frame = (start_frame + end_frame) // 2
    best_score = 0.0

    for frame_no in range(start_frame, end_frame + 1):
        box = _get_track_box(
            tracking_history.get(frame_no, []),
            track_id,
        )

        if box is None:
            previous_box = None
            continue

        if previous_box is not None:
            score = _bbox_jump_score(previous_box, box)

            if score > best_score:
                best_score = score
                best_frame = frame_no

        previous_box = box

    return best_frame, best_score

def _get_track_box(tracks, track_id):
    for track in tracks:
        if track["track_id"] == track_id:
            return np.asarray(track["bbox"], dtype=float)

    return None

def _bbox_jump_score(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    aw = max(ax2 - ax1, 1)
    ah = max(ay2 - ay1, 1)
    bw = max(bx2 - bx1, 1)
    bh = max(by2 - by1, 1)

    acx = (ax1 + ax2) / 2
    acy = (ay1 + ay2) / 2
    bcx = (bx1 + bx2) / 2
    bcy = (by1 + by2) / 2

    diagonal = np.hypot(aw, ah)

    position_change = np.hypot(
        bcx - acx,
        bcy - acy,
    ) / diagonal

    size_change = max(
        abs(np.log(bw / aw)),
        abs(np.log(bh / ah)),
    )

    return float(position_change + 0.25 * size_change)

def build_identity_history_from_boundaries(
    tracking_history,
    tracklets,
    switch_boundaries,
    track_identity_anchors,
    frame_size,
):
    """
    Build frame-by-frame logical person assignments with identity confidence.

    Confidence for each assignment uses the last earlier frame where the
    same logical person was visible, its previous bounding-box position,
    face identity evidence, tracker-switch boundaries, and track overlap.

    Args:
        tracking_history (dict[int, list[dict]]):
            Tracking results keyed by frame number. Each track must contain
            ``track_id`` and ``bbox``.

        tracklets (dict[int, list[dict]]):
            Identity-consistent segments for each tracker ID. Each segment
            must contain ``person_id``.

        switch_boundaries (dict[int, list[dict]]):
            Detected identity-switch boundaries for each tracker ID.
            Each boundary must contain ``switch_frame`` and ``to_person``.

        track_identity_anchors (dict[int, list[tuple[int, int]]]):
            Face identity observations for each tracker ID as
            ``(frame_no, person_id)`` pairs.

        frame_size (tuple[int, int]):
            Video frame size as ``(width, height)``. Used to normalize
            spatial distance between the person's previous and current
            bounding boxes.

    Returns:
        dict[int, dict[int, dict]]:
            Frame-level identity history in the form
            ``frame_no -> track_id -> identity information``.
    """
    identity_history = {}

    # Last earlier frame where each logical person was visible.
    #
    # person_id -> {
    #     "frame_no": int,
    #     "bbox": tuple | np.ndarray,
    # }
    last_visible = {}

    for frame_no in sorted(tracking_history):
        tracks = tracking_history[frame_no]
        frame_identities = {}

        # Keep resolved tracks first so all confidence calculations in this
        # frame use visibility information only from earlier frames.
        resolved_tracks = []

        for track in tracks:
            track_id = track["track_id"]
            segments = tracklets.get(track_id)

            if not segments:
                continue

            person_id = segments[0]["person_id"]

            # Apply tracker-ID identity switches in chronological order.
            for boundary in switch_boundaries.get(track_id, []):
                if frame_no >= boundary["switch_frame"]:
                    person_id = boundary["to_person"]
                else:
                    break

            resolved_tracks.append(
                (track, track_id, person_id)
            )

        # Calculate confidence using the last time this logical person
        # was visible before the current frame.
        for track, track_id, person_id in resolved_tracks:
            previous = last_visible.get(person_id)

            previous_visible_frame = (
                previous["frame_no"]
                if previous is not None
                else None
            )

            previous_visible_bbox = (
                previous["bbox"]
                if previous is not None
                else None
            )

            overlap = _track_overlap(
                track_id,
                tracks,
            )

            confidence = identity_confidence(
                frame_no=frame_no,
                person_id=person_id,
                anchors=track_identity_anchors.get(
                    track_id,
                    [],
                ),
                current_bbox=track["bbox"],
                frame_size=frame_size,
                previous_visible_frame=previous_visible_frame,
                previous_visible_bbox=previous_visible_bbox,
                switch_boundaries=switch_boundaries.get(
                    track_id,
                    [],
                ),
                overlap=overlap,
            )

            frame_identities[track_id] = {
                "person_id": person_id,
                "confidence": confidence,
                "overlap": overlap,
            }

        identity_history[frame_no] = frame_identities

        # Only update last-visible state after every track in this frame
        # has been evaluated. This prevents another track in the same frame
        # from being treated as the "previously visible" observation.
        #
        # If multiple tracks resolve to the same Person ID, retain the one
        # with the highest identity confidence.
        best_track_by_person = {}

        for track, track_id, person_id in resolved_tracks:
            identity = frame_identities[track_id]

            current_best = best_track_by_person.get(
                person_id
            )

            if (
                current_best is None
                or identity["confidence"]
                > current_best["confidence"]
            ):
                best_track_by_person[person_id] = {
                    "bbox": track["bbox"],
                    "confidence": identity["confidence"],
                }

        for person_id, best in best_track_by_person.items():
            last_visible[person_id] = {
                "frame_no": frame_no,
                "bbox": best["bbox"].copy(),
            }

    return identity_history

def _temporal_identity_candidates(
    frame_no,
    current_bbox,
    tracking_history,
    identity_history,
    max_search_frames,
    min_overlap,
    temporal_decay_frames,
):
    """Collect nearby direct identity evidence using spatial overlap."""
    candidates = defaultdict(list)

    for offset in range(1, max_search_frames + 1):
        found_at_distance = False

        for neighbor_frame in (frame_no - offset, frame_no + offset):
            identities = identity_history.get(neighbor_frame, {})

            for track in tracking_history.get(neighbor_frame, []):
                identity = identities.get(track["track_id"])

                if identity is None or identity.get("source", "direct") != "direct":
                    continue

                overlap = _bbox_iou(current_bbox, track["bbox"])

                if overlap < min_overlap:
                    continue

                temporal_factor = max(0.0, 1.0 - offset / temporal_decay_frames)
                score = overlap * identity["confidence"] * temporal_factor

                if score <= 0.0:
                    continue

                candidates[identity["person_id"]].append({
                    "score": score,
                    "overlap": overlap,
                    "frame_no": neighbor_frame,
                    "track_id": track["track_id"],
                    "confidence": identity["confidence"],
                })

                found_at_distance = True

        if found_at_distance:
            break

    return candidates

def _summarize_identity_candidates(candidates):
    """Convert temporal evidence into relative per-person candidate scores."""
    if not candidates:
        return {}, None

    scores = {
        person_id: max(item["score"] for item in evidence)
        for person_id, evidence in candidates.items()
        if evidence
    }

    total = sum(scores.values())

    if total <= 0.0:
        return {}, None

    probabilities = {
        person_id: score / total
        for person_id, score in scores.items()
    }

    best_person = max(scores, key=scores.get)
    best_evidence = max(candidates[best_person], key=lambda item: item["score"])

    return probabilities, {
        "person_id": best_person,
        "score": scores[best_person],
        "evidence": best_evidence,
    }

def resolve_unidentified_tracks(
    tracking_history,
    identity_history,
    fps,
    max_search_seconds=1.0,
    min_overlap=0.15,
    temporal_decay_seconds=1.0,
    fallback_confidence_scale=0.75,
    min_fallback_confidence=0.10,
):
    """
    Resolve tracks without a Person ID using nearby identified tracks.

    Search order is one frame backward/forward, two frames backward/forward,
    and so on. Spatial overlap, identity confidence and temporal distance
    determine the fallback assignment.

    Args:
        tracking_history (dict[int, list[dict]]):
            Tracking results keyed by frame number.
        identity_history (dict[int, dict[int, dict]]):
            Existing identity assignments keyed by frame number and track ID.
        fps (float):
            Source video frame rate.
        max_search_seconds (float, optional):
            Maximum temporal distance searched in both directions.
        min_overlap (float, optional):
            Minimum IoU required with an identified nearby track.
        temporal_decay_seconds (float, optional):
            Time over which nearby identity evidence decays to zero.
        fallback_confidence_scale (float, optional):
            Confidence penalty applied to inferred identities.
        min_fallback_confidence (float, optional):
            Minimum fallback confidence required to assign a Person ID.

    Returns:
        dict[int, dict[int, dict]]:
            Identity history containing direct and inferred assignments.
    """
    max_search_frames = max(1, round(max_search_seconds * fps))
    temporal_decay_frames = max(
        max_search_frames + 1,
        round(temporal_decay_seconds * fps),
    )

    resolved_history = {
        frame_no: {
            track_id: {
                **identity,
                "source": identity.get("source", "direct"),
            }
            for track_id, identity in identities.items()
        }
        for frame_no, identities in identity_history.items()
    }

    for frame_no in sorted(tracking_history):
        frame_identities = resolved_history.setdefault(frame_no, {})

        for track in tracking_history[frame_no]:
            track_id = track["track_id"]

            if track_id in frame_identities:
                continue

            candidates = _temporal_identity_candidates(
                frame_no,
                track["bbox"],
                tracking_history,
                resolved_history,
                max_search_frames,
                min_overlap,
                temporal_decay_frames,
            )

            probabilities, best = _summarize_identity_candidates(candidates)

            if best is None or not probabilities:
                continue

            sorted_probabilities = sorted(probabilities.values(), reverse=True)
            best_probability = sorted_probabilities[0]
            second_probability = sorted_probabilities[1] if len(sorted_probabilities) > 1 else 0.0
            ambiguity_margin = best_probability - second_probability

            confidence = (
                best["score"]
                * fallback_confidence_scale
                * (0.5 + 0.5 * ambiguity_margin)
            )

            if confidence < min_fallback_confidence:
                continue

            evidence = best["evidence"]

            frame_identities[track_id] = {
                "person_id": best["person_id"],
                "confidence": float(np.clip(confidence, 0.0, 1.0)),
                "overlap": evidence["overlap"],
                "source": "temporal_overlap",
                "source_frame": evidence["frame_no"],
                "source_track_id": evidence["track_id"],
                "candidates": probabilities,
            }

    return resolved_history

def _track_overlap(track_id, tracks):
    current = next(
        (track for track in tracks if track["track_id"] == track_id),
        None,
    )

    if current is None:
        return 0.0

    return max(
        (
            _bbox_iou(current["bbox"], other["bbox"])
            for other in tracks
            if other["track_id"] != track_id
        ),
        default=0.0,
    )

def _bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    return intersection / union if union else 0.0

def identity_confidence(
    frame_no,
    person_id,
    anchors,
    current_bbox,
    frame_size,
    previous_visible_frame=None,
    previous_visible_bbox=None,
    switch_boundaries=None,
    overlap=0.0,
    max_anchor_gap=90,
    max_visible_gap=300,
    max_spatial_distance_ratio=0.30,
    switch_margin=15,
    face_match_floor=0.92,
):
    """
    Estimate how trustworthy the current person assignment is.

    Confidence is primarily based on where the same logical person was
    last visible, how recently they were visible, and supporting face
    identity evidence. Tracker-switch and overlap conditions reduce
    confidence, while fresh matching face evidence provides a strong boost.

    Args:
        frame_no (int):
            Current video frame number.

        person_id (int):
            Logical Person ID assigned to the current track.

        anchors (list[tuple[int, int]]):
            Face identity observations for the current track as
            ``(frame_no, person_id)`` pairs. Only anchors at or before
            the current frame are used.

        current_bbox (tuple[int, int, int, int] | np.ndarray):
            Current tracked-person bounding box as
            ``(x1, y1, x2, y2)``.

        frame_size (tuple[int, int]):
            Video frame size as ``(width, height)``. Used to normalize
            spatial movement.

        previous_visible_frame (int | None, optional):
            Most recent earlier frame where this logical person was
            visible. Defaults to None.

        previous_visible_bbox (tuple[int, int, int, int] | np.ndarray | None, optional):
            Bounding box of this logical person at
            ``previous_visible_frame``. Defaults to None.

        switch_boundaries (list[dict] | None, optional):
            Suspected identity-switch boundaries for the current tracker
            ID. Each boundary must contain ``switch_frame``.
            Defaults to None.

        overlap (float, optional):
            Maximum overlap between the current track and another person's
            track, usually IoU in the range 0.0 to 1.0.
            Defaults to 0.0.

        max_anchor_gap (int, optional):
            Number of frames over which previous matching face evidence
            gradually loses its influence. Defaults to 90.

        max_visible_gap (int, optional):
            Number of frames after which previous visibility contributes
            no temporal confidence. Defaults to 300.

        max_spatial_distance_ratio (float, optional):
            Distance between bounding-box centers, expressed as a fraction
            of the frame diagonal, at which spatial confidence reaches zero.
            Defaults to 0.30.

        switch_margin (int, optional):
            Number of frames around a suspected tracker-ID switch where
            confidence is penalized. Defaults to 15.

        face_match_floor (float, optional):
            Minimum confidence assigned when fresh face evidence on the
            current frame confirms the Person ID. Defaults to 0.92.

    Returns:
        float:
            Identity confidence in the range 0.0 to 1.0.
    """
    spatial_factor = _spatial_proximity_factor(
        previous_visible_bbox,
        current_bbox,
        frame_size,
        max_spatial_distance_ratio,
    )

    recency_factor = _temporal_recency_factor(
        frame_no,
        previous_visible_frame,
        max_visible_gap,
    )

    (
        face_proximity,
        fresh_face_match,
        fresh_face_conflict,
    ) = _face_evidence(
        frame_no,
        person_id,
        anchors,
        max_anchor_gap,
    )

    # Core tracking evidence:
    # - closer to the last known location is better
    # - more recently seen is better
    confidence = (
        0.60 * spatial_factor
        + 0.40 * recency_factor
    )

    # Recent matching face evidence strengthens tracking confidence.
    # The influence naturally decays as the face observation gets older.
    if face_proximity > 0.0:
        confidence += (
            1.0 - confidence
        ) * 0.50 * face_proximity

    # Reduce trust around a suspected tracker-ID switch.
    for boundary in switch_boundaries or []:
        distance = abs(
            frame_no - boundary["switch_frame"]
        )

        if distance <= switch_margin:
            switch_factor = (
                0.5
                + 0.5 * distance / switch_margin
            )

            confidence *= switch_factor

    # Overlap raises the risk that the tracker jumped between people.
    confidence *= _overlap_factor(overlap)

    # A face observed on this exact frame is stronger evidence than
    # tracker continuity, overlap, or spatial uncertainty.
    if fresh_face_match:
        confidence = max(
            confidence,
            face_match_floor,
        )

    # A fresh face identifying somebody else is strong contradictory evidence.
    if fresh_face_conflict:
        confidence *= 0.15

    return float(
        np.clip(
            confidence,
            0.0,
            1.0,
        )
    )

def _spatial_proximity_factor(
    previous_bbox,
    current_bbox,
    frame_size,
    max_distance_ratio,
):
    """Return spatial proximity between the current and last visible boxes."""
    if previous_bbox is None:
        return 0.0

    px1, py1, px2, py2 = previous_bbox
    cx1, cy1, cx2, cy2 = current_bbox

    previous_cx = (px1 + px2) / 2
    previous_cy = (py1 + py2) / 2
    current_cx = (cx1 + cx2) / 2
    current_cy = (cy1 + cy2) / 2

    distance = np.hypot(
        current_cx - previous_cx,
        current_cy - previous_cy,
    )

    frame_width, frame_height = frame_size
    frame_diagonal = np.hypot(
        frame_width,
        frame_height,
    )

    if frame_diagonal <= 0:
        return 0.0

    distance_ratio = distance / frame_diagonal

    return float(
        np.clip(
            1.0 - distance_ratio / max_distance_ratio,
            0.0,
            1.0,
        )
    )

def _temporal_recency_factor(
    frame_no,
    previous_visible_frame,
    max_visible_gap,
):
    """Return temporal confidence based on when the person was last visible."""
    if previous_visible_frame is None:
        return 0.0

    frame_gap = max(
        0,
        frame_no - previous_visible_frame,
    )

    return float(
        np.clip(
            1.0 - frame_gap / max_visible_gap,
            0.0,
            1.0,
        )
    )

def _face_evidence(
    frame_no,
    person_id,
    anchors,
    max_anchor_gap,
):
    """
    Return recent face evidence using only anchors at or before this frame.

    Returns:
        tuple[float, bool, bool]:
            Face proximity, fresh face match, and fresh face conflict.
    """
    if not anchors:
        return 0.0, False, False

    frames = np.array(
        [anchor_frame for anchor_frame, _ in anchors]
    )

    # Use only evidence already observed by the current frame.
    index = np.searchsorted(
        frames,
        frame_no,
        side="right",
    ) - 1

    if index < 0:
        return 0.0, False, False

    anchor_frame, anchor_person_id = anchors[index]
    anchor_gap = frame_no - anchor_frame

    face_proximity = float(
        np.clip(
            1.0 - anchor_gap / max_anchor_gap,
            0.0,
            1.0,
        )
    )

    face_matches = anchor_person_id == person_id
    fresh_face_match = face_matches and anchor_gap == 0
    fresh_face_conflict = not face_matches and anchor_gap == 0

    # A mismatching historical anchor must not provide positive support.
    if not face_matches:
        face_proximity = 0.0

    return (
        face_proximity,
        fresh_face_match,
        fresh_face_conflict,
    )

def _overlap_factor(overlap):
    if overlap < 0.05:
        return 1.0
    if overlap < 0.20:
        return 0.90
    if overlap < 0.50:
        return 0.75
    return 0.55

def resolve_frame_identity_conflicts(
    tracking_history,
    identity_history,
    min_candidate_score=0.10,
):
    """
    Enforce one-to-one Track-to-Person assignments within each frame.

    All candidate Person IDs for active tracks are considered jointly.
    The assignment maximizing the total candidate score is selected while
    ensuring that one Person ID cannot belong to multiple tracks in the
    same frame.

    Args:
        tracking_history (dict[int, list[dict]]):
            Tracking results keyed by frame number.
        identity_history (dict[int, dict[int, dict]]):
            Identity information keyed by frame number and track ID.
            Identity entries may contain a ``candidates`` mapping.
        min_candidate_score (float, optional):
            Minimum candidate score required for a Track-to-Person assignment.
            Lower-scoring assignments remain unresolved.

    Returns:
        dict[int, dict[int, dict]]:
            Identity history with one-to-one assignments enforced per frame.
    """
    resolved_history = {}

    for frame_no, tracks in tracking_history.items():
        identities = identity_history.get(frame_no, {})
        active_track_ids = [track["track_id"] for track in tracks]

        candidate_scores = {}
        person_ids = set()

        for track_id in active_track_ids:
            identity = identities.get(track_id)

            if identity is None:
                continue

            candidates = identity.get("candidates")

            if candidates:
                scores = dict(candidates)
            else:
                scores = {
                    identity["person_id"]: identity["confidence"]
                }

            candidate_scores[track_id] = scores
            person_ids.update(scores)

        frame_identities = {}

        if not candidate_scores or not person_ids:
            resolved_history[frame_no] = frame_identities
            continue

        track_ids = list(candidate_scores)
        person_ids = sorted(person_ids)

        score_matrix = np.zeros(
            (len(track_ids), len(person_ids)),
            dtype=float,
        )

        for row, track_id in enumerate(track_ids):
            for col, person_id in enumerate(person_ids):
                score_matrix[row, col] = candidate_scores[track_id].get(
                    person_id,
                    0.0,
                )

        # Hungarian solves minimum cost, so negate scores to maximize them.
        rows, cols = linear_sum_assignment(-score_matrix)

        for row, col in zip(rows, cols):
            track_id = track_ids[row]
            person_id = person_ids[col]
            score = score_matrix[row, col]

            if score < min_candidate_score:
                continue

            original = identities[track_id]

            frame_identities[track_id] = {
                **original,
                "person_id": person_id,
                "confidence": min(original["confidence"], float(score)),
                "assignment_score": float(score),
                "source": (
                    original.get("source", "direct")
                    if person_id == original["person_id"]
                    else "global_assignment"
                ),
            }

        resolved_history[frame_no] = frame_identities

    return resolved_history
