
import math
import cv2
import queue
import subprocess
import threading
import numpy as np

from pathlib import Path
from person_tracker.core import ProcessingRange, VideoInfo

def resolve_processing_range(
    info: VideoInfo,
    start_seconds: float = 0.0,
    end_seconds: float = -1.0,
) -> ProcessingRange:
    start = float(start_seconds)
    end = float(end_seconds)

    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError("Processing start and end times must be finite numbers")

    if start < 0:
        raise ValueError("PROCESS_START_SECONDS cannot be negative")

    if start >= info.duration_seconds:
        raise ValueError(
            "PROCESS_START_SECONDS must be before the end of the source video"
        )

    if end == -1:
        end_frame = info.frame_count
    else:
        if end < 0:
            raise ValueError("PROCESS_END_SECONDS must be -1 or non-negative")

        if end > info.duration_seconds:
            raise ValueError(
                f"PROCESS_END_SECONDS exceeds video duration "
                f"({info.duration_seconds:.3f}s)"
            )

        end_frame = min(info.frame_count, int(round(end * info.fps)))

    start_frame = max(0, int(round(start * info.fps)))

    if end_frame <= start_frame:
        raise ValueError("Processing end time must be after the start time")

    return ProcessingRange(
        start_frame=start_frame,
        end_frame=end_frame,
        fps=info.fps,
    )
def render_video(
    input_video,
    output_video,
    draw_frame,
    start_frame=0,
    end_frame=None,
    preset="p1",
    quality=23,
    queue_size=16,
    progress_callback=None,
    output_size=None,
):
    """
    Render a video by applying a frame transformation and encoding with NVENC.

    Args:
        input_video (str | Path):
            Source video path.
        output_video (str | Path):
            Output video path.
        draw_frame (callable):
            Function called as ``draw_frame(frame_no, frame)``. It should
            return the rendered frame, or None to keep the original frame.
        start_frame (int, optional):
            First frame to process, inclusive. Defaults to 0.
        end_frame (int | None, optional):
            Last frame boundary, exclusive. None processes to the end.
        preset (str, optional):
            FFmpeg NVENC preset. Defaults to ``"p1"``.
        quality (int, optional):
            NVENC constant-quality value. Lower values give better quality.
            Defaults to 23.
        queue_size (int, optional):
            Maximum number of decoded/rendered frames buffered between
            pipeline stages. Defaults to 16.
        progress_callback (callable | None, optional):
            Called as ``progress_callback(done, total)`` after each rendered
            frame. Defaults to None.
        output_size (tuple[int, int] | None, optional):
            Expected rendered-frame size as ``(width, height)``. None keeps
            the source video resolution.

    Returns:
        dict:
            Output path, rendered frame count, FPS, duration, and output size.

    Raises:
        RuntimeError:
            If the source video cannot be opened or FFmpeg fails.
        ValueError:
            If a rendered frame has an unexpected size or format.
    """
    input_video = Path(input_video)
    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    start_frame = max(0, start_frame)
    end_frame = frame_count if end_frame is None else min(end_frame, frame_count)

    if end_frame <= start_frame:
        raise ValueError(f"Invalid frame range: {start_frame} -> {end_frame}")

    if output_size is None:
        output_width, output_height = source_width, source_height
    else:
        output_width, output_height = output_size

    if output_width <= 0 or output_height <= 0:
        raise ValueError(f"Invalid output size: {output_width}x{output_height}")

    total_frames = end_frame - start_frame
    frame_queue = queue.Queue(maxsize=queue_size)
    encode_queue = queue.Queue(maxsize=queue_size)
    errors = queue.Queue()
    stop = object()

    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{output_width}x{output_height}",
            "-r", str(fps),
            "-i", "-",
            "-an",
            "-c:v", "h264_nvenc",
            "-preset", preset,
            "-cq", str(quality),
            "-pix_fmt", "yuv420p",
            str(output_video),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    def read_frames():
        cap = None

        try:
            cap = cv2.VideoCapture(str(input_video))
            if not cap.isOpened():
                raise RuntimeError(f"Could not open {input_video}")

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            for frame_no in range(start_frame, end_frame):
                ok, frame = cap.read()
                if not ok:
                    break

                frame_queue.put((frame_no, frame))
        except Exception as exc:
            errors.put(exc)
        finally:
            if cap is not None:
                cap.release()

            frame_queue.put(stop)

    def encode_frames():
        try:
            while True:
                frame = encode_queue.get()

                if frame is stop:
                    break

                if not frame.flags.c_contiguous:
                    frame = np.ascontiguousarray(frame)

                ffmpeg.stdin.write(memoryview(frame))
        except Exception as exc:
            errors.put(exc)
        finally:
            if ffmpeg.stdin:
                ffmpeg.stdin.close()

    reader = threading.Thread(target=read_frames, daemon=True)
    encoder = threading.Thread(target=encode_frames, daemon=True)

    reader.start()
    encoder.start()

    rendered_frames = 0

    while True:
        item = frame_queue.get()

        if item is stop:
            break

        frame_no, frame = item
        rendered_frame = draw_frame(frame_no, frame)

        if rendered_frame is None:
            rendered_frame = frame

        if rendered_frame.ndim != 3 or rendered_frame.shape[2] != 3:
            raise ValueError(
                f"Frame {frame_no}: expected BGR image with 3 channels, "
                f"got shape {rendered_frame.shape}"
            )

        if rendered_frame.shape[:2] != (output_height, output_width):
            raise ValueError(
                f"Frame {frame_no}: expected {output_width}x{output_height}, "
                f"got {rendered_frame.shape[1]}x{rendered_frame.shape[0]}"
            )

        if rendered_frame.dtype != np.uint8:
            raise ValueError(
                f"Frame {frame_no}: expected uint8, got {rendered_frame.dtype}"
            )

        encode_queue.put(rendered_frame)
        rendered_frames += 1

        if progress_callback is not None:
            progress_callback(rendered_frames, total_frames)

    encode_queue.put(stop)

    reader.join()
    encoder.join()

    return_code = ffmpeg.wait()

    if not errors.empty():
        raise errors.get()

    if return_code != 0:
        error = ffmpeg.stderr.read().decode(errors="replace")
        raise RuntimeError(f"FFmpeg failed:\n{error}")

    return {
        "path": output_video,
        "frames": rendered_frames,
        "fps": fps,
        "duration_seconds": rendered_frames / fps,
        "width": output_width,
        "height": output_height,
    }
