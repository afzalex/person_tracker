
from pathlib import Path
import shutil
import subprocess
import cv2

from person_tracker.video import VideoInfo

def inspect_video(video_path: str | Path) -> VideoInfo:
    """Read basic metadata and fail early if the input cannot be decoded."""
    path = Path(video_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input video does not exist: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open: {path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
        raise RuntimeError(
            "Video metadata is invalid: "
            f"{width=} {height=} {fps=} {frame_count=}"
        )

    return VideoInfo(
        path=str(path),
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_seconds=frame_count / fps,
    )

def ffmpeg_available() -> bool:
    """Check if FFmpeg is available on the system PATH."""
    return shutil.which("ffmpeg") is not None

def read_nth_frame(video_path, frame_no):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Could not read frame {frame_no}")

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def create_browser_preview(
    video_path: str | Path,
    output_path: str | Path,
    *,
    width: int = 360,
) -> Path:
    """Create a small H.264 preview suitable for embedding in a notebook."""
    if not ffmpeg_available():
        raise RuntimeError("FFmpeg was not found on PATH")
    if width < 120:
        raise ValueError("Browser preview width must be at least 120 pixels")
    if width % 2:
        width -= 1

    source = Path(video_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-vf",
        f"scale={width}:-2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "27",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        "-movflags",
        "+faststart",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"FFmpeg could not create the browser preview:\n{completed.stderr[-4000:]}"
        )
    return output
