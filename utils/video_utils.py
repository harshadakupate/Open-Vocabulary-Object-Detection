"""Video I/O helpers built on OpenCV."""
from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import cv2

from utils.logger import get_logger

logger = get_logger("video_utils")


class VideoError(RuntimeError):
    """Raised for unreadable videos or writer/codec failures."""


def get_video_info(path: str | Path) -> dict:
    """Return fps, frame count, size and duration of a video file."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoError(f"Cannot open video '{Path(path).name}'. The format or codec may be unsupported.")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0
        logger.warning("Video reports no FPS; assuming 25.")
    duration = frames / fps if frames > 0 else 0.0
    return {"fps": fps, "total_frames": frames, "width": width, "height": height, "duration_s": duration}


def create_video_writer(path: str | Path, fps: float, size: tuple[int, int], codecs: list[str]):
    """Open a VideoWriter trying each codec in order. Returns (writer, codec_used)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    for codec in codecs:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer, codec
        writer.release()
        logger.warning("Video codec '%s' unavailable; trying next.", codec)
    raise VideoError(f"Could not create a video writer with any of the codecs {codecs}.")


def reencode_h264(src: str | Path, dst: str | Path) -> bool:
    """Re-encode to browser-playable H.264 with ffmpeg, if installed. Returns success."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("ffmpeg re-encode failed: %s", exc)
        return False
    if proc.returncode != 0:
        logger.warning("ffmpeg re-encode failed: %s", proc.stderr.strip()[:300])
        return False
    return Path(dst).exists() and Path(dst).stat().st_size > 0
