"""Video detection pipeline (also usable from the command line)."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from detection.prompt_manager import parse_prompts, require_prompts
from detection.visualization import draw_detections
from utils.config_loader import ensure_project_dirs, load_config, resolve_path
from utils.image_utils import resize_long_side
from utils.logger import get_logger
from utils.video_utils import VideoError, create_video_writer, get_video_info, reencode_h264

logger = get_logger("video_detection")


def process_video(manager, model_name: str, input_path: str | Path, output_path: str | Path,
                  prompts: list[str], device: str = "auto", conf: float | None = None,
                  text_threshold: float | None = None, frame_skip: int | None = None,
                  resize_side: int | None = None, imgsz: int | None = None,
                  max_frames: int | None = None, warmup: bool | None = None,
                  progress_callback: Callable | None = None) -> dict:
    """Detect objects in every Nth frame, write an annotated video and return timing statistics.

    Timing definitions
      inference FPS  = processed frames / summed MODEL inference time (forward + post-processing)
      end-to-end FPS = processed frames / wall time of the whole loop (decode, resize, model,
                       drawing, encoding)
    The input video's FPS is reported separately and is never used as a model FPS.

    When frame_skip > 1 only every Nth frame is processed AND written; the output video is
    written at input_fps / frame_skip so it still plays back at real-time speed.
    """
    cfg = load_config()["video"]
    frame_skip = max(1, int(cfg["frame_skip"] if frame_skip is None else frame_skip))
    resize_side = cfg["resize_long_side"] if resize_side is None else resize_side
    max_frames = cfg["max_frames"] if max_frames is None else max_frames
    warmup = cfg["warmup"] if warmup is None else warmup
    prompts = require_prompts(prompts)

    input_path, output_path = Path(input_path), Path(output_path)
    info = get_video_info(input_path)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise VideoError(f"Cannot open video '{input_path.name}'.")

    writer = None
    codec_used = None
    tmp_path = output_path.with_suffix(".tmp.mp4")
    records: list[dict] = []
    frame_idx = -1
    processed = 0
    try:
        if warmup:  # untimed warm-up on a blank frame of the working resolution
            blank, _ = resize_long_side(np.full((info["height"] or 480, info["width"] or 640, 3), 127, np.uint8),
                                        resize_side)
            manager.detect(model_name, blank, prompts, device=device, conf=conf,
                           text_threshold=text_threshold, imgsz=imgsz)

        loop_start = time.perf_counter()
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % frame_skip != 0:
                continue
            if max_frames and processed >= max_frames:
                break

            frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frame, _ = resize_long_side(frame, resize_side)
            result = manager.detect(model_name, frame, prompts, device=device, conf=conf,
                                    text_threshold=text_threshold, imgsz=imgsz)
            total = info["total_frames"] or "?"
            header = (f"Frame {frame_idx + 1}/{total} | {result['model_name']} | "
                      f"{result['num_detections']} det | {result['inference_time'] * 1000:.1f} ms")
            annotated = draw_detections(frame, result, header=header)

            if writer is None:
                out_fps = info["fps"] / frame_skip
                h, w = annotated.shape[:2]
                writer, codec_used = create_video_writer(tmp_path, out_fps, (w, h), cfg["codecs"])
            writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

            processed += 1
            scores = result["scores"]
            records.append({
                "frame": frame_idx + 1,
                "inference_ms": result["inference_time"] * 1000,
                "num_detections": result["num_detections"],
                "avg_confidence": float(np.mean(scores)) if scores else None,
            })
            if progress_callback:
                progress_callback(frame_idx + 1, info["total_frames"], processed, annotated)
        wall_time = time.perf_counter() - loop_start
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if processed == 0:
        tmp_path.unlink(missing_ok=True)
        raise VideoError("No frames were processed - the video may be empty or unreadable.")

    # Browser-playable H.264 when ffmpeg exists; otherwise keep the OpenCV file as written.
    browser_playable = codec_used == "avc1"
    if codec_used != "avc1" and reencode_h264(tmp_path, output_path):
        tmp_path.unlink(missing_ok=True)
        browser_playable = True
    else:
        shutil.move(str(tmp_path), str(output_path))

    inference_total = sum(r["inference_ms"] for r in records) / 1000.0
    stats = {
        "model_name": model_name,
        "output_path": str(output_path),
        "codec": codec_used,
        "browser_playable": browser_playable,
        "input_fps": info["fps"],
        "input_total_frames": info["total_frames"],
        "input_duration_s": info["duration_s"],
        "input_resolution": f"{info['width']}x{info['height']}",
        "processed_frames": processed,
        "frame_skip": frame_skip,
        "frame_skip_enabled": frame_skip > 1,
        "resize_long_side": resize_side or None,
        "avg_inference_ms": 1000.0 * inference_total / processed,
        "inference_fps": processed / inference_total if inference_total > 0 else float("nan"),
        "end_to_end_fps": processed / wall_time if wall_time > 0 else float("nan"),
        "total_processing_time_s": wall_time,
        "total_detections": int(sum(r["num_detections"] for r in records)),
        "per_frame": records,
        "note": (f"Frame skipping enabled: every {frame_skip}th frame was processed and written; "
                 f"output plays at {info['fps'] / frame_skip:.2f} FPS." if frame_skip > 1
                 else "All frames were processed."),
    }
    logger.info("Video done: %d frames, inference %.1f FPS, end-to-end %.1f FPS",
                processed, stats["inference_fps"], stats["end_to_end_fps"])
    return stats


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Open-vocabulary detection on a video.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default="YOLO-World")
    parser.add_argument("--prompts", required=True, help="Comma-separated categories")
    parser.add_argument("--device", default=cfg["device"]["default"])
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--text-threshold", type=float, default=None)
    parser.add_argument("--frame-skip", type=int, default=None)
    parser.add_argument("--resize-side", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args(argv)

    from models.model_manager import ModelManager, normalize_model_name

    ensure_project_dirs()
    out = resolve_path(cfg["paths"]["videos_out"]) / f"{Path(args.video).stem}_{int(time.time())}_annotated.mp4"
    stats = process_video(ModelManager(), normalize_model_name(args.model), args.video, out,
                          parse_prompts(args.prompts), args.device, args.conf, args.text_threshold,
                          args.frame_skip, args.resize_side, max_frames=args.max_frames)
    stats.pop("per_frame")
    print(json.dumps(stats, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
