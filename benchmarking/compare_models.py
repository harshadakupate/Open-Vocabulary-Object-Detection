"""Run both models (each in its own process for fair RAM/GPU numbers) and build the report.

Usage:
  python -m benchmarking.compare_models --images-dir dataset/images \
      --annotations dataset/annotations/annotations.json
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from benchmarking.report_generator import MODELS, build_comparison_table, generate_report, load_summaries
from utils.config_loader import PROJECT_ROOT, ensure_project_dirs, load_config
from utils.logger import get_logger

logger = get_logger("compare_models")


def run_model_subprocess(model: str, extra_args: list[str]) -> subprocess.CompletedProcess:
    """Evaluate one model in a fresh Python process so models do not share RAM/GPU memory."""
    cmd = [sys.executable, "-m", "benchmarking.evaluate", "--model", model, *extra_args]
    logger.info("Running: %s", " ".join(cmd))
    return subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)


def main(argv: list[str] | None = None) -> int:
    cfg = load_config()
    p = argparse.ArgumentParser(description="Benchmark Grounding DINO vs YOLO-World.")
    p.add_argument("--images-dir", default=cfg["benchmark"]["images_dir"])
    p.add_argument("--annotations", default=cfg["benchmark"]["annotations"])
    p.add_argument("--device", default=cfg["device"]["default"])
    p.add_argument("--dino-conf", type=float, default=None)
    p.add_argument("--yolo-conf", type=float, default=None)
    p.add_argument("--text-threshold", type=float, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--report-only", action="store_true", help="Skip inference; rebuild the report from saved summaries")
    args = p.parse_args(argv)
    ensure_project_dirs()

    if not args.report_only:
        for model in MODELS:
            extra = ["--images-dir", args.images_dir, "--annotations", args.annotations, "--device", args.device]
            conf = args.dino_conf if model == "Grounding DINO" else args.yolo_conf
            if conf is not None:
                extra += ["--conf", str(conf)]
            if args.text_threshold is not None:
                extra += ["--text-threshold", str(args.text_threshold)]
            if args.limit:
                extra += ["--limit", str(args.limit)]
            proc = run_model_subprocess(model, extra)
            print(proc.stdout)
            if proc.returncode != 0:
                print(proc.stderr, file=sys.stderr)
                print(f"ERROR: evaluation of {model} failed (exit {proc.returncode}).", file=sys.stderr)
                return proc.returncode

    outputs = generate_report()
    if outputs is None:
        print("Evaluation has not been completed - no summaries found.")
        return 1
    print(build_comparison_table(load_summaries()).to_string(index=False))
    print("\nReport files:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
