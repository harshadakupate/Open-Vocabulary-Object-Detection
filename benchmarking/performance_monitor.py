"""Reusable performance monitor: timing, FPS, RAM, CPU, GPU memory and GPU utilisation."""
from __future__ import annotations

import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from utils.device import cuda_sync
from utils.logger import get_logger

logger = get_logger("performance_monitor")
MB = 1024 ** 2


@contextmanager
def timer(device: str = "cpu"):
    """Time a block with CUDA synchronisation before and after. Usage: `with timer(dev) as t: ...; t['elapsed']`."""
    cuda_sync(device)
    box = {"elapsed": 0.0}
    start = time.perf_counter()
    try:
        yield box
    finally:
        cuda_sync(device)
        box["elapsed"] = time.perf_counter() - start


class PerformanceMonitor:
    """Samples RAM/CPU/GPU in a background thread and accumulates per-item timings.

    Model inference time and end-to-end time are recorded separately by the caller via
    record_inference() and record_item(). Call start() before and stop() after the measured work.
    """

    def __init__(self, device: str = "cpu", interval: float = 0.2):
        self.device = device
        self.interval = interval
        self._proc = psutil.Process()
        self._samples: list[dict] = []
        self._inference: list[float] = []
        self._e2e: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml = None
        self._nvml_handle = None
        self._smi_last = 0.0
        self._smi_value: float | None = None
        self._smi_ok = True
        self._t0 = 0.0
        self.ram_initial_mb = 0.0
        self.ram_final_mb = 0.0
        self._cuda = device.startswith("cuda")

    # ------------------------------------------------------------ lifecycle
    def start(self) -> "PerformanceMonitor":
        self._samples.clear(); self._inference.clear(); self._e2e.clear()
        self._stop.clear()
        self.ram_initial_mb = self._proc.memory_info().rss / MB
        psutil.cpu_percent(None)  # prime
        if self._cuda:
            import torch

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            try:
                import pynvml

                pynvml.nvmlInit()
                self._nvml = pynvml
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            except Exception as exc:  # noqa: BLE001 - NVML optional; fall back to nvidia-smi
                logger.info("pynvml unavailable (%s); using nvidia-smi for GPU utilisation.", exc)
        self._t0 = time.perf_counter()
        self._sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._sample()
        self.ram_final_mb = self._proc.memory_info().rss / MB

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ------------------------------------------------------------- sampling
    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def _gpu_util(self) -> float | None:
        if not self._cuda:
            return None
        if self._nvml is not None:
            try:
                return float(self._nvml.nvmlDeviceGetUtilizationRates(self._nvml_handle).gpu)
            except Exception:  # noqa: BLE001
                return None
        if self._smi_ok and time.time() - self._smi_last > 1.0:
            self._smi_last = time.time()
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu",
                                      "--format=csv,noheader,nounits"], capture_output=True,
                                     text=True, timeout=2)
                self._smi_value = float(out.stdout.strip().splitlines()[0])
            except Exception:  # noqa: BLE001
                self._smi_ok = False
        return self._smi_value

    def _sample(self) -> None:
        row = {"t_s": time.perf_counter() - self._t0,
               "ram_mb": self._proc.memory_info().rss / MB,
               "cpu_pct": psutil.cpu_percent(None),
               "gpu_util_pct": self._gpu_util(), "gpu_alloc_mb": None}
        if self._cuda:
            import torch

            row["gpu_alloc_mb"] = torch.cuda.memory_allocated() / MB
        self._samples.append(row)

    # ------------------------------------------------------------ recording
    def record_inference(self, seconds: float) -> None:
        """Model inference time of one item (excludes decoding, preprocessing and drawing)."""
        self._inference.append(float(seconds))

    def record_item(self, seconds: float) -> None:
        """End-to-end time of one item (everything the pipeline does for it)."""
        self._e2e.append(float(seconds))

    # -------------------------------------------------------------- summary
    def summary(self) -> dict:
        s = pd.DataFrame(self._samples) if self._samples else pd.DataFrame(columns=["ram_mb", "cpu_pct"])
        nan = float("nan")
        inf = np.asarray(self._inference, dtype=float)
        e2e = np.asarray(self._e2e, dtype=float)
        out = {
            "ram_initial_mb": self.ram_initial_mb,
            "ram_peak_mb": float(max(s["ram_mb"].max() if len(s) else 0.0, self.ram_initial_mb, self.ram_final_mb)),
            "ram_final_mb": self.ram_final_mb,
            "cpu_avg_pct": float(s["cpu_pct"].mean()) if len(s) else nan,
            "cpu_peak_pct": float(s["cpu_pct"].max()) if len(s) else nan,
            "gpu_available": self._cuda,
            "gpu_total_mb": nan, "gpu_allocated_mb": nan, "gpu_peak_allocated_mb": nan,
            "gpu_util_avg_pct": nan, "gpu_util_max_pct": nan,
            "inference_count": int(inf.size),
            "avg_inference_ms": float(inf.mean() * 1000) if inf.size else nan,
            "min_inference_ms": float(inf.min() * 1000) if inf.size else nan,
            "max_inference_ms": float(inf.max() * 1000) if inf.size else nan,
            "inference_fps": float(inf.size / inf.sum()) if inf.size and inf.sum() > 0 else nan,
            "e2e_count": int(e2e.size),
            "e2e_total_s": float(e2e.sum()) if e2e.size else nan,
            "end_to_end_fps": float(e2e.size / e2e.sum()) if e2e.size and e2e.sum() > 0 else nan,
        }
        if self._cuda:
            import torch

            out["gpu_total_mb"] = torch.cuda.get_device_properties(0).total_memory / MB
            out["gpu_allocated_mb"] = torch.cuda.memory_allocated() / MB
            out["gpu_peak_allocated_mb"] = torch.cuda.max_memory_allocated() / MB
            if "gpu_util_pct" in s and s["gpu_util_pct"].notna().any():
                out["gpu_util_avg_pct"] = float(s["gpu_util_pct"].mean())
                out["gpu_util_max_pct"] = float(s["gpu_util_pct"].max())
        return out

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._samples)

    def export_csv(self, path: str | Path) -> Path:
        """Write raw samples to CSV and the summary to <name>_summary.csv."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)
        pd.DataFrame([self.summary()]).to_csv(path.with_name(path.stem + "_summary.csv"), index=False)
        return path
