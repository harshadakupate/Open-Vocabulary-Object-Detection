"""Device selection and CUDA helpers."""
from __future__ import annotations

from utils.logger import get_logger

logger = get_logger("device")


def resolve_device(requested: str | None = "auto") -> str:
    """Map a user choice (auto/cuda/gpu/cpu) to 'cuda' or 'cpu', falling back to CPU."""
    import torch

    requested = (requested or "auto").strip().lower()
    if requested in ("auto", "cuda", "gpu") or requested.startswith("cuda"):
        if torch.cuda.is_available():
            return "cuda"
        if requested != "auto":
            logger.warning("CUDA requested but not available - falling back to CPU.")
        return "cpu"
    if requested != "cpu":
        logger.warning("Unknown device '%s' - using CPU.", requested)
    return "cpu"


def cuda_sync(device: str) -> None:
    """Synchronise CUDA so wall-clock timings include queued GPU work."""
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def is_cuda_oom(exc: BaseException) -> bool:
    """True if an exception looks like a CUDA out-of-memory error."""
    return "out of memory" in str(exc).lower()


def device_info() -> dict:
    """Return basic information about the available compute device."""
    import torch

    info = {"cuda_available": torch.cuda.is_available(), "torch": torch.__version__}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            gpu_name=props.name,
            gpu_total_mb=round(props.total_memory / 1024**2, 1),
            cuda_version=torch.version.cuda,
        )
    return info
