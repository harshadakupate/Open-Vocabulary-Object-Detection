"""Logging setup: console + logs/app.log."""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "ovd") -> logging.Logger:
    """Return a child of the shared 'ovd' logger, configuring handlers once."""
    global _CONFIGURED
    if not _CONFIGURED:
        root = logging.getLogger("ovd")
        root.setLevel(logging.INFO)
        root.propagate = False
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        root.addHandler(console)
        try:
            from utils.config_loader import PROJECT_ROOT

            log_dir = PROJECT_ROOT / "logs"
            log_dir.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
            file_handler.setFormatter(fmt)
            root.addHandler(file_handler)
        except OSError:
            pass  # read-only filesystem: console logging only
        _CONFIGURED = True
    return logging.getLogger(name if name.startswith("ovd") else f"ovd.{name}")
