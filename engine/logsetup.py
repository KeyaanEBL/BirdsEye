"""File logging for BirdsEye runs.

`make_logger(tag)` returns a logger that writes ONLY to a timestamped file under
LOG_DIR (from .env) — no console handler — so a run's diagnostics and results go
to disk instead of stdout. The chosen file is on `logger.log_path`.
"""
import logging
import os
from datetime import datetime

from .env import LOG_DIR


def make_logger(tag: str) -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(LOG_DIR, f"{tag}_{ts}.log")

    logger = logging.getLogger(f"birdseye.{tag}.{ts}")   # unique per run -> no handler reuse
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    h = logging.FileHandler(path)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(h)
    logger.log_path = path                               # type: ignore[attr-defined]
    return logger
