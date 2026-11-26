"""Timestamped experiment progress to stdout and the run log."""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import time


log = logging.getLogger("representation_kv")


def configure_logging(log_path: str | Path | None = None) -> None:
    log.setLevel(logging.INFO)
    log.propagate = False
    formatter = logging.Formatter("%(asctime)s UTC %(levelname)s %(message)s")
    formatter.converter = time.gmtime
    if not any(type(handler) is logging.StreamHandler for handler in log.handlers):
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(formatter)
        log.addHandler(stream)
    if log_path is not None and not any(
        isinstance(handler, logging.FileHandler) for handler in log.handlers
    ):
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        log.addHandler(file_handler)
