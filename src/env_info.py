"""Shared run metadata and timestamped logging helpers for SC203 scripts."""

from __future__ import annotations

import argparse
import logging
import platform
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

DEFAULT_LOG_DIR = Path("/content/drive/MyDrive/lemmepass/logs")
_SENSITIVE_NAME = re.compile(r"token|password|secret|credential|api.?key", re.IGNORECASE)
_CREDENTIAL_TEXT = re.compile(
    r"(?i)(https?://)[^/\s:@]+:[^@\s/]+@|"
    r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,})\b|"
    r"\b(Bearer\s+)[A-Za-z0-9._~+/=-]+"
)


def runtime_details() -> tuple[str, str]:
    """Return the best available device label and PyTorch version."""
    try:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        version = str(torch.__version__)
    except Exception:
        device, version = "cpu", "unavailable"
    return device, version


def _argument_mapping(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, argparse.Namespace):
        values = vars(arguments)
    elif isinstance(arguments, dict):
        values = arguments
    else:
        values = {"value": arguments}
    safe: dict[str, Any] = {}
    for key, value in values.items():
        name = str(key)
        if _SENSITIVE_NAME.search(name):
            safe[name] = "[REDACTED]"
        elif isinstance(value, str):
            safe[name] = _CREDENTIAL_TEXT.sub("[REDACTED]", value)
        else:
            safe[name] = value
    return safe


def configure_logging(
    script_name: str,
    log_dir: str | Path = DEFAULT_LOG_DIR,
) -> tuple[logging.Logger, Path]:
    """Create console and timestamped file logging; return logger and path."""
    destination = Path(log_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", script_name)
    log_path = destination / f"{safe_name}_{timestamp}.log"

    logger = logging.getLogger(f"sc203.{safe_name}.{timestamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger, log_path


def log_run_info(
    logger: logging.Logger,
    arguments: Any,
    seed: int | None = None,
) -> None:
    """Log the required runtime context and all non-sensitive arguments."""
    device, torch_version = runtime_details()
    logger.info(
        "RUN INFO | platform=%s | device=%s | torch=%s | seed=%s | python=%s | args=%s",
        platform.platform(),
        device,
        torch_version,
        seed,
        platform.python_version(),
        _argument_mapping(arguments),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    args = parser.parse_args()
    logger, log_path = configure_logging("env_info", args.log_dir)
    log_run_info(logger, args, seed=args.seed)
    logger.info("Log file: %s", log_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
