#!/usr/bin/env python3
"""Compare one-image DoSSR output-cache miss and hit runs, with CUDA memory logs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.env_info import configure_logging, log_run_info  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("YAML config must contain a mapping at its top level")
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional parent for a new timestamped profile output folder",
    )
    args = parser.parse_args()

    try:
        config_path = args.config.expanduser().resolve()
        config = load_config(config_path)
        log_dir = Path(config["log_dir"]).expanduser()
    except Exception as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    logger, log_path = configure_logging("profile_cache", log_dir)
    log_run_info(logger, {"cli": vars(args), "config": config}, seed=config.get("seed"))
    logger.info("Log file: %s", log_path)

    try:
        input_dir = Path(config["input_dir"]).expanduser()
        images = [
            path for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if len(images) != 1:
            raise ValueError(
                f"This quick diagnostic requires exactly one test image; found {len(images)} in {input_dir}"
            )

        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        profile_parent = (
            args.output_root.expanduser()
            if args.output_root is not None
            else Path(config["output_root"]).expanduser()
        )
        output_root = profile_parent / f"cache-profile-{timestamp}"
        if output_root.exists():
            raise FileExistsError(f"Refusing to reuse profile output path: {output_root}")
        profile_config = {**config, "output_root": str(output_root)}

        import tempfile

        with tempfile.TemporaryDirectory(prefix="sc203-cache-profile-") as temp_dir:
            temp_config = Path(temp_dir) / "profile.yaml"
            temp_config.write_text(yaml.safe_dump(profile_config, sort_keys=False), encoding="utf-8")
            command = [
                sys.executable,
                str(REPO_ROOT / "src" / "run_dossr.py"),
                "--config",
                str(temp_config),
                "--profile-memory",
            ]
            summaries: list[tuple[str, float, str | None]] = []
            for label in ("first / cold output cache", "repeat / warm output cache"):
                logger.info("PROFILE PASS | %s", label)
                started = time.monotonic()
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    env=os.environ.copy(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                cache_line = None
                assert process.stdout is not None
                for line in process.stdout:
                    clean = line.rstrip()
                    logger.info("RUNNER | %s", clean)
                    if "OUTPUT CACHE |" in clean:
                        cache_line = clean.split("OUTPUT CACHE |", 1)[1].strip()
                return_code = process.wait()
                elapsed = time.monotonic() - started
                logger.info(
                    "PROFILE RESULT | pass=%s | elapsed=%.2fs | %s",
                    label,
                    elapsed,
                    cache_line or "cache counts unavailable",
                )
                summaries.append((label, elapsed, cache_line))
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, command)

        logger.info(
            "PROFILE SUMMARY | cold_output_cache=%.2fs | warm_output_cache=%.2fs | output_root=%s",
            summaries[0][1],
            summaries[1][1],
            output_root,
        )
        logger.info(
            "CACHE SCOPE | counts describe existing DoSSR output PNG reuse; "
            "dataset preprocessing cache is absent; checkpoint mmap uses the OS "
            "file cache, whose hit rate is not measured or forcibly cleared",
        )
        return 0
    except Exception as error:
        logger.exception("CACHE PROFILE FAILED | %s", type(error).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
