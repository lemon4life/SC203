#!/usr/bin/env python3
"""Run a fixed baseline config for each seed listed in a study YAML file."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.env_info import DEFAULT_LOG_DIR, configure_logging, log_run_info  # noqa: E402


def read_mapping(path: Path, label: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a YAML mapping")
    return value


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "seed_study_realsr_5img.yaml",
    )
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=None,
        help="Override baseline_config from the study YAML (for example, the staged Colab config)",
    )
    args = parser.parse_args()

    try:
        study_path = args.config.expanduser().resolve()
        study = read_mapping(study_path, "Study config")
        log_dir = Path(study.get("log_dir", DEFAULT_LOG_DIR)).expanduser()
    except Exception as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    logger, log_path = configure_logging("run_seed_study", log_dir)
    log_run_info(logger, {"cli": vars(args), "study_config": study}, seed=None)
    logger.info("Log file: %s", log_path)

    try:
        baseline_value = args.baseline_config or study.get("baseline_config")
        if not baseline_value:
            raise ValueError("Study YAML must set baseline_config")
        baseline_path = resolve_path(baseline_value)
        baseline = read_mapping(baseline_path, "Baseline config")
        seeds = study.get("seeds")
        if not isinstance(seeds, list) or not seeds:
            raise ValueError("Study YAML seeds must be a non-empty list")
        if any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds):
            raise ValueError("Every seed must be a non-negative integer")
        if len(set(seeds)) != len(seeds):
            raise ValueError("Study YAML contains duplicate seeds")

        input_dir = Path(baseline["input_dir"]).expanduser()
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Prepared input folder not found: {input_dir}")
        output_root = Path(study["output_root"]).expanduser()
        output_root.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="sc203-seed-study-") as temp_dir:
            for seed in seeds:
                seed_config = {
                    **baseline,
                    "seed": seed,
                    "output_root": str(output_root),
                    "log_dir": str(log_dir),
                }
                temp_config = Path(temp_dir) / f"seed_{seed}.yaml"
                temp_config.write_text(
                    yaml.safe_dump(seed_config, sort_keys=False), encoding="utf-8"
                )
                command = [
                    sys.executable,
                    "-u",
                    str(REPO_ROOT / "src" / "run_dossr.py"),
                    "--config",
                    str(temp_config),
                ]
                logger.info("SEED STUDY | seed=%s | output=%s/seed_%s", seed, output_root, seed)
                process = subprocess.Popen(
                    command,
                    cwd=REPO_ROOT,
                    env=os.environ.copy(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                for line in process.stdout:
                    logger.info("RUNNER | %s", line.rstrip())
                return_code = process.wait()
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, command)

        logger.info("SEED STUDY COMPLETED | seeds=%s | output_root=%s", seeds, output_root)
        return 0
    except Exception as error:
        logger.exception("SEED STUDY FAILED | %s", type(error).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
