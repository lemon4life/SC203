#!/usr/bin/env python3
"""Run the checked-in DoSSR inference entry point from a YAML config."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.env_info import DEFAULT_LOG_DIR, configure_logging, log_run_info  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
REQUIRED_CONFIG = {
    "seed",
    "input_dir",
    "output_root",
    "checkpoint",
    "model_config",
    "inference_script",
    "dossr_root",
    "log_dir",
    "steps",
    "solver_order",
    "t1_fraction",
    "device",
    "sr_scale",
    "color_fix_type",
    "preprocessor",
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, dict):
        raise ValueError("YAML config must contain a mapping at its top level")
    missing = REQUIRED_CONFIG.difference(values)
    if missing:
        raise ValueError(f"Config is missing required fields: {sorted(missing)}")
    return values


def output_path(config: dict[str, Any]) -> Path:
    return Path(config["output_root"]).expanduser() / f"seed_{int(config['seed'])}"


def manifest_for(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": int(config["seed"]),
        "input_dir": str(Path(config["input_dir"]).expanduser()),
        "checkpoint": str(Path(config["checkpoint"]).expanduser()),
        "model_config": str(Path(config["model_config"]).expanduser()),
        "steps": int(config["steps"]),
        "solver_order": int(config["solver_order"]),
        "t1_fraction": float(config["t1_fraction"]),
        "device": config["device"],
        "sr_scale": float(config["sr_scale"]),
        "color_fix_type": config["color_fix_type"],
        "preprocessor": config["preprocessor"],
        "output_root": str(Path(config["output_root"]).expanduser()),
    }


def validate_runtime(config: dict[str, Any]) -> None:
    if int(config["steps"]) != 5:
        raise ValueError("The Phase 1 paper-default baseline requires steps: 5")
    if int(config["solver_order"]) != 3:
        raise ValueError("DoSSR inference fixes the solver to 3rd order; solver_order must be 3")
    if float(config["t1_fraction"]) != 0.5:
        raise ValueError("DoSSR trunc_12 fixes t1 to T/2; t1_fraction must be 0.5")
    if config["preprocessor"] != "RealESRNet":
        raise ValueError("The vanilla multi-step checkpoint uses the RealESRNet preprocessor")
    if config["device"] not in {"cuda", "mps", "cpu"}:
        raise ValueError("device must be one of: cuda, mps, cpu")
    if int(config["seed"]) < 0:
        raise ValueError("seed must be a non-negative integer")

    import torch

    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config requests CUDA, but torch.cuda.is_available() is false")
    if config["device"] == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("Config requests MPS, but this PyTorch runtime has no available MPS device")


def prepare_output(directory: Path, manifest: dict[str, Any], overwrite: bool) -> None:
    manifest_path = directory / "run_manifest.json"
    if directory.exists():
        if overwrite:
            if directory.resolve().parent != Path(manifest["output_root"]).resolve():
                raise ValueError(f"Refusing to remove unexpected output path: {directory}")
            shutil.rmtree(directory)
        elif manifest_path.is_file():
            with manifest_path.open(encoding="utf-8") as stream:
                existing = json.load(stream)
            current = manifest
            if existing != current:
                raise FileExistsError(
                    f"Output exists with a different run manifest: {directory}. "
                    "Choose a new output_root/seed or pass --overwrite to replace it."
                )
        elif any(directory.iterdir()):
            raise FileExistsError(
                f"Output directory exists without a run manifest: {directory}. "
                "Choose a new output_root/seed or pass --overwrite to replace it."
            )

    directory.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def build_command(config: dict[str, Any], output_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(Path(config["inference_script"]).expanduser()),
        "--ckpt",
        str(Path(config["checkpoint"]).expanduser()),
        "--config",
        str(Path(config["model_config"]).expanduser()),
        "--input",
        str(Path(config["input_dir"]).expanduser()),
        "--steps",
        str(int(config["steps"])),
        "--sr_scale",
        str(config["sr_scale"]),
        "--color_fix_type",
        str(config["color_fix_type"]),
        "--output",
        str(output_dir),
        "--device",
        str(config["device"]),
        "--seed",
        str(int(config["seed"])),
        "--skip_if_exist",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "baseline_realsr.yaml",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove and recreate this seed's output directory before inference",
    )
    args = parser.parse_args()

    try:
        config_path = args.config.expanduser().resolve()
        config = load_config(config_path)
        log_dir = Path(config["log_dir"]).expanduser()
    except Exception as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    logger, log_path = configure_logging("run_dossr", log_dir)
    run_arguments = {"cli": vars(args), "config": config}
    log_run_info(logger, run_arguments, seed=int(config["seed"]))
    logger.info("Log file: %s", log_path)
    logger.info("Config file: %s", config_path)

    try:
        validate_runtime(config)
        input_dir = Path(config["input_dir"]).expanduser()
        checkpoint = Path(config["checkpoint"]).expanduser()
        model_config = Path(config["model_config"]).expanduser()
        inference_script = Path(config["inference_script"]).expanduser()
        dossr_root = Path(config["dossr_root"]).expanduser()
        for label, path in (
            ("input directory", input_dir),
            ("checkpoint", checkpoint),
            ("model config", model_config),
            ("DoSSR inference script", inference_script),
            ("DoSSR repository", dossr_root),
        ):
            if not path.exists():
                raise FileNotFoundError(f"Missing {label}: {path}")
        if not input_dir.is_dir():
            raise NotADirectoryError(input_dir)
        images = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        if not images:
            raise ValueError(f"No supported images found under {input_dir}")

        destination = output_path(config)
        manifest = manifest_for(config)
        prepare_output(destination, manifest, args.overwrite)
        command = build_command(config, destination)
        logger.info("Input images: %s", len(images))
        logger.info("Output folder: %s", destination)
        logger.info("DoSSR command: %s", command)

        process = subprocess.Popen(
            command,
            cwd=dossr_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            logger.info("DoSSR | %s", line.rstrip())
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)
        logger.info("BASELINE COMPLETED | images=%s | output=%s", len(images), destination)
        return 0
    except Exception as error:
        logger.exception("BASELINE FAILED | %s", type(error).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
