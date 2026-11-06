#!/usr/bin/env python3
"""Run a fixed baseline config for each seed listed in a study YAML file."""

from __future__ import annotations

import argparse
import json
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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


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
    parser.add_argument(
        "--seeds-file",
        type=Path,
        default=None,
        help="Optional YAML file containing 'seeds: [...]' (for example, configs/seeds_10.yaml)",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("default", "empty", "unwrapped"),
        default=None,
        help="Optional override for prompt_mode across all seeds",
    )
    parser.add_argument(
        "--disable-sdpa",
        action="store_true",
        help="Force disable_sdpa=True across all seeds",
    )
    parser.add_argument(
        "--expected-input-size",
        type=int,
        default=128,
        help="Expected square width/height of every prepared input image in pixels (default: 128)",
    )
    parser.add_argument(
        "--expected-output-size",
        type=int,
        default=512,
        help="Expected square width/height of every generated output image in pixels (default: 512)",
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
        from PIL import Image

        baseline_value = args.baseline_config or study.get("baseline_config")
        if not baseline_value:
            raise ValueError("Study YAML must set baseline_config")
        baseline_path = resolve_path(baseline_value)
        baseline = read_mapping(baseline_path, "Baseline config")
        seeds_file_value = args.seeds_file or study.get("seeds_file")
        if seeds_file_value:
            seeds_file_path = resolve_path(seeds_file_value)
            seeds_mapping = read_mapping(seeds_file_path, "Seeds file")
            seeds = seeds_mapping.get("seeds")
            logger.info("Loaded seeds from %s: %s", seeds_file_path, seeds)
        else:
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
        subfolders = sorted(p.name for p in input_dir.iterdir() if p.is_dir())
        if subfolders:
            raise ValueError(
                f"Prepared input folder {input_dir} contains subfolders {subfolders}; "
                "expected a flat staged image folder."
            )
        input_images = sorted(
            p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not input_images:
            raise ValueError(f"No images found in prepared input folder: {input_dir}")
        expected_in_dim = (int(args.expected_input_size), int(args.expected_input_size))
        for img_path in input_images:
            with Image.open(img_path) as im:
                if im.size != expected_in_dim:
                    raise ValueError(
                        f"Input image {img_path} has size {im.size}, expected {expected_in_dim}"
                    )
        logger.info(
            "Verified %s input images in %s with size %s",
            len(input_images),
            input_dir,
            expected_in_dim,
        )

        output_root = Path(study["output_root"]).expanduser()
        output_root.mkdir(parents=True, exist_ok=True)
        expected_out_dim = (int(args.expected_output_size), int(args.expected_output_size))

        with tempfile.TemporaryDirectory(prefix="sc203-seed-study-") as temp_dir:
            for seed in seeds:
                seed_config = {
                    **baseline,
                    "seed": seed,
                    "output_root": str(output_root),
                    "log_dir": str(log_dir),
                }
                if "prompt_mode" in study:
                    seed_config["prompt_mode"] = study["prompt_mode"]
                if args.prompt_mode is not None:
                    seed_config["prompt_mode"] = args.prompt_mode
                if "disable_sdpa" in study:
                    seed_config["disable_sdpa"] = bool(study["disable_sdpa"])
                if args.disable_sdpa:
                    seed_config["disable_sdpa"] = True
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

                seed_out_dir = output_root / f"seed_{seed}"
                manifest_path = seed_out_dir / "run_manifest.json"
                if not manifest_path.is_file():
                    raise RuntimeError(
                        f"Seed {seed} missing run_manifest.json at {manifest_path}"
                    )
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("precision") != "fp16":
                    raise RuntimeError(
                        f"Seed study stopped: {manifest_path} does not say precision 'fp16' "
                        f"(found {manifest.get('precision')!r})"
                    )
                out_images = sorted(
                    p
                    for p in seed_out_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS and not p.name.endswith("_compare.png")
                )
                if len(out_images) != len(input_images):
                    raise RuntimeError(
                        f"Seed {seed} produced {len(out_images)} output images in {seed_out_dir}, "
                        f"expected {len(input_images)}"
                    )
                for out_path in out_images:
                    with Image.open(out_path) as out_im:
                        if out_im.size != expected_out_dim:
                            raise ValueError(
                                f"Output image {out_path} has size {out_im.size}, expected {expected_out_dim}"
                            )
                logger.info(
                    "Verified %s output images for seed %s with size %s",
                    len(out_images),
                    seed,
                    expected_out_dim,
                )

        logger.info("SEED STUDY COMPLETED | seeds=%s | output_root=%s", seeds, output_root)
        return 0
    except Exception as error:
        logger.exception("SEED STUDY FAILED | %s", type(error).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
