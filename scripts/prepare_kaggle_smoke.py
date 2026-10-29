#!/usr/bin/env python3
"""Prepare a one-image Kaggle DoSSR smoke config from attached Kaggle datasets."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.env_info import configure_logging, log_run_info  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_WORKING = Path("/kaggle/working")


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def locate_input(root: Path, requested: Path | None) -> Path:
    if requested is not None:
        candidate = requested.expanduser().resolve()
        if candidate.is_dir():
            return candidate
        raise NotADirectoryError(candidate)
    matches = sorted(
        path for path in root.rglob("RealSRVal_crop128") if path.is_dir()
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            "Could not uniquely locate RealSRVal_crop128 under "
            f"{root}; found {len(matches)} candidate folders. Pass --input-dir explicitly."
        )
    return matches[0]


def locate_checkpoint(root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        candidate = explicit.expanduser().resolve()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Checkpoint not found: {candidate}")

    search_roots = [root, DEFAULT_WORKING / "weights"]
    matches = sorted(
        {
            path.resolve()
            for search_root in search_roots
            if search_root.is_dir()
            for path in search_root.rglob("dossr_default.ckpt")
            if path.is_file()
        }
    )
    if len(matches) != 1:
        formatted = ", ".join(str(path) for path in matches) or "none"
        raise FileNotFoundError(
            "Could not uniquely locate dossr_default.ckpt in /kaggle/input or "
            f"/kaggle/working/weights; candidates: {formatted}. Pass --checkpoint."
        )
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "baseline_realsr.yaml")
    parser.add_argument("--assets-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Existing checkpoint in /kaggle/input or /kaggle/working; no download is started")
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="RealSR image directory; defaults to the unique RealSRVal_crop128 folder")
    parser.add_argument("--smoke-dir", type=Path, default=DEFAULT_WORKING / "realsr_smoke")
    parser.add_argument("--config-out", type=Path, default=DEFAULT_WORKING / "configs" / "baseline_kaggle_smoke.yaml")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_WORKING / "outputs" / "baseline-kaggle-smoke")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_WORKING / "logs")
    args = parser.parse_args()

    seed = None
    config: dict[str, Any] | None = None
    try:
        config = load_yaml(args.config.expanduser())
        seed = int(config["seed"])
    except Exception:
        pass
    logger, log_path = configure_logging("prepare_kaggle_smoke", args.log_dir)
    log_run_info(logger, args, seed=seed)
    logger.info("Log file: %s", log_path)

    try:
        if config is None:
            config = load_yaml(args.config.expanduser())
        assets_root = args.assets_root.expanduser().resolve()
        input_dir = locate_input(assets_root, args.input_dir)
        checkpoint = locate_checkpoint(assets_root, args.checkpoint)
        images = sorted(
            path for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise FileNotFoundError(f"No supported images found under {input_dir}")

        smoke_dir = args.smoke_dir.expanduser().resolve()
        smoke_dir.mkdir(parents=True, exist_ok=True)
        destination = smoke_dir / images[0].name
        if destination.exists():
            if digest(destination) != digest(images[0]):
                raise FileExistsError(
                    f"Existing smoke image differs from selected input: {destination}. "
                    "Choose a different --smoke-dir."
                )
            logger.info("Smoke image already staged; skipping copy: %s", destination)
        else:
            destination.write_bytes(images[0].read_bytes())
            logger.info("Staged one smoke image: %s", destination)

        config.update(
            input_dir=str(smoke_dir),
            output_root=str(args.output_root.expanduser().resolve()),
            checkpoint=str(checkpoint),
            model_config=str(REPO_ROOT / "external" / "DOSSR" / "configs" / "model" / "cldm_v21.yaml"),
            inference_script=str(REPO_ROOT / "external" / "DOSSR" / "scripts" / "inference.py"),
            dossr_root=str(REPO_ROOT / "external" / "DOSSR"),
            log_dir=str(args.log_dir.expanduser().resolve()),
            device="cuda",
            precision="fp16",
            offload_condition_encoder=True,
        )
        config_out = args.config_out.expanduser().resolve()
        config_out.parent.mkdir(parents=True, exist_ok=True)
        rendered = yaml.safe_dump(config, sort_keys=False)
        if config_out.exists():
            if config_out.read_text(encoding="utf-8") != rendered:
                raise FileExistsError(
                    f"Kaggle smoke config exists with different settings: {config_out}. "
                    "Choose a new --config-out."
                )
            logger.info("Smoke config already exists; skipping write: %s", config_out)
        else:
            config_out.write_text(rendered, encoding="utf-8")
            logger.info("Created Kaggle smoke config: %s", config_out)

        logger.info("Checkpoint: %s (%s bytes)", checkpoint, checkpoint.stat().st_size)
        logger.info("Selected RealSR source folder: %s", input_dir)
        logger.info("Smoke input count: 1")
        logger.info("Output root: %s", args.output_root.expanduser().resolve())
        logger.info("Smoke config: %s", config_out)
        logger.info("KAGGLE SMOKE CONFIG READY")
        return 0
    except Exception as error:
        logger.exception("KAGGLE SMOKE PREPARATION FAILED | %s", type(error).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
