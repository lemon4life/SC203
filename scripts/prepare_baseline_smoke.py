#!/usr/bin/env python3
"""Stage a deterministic RealSR image subset and derive a temporary run config."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.env_info import DEFAULT_LOG_DIR, configure_logging, log_run_info  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        "--source-config",
        dest="config",
        type=Path,
        default=REPO_ROOT / "configs" / "baseline_realsr.yaml",
        help="Baseline YAML config to copy and adjust for a smoke or small-batch run",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Optional source image folder override; defaults to input_dir in --config",
    )
    parser.add_argument(
        "--image-count",
        "--limit",
        dest="image_count",
        type=int,
        default=1,
        help="Number of lexically first images to stage (default: 1)",
    )
    parser.add_argument(
        "--smoke-dir",
        "--staging-dir",
        dest="smoke_dir",
        type=Path,
        default=Path("/content/realsr_smoke"),
    )
    parser.add_argument(
        "--config-out",
        "--output-config",
        dest="config_out",
        type=Path,
        default=Path("/content/baseline_smoke.yaml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/content/outputs/baseline-smoke"),
    )
    parser.add_argument(
        "--expected-size",
        type=int,
        default=None,
        help="Optional expected square width/height of staged images in pixels (for example, 128)",
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    args = parser.parse_args()

    try:
        config = load_yaml(args.config.expanduser())
        args.seed = int(config["seed"])
    except Exception as error:
        logger, log_path = configure_logging("prepare_baseline_smoke", args.log_dir)
        log_run_info(logger, args, seed=None)
        logger.info("Log file: %s", log_path)
        logger.exception("SMOKE PREPARATION FAILED | %s", type(error).__name__)
        return 1

    logger, log_path = configure_logging("prepare_baseline_smoke", args.log_dir)
    log_run_info(logger, args, seed=args.seed)
    logger.info("Log file: %s", log_path)

    try:
        import yaml
        from PIL import Image

        if args.image_count < 1:
            raise ValueError("--image-count must be at least 1")
        input_dir = (
            args.source_dir.expanduser()
            if args.source_dir is not None
            else Path(config["input_dir"]).expanduser()
        )
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Source input directory not found: {input_dir}")

        subfolders = sorted(path.name for path in input_dir.iterdir() if path.is_dir())
        if subfolders:
            raise ValueError(
                f"Input directory {input_dir} contains subfolders {subfolders}. "
                "Pass a leaf image directory such as RealSRVal_crop128/test_LR so test_HR and test_LR are never mixed."
            )

        images = sorted(
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(images) < args.image_count:
            raise FileNotFoundError(
                f"Requested {args.image_count} images but found {len(images)} under {input_dir}"
            )

        observed_sizes: dict[tuple[int, int], list[str]] = {}
        for image_path in images:
            with Image.open(image_path) as img:
                observed_sizes.setdefault(img.size, []).append(image_path.name)
        if len(observed_sizes) > 1:
            summary = {str(size): names[:3] for size, names in observed_sizes.items()}
            raise ValueError(
                f"Input directory {input_dir} contains mixed image sizes {summary}. "
                "All images in the source directory must share a uniform size."
            )
        if args.expected_size is not None:
            expected_dim = (int(args.expected_size), int(args.expected_size))
            actual_dim = next(iter(observed_sizes.keys()))
            if actual_dim != expected_dim:
                raise ValueError(
                    f"Input directory {input_dir} has image size {actual_dim}, expected {expected_dim}"
                )

        selected_images = images[: args.image_count]
        selected_names = {image.name for image in selected_images}
        if len(selected_names) != len(selected_images):
            raise ValueError(
                "Selected images contain duplicate filenames; choose another staging method."
            )

        smoke_dir = args.smoke_dir.expanduser().resolve()
        smoke_dir.mkdir(parents=True, exist_ok=True)
        unrelated_images = [
            path
            for path in smoke_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and path.name not in selected_names
        ]
        if unrelated_images:
            raise FileExistsError(
                f"Smoke folder contains other images: {smoke_dir}. "
                "Choose a fresh --smoke-dir so staging contains only the requested images."
            )

        staged_sizes: set[tuple[int, int]] = set()
        for source_image in selected_images:
            destination = smoke_dir / source_image.name
            if destination.exists():
                if digest(destination) != digest(source_image):
                    raise FileExistsError(
                        f"Staged image has different content: {destination}. "
                        "Choose a fresh --smoke-dir."
                    )
                with Image.open(destination) as staged_img:
                    img_size = staged_img.size
                staged_sizes.add(img_size)
                logger.info(
                    "Image already staged; skipping copy: %s | size=%sx%s",
                    destination,
                    img_size[0],
                    img_size[1],
                )
            else:
                destination.write_bytes(source_image.read_bytes())
                with Image.open(destination) as staged_img:
                    img_size = staged_img.size
                staged_sizes.add(img_size)
                logger.info(
                    "Staged image: %s | size=%sx%s",
                    destination,
                    img_size[0],
                    img_size[1],
                )

        config["input_dir"] = str(smoke_dir)
        config["output_root"] = str(args.output_root.expanduser())
        config["log_dir"] = str(args.log_dir.expanduser())
        config_out = args.config_out.expanduser().resolve()
        config_out.parent.mkdir(parents=True, exist_ok=True)
        rendered = yaml.safe_dump(config, sort_keys=False)
        if config_out.exists():
            if config_out.read_text(encoding="utf-8") != rendered:
                raise FileExistsError(
                    f"Smoke config already exists with different settings: {config_out}. "
                    "Choose another --config-out."
                )
            logger.info("Smoke config already exists; skipping write: %s", config_out)
        else:
            config_out.write_text(rendered, encoding="utf-8")
            logger.info("Created staged-image config: %s", config_out)

        logger.info("Smoke input count: %s | staged_sizes=%s", len(selected_images), sorted(staged_sizes))
        logger.info("Smoke output root: %s", args.output_root.expanduser())
        logger.info("SMOKE CONFIG READY")
        return 0
    except Exception as error:
        logger.exception("SMOKE PREPARATION FAILED | %s", type(error).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
