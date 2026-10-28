#!/usr/bin/env python3
"""Verify Phase 1 DoSSR / RealSR assets in a mounted Drive folder."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.env_info import DEFAULT_LOG_DIR, configure_logging, log_run_info  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
REQUIRED_DATASETS = {"RealSR", "DRealSR"}
DEFAULT_ASSETS_DIR = Path("/content/drive/MyDrive/lemmepass")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_count(folder: Path) -> int:
    return sum(
        1
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_dataset_folders(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    if not root.is_dir():
        return found
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        name = path.name.lower().replace(" ", "")
        if "drealsr" in name:
            found.setdefault("DRealSR", path)
        elif "realsr" in name:
            found.setdefault("RealSR", path)
    return found


def load_checkpoint(checkpoint: Path) -> list[str]:
    import torch

    kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    obj = torch.load(checkpoint, **kwargs)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected a dictionary checkpoint, got {type(obj).__name__}")
    return [str(key) for key in obj.keys()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=DEFAULT_ASSETS_DIR,
        help=f"Asset root (default: {DEFAULT_ASSETS_DIR})",
    )
    parser.add_argument(
        "--image-dir",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Explicit image folder to count; may be repeated. Otherwise RealSR/DRealSR are discovered.",
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    args = parser.parse_args()

    logger, log_path = configure_logging("verify_assets", args.log_dir)
    log_run_info(logger, args, seed=None)
    logger.info("Log file: %s", log_path)

    root = args.assets_dir.expanduser()
    files = {
        "DoSSR checkpoint": root / "weights" / "dossr_default.ckpt",
        "RealESRNet weights": root / "weights" / "RealESRNet_x4plus.pth",
        "StableSR test bundle": root / "data" / "StableSR_testsets.zip",
    }

    failures = 0
    logger.info("Asset root: %s", root)
    for label, path in files.items():
        if not path.is_file():
            logger.error("MISSING | %s | %s", label, path)
            failures += 1
            continue
        size = path.stat().st_size
        logger.info(
            "OK | %s | path=%s | size_bytes=%s | size_MiB=%.2f | sha256=%s",
            label,
            path,
            size,
            size / (1024 ** 2),
            sha256_file(path),
        )

    extracted_root = root / "data" / "StableSR_testsets"
    image_folders: dict[str, Path]
    if args.image_dir:
        image_folders = {}
        for value in args.image_dir:
            if "=" not in value:
                parser.error(f"--image-dir must be LABEL=PATH, got: {value}")
            label, raw_path = value.split("=", 1)
            image_folders[label] = Path(raw_path).expanduser()
    else:
        image_folders = find_dataset_folders(extracted_root)
    missing_datasets = REQUIRED_DATASETS.difference(image_folders)
    if missing_datasets:
        logger.error(
            "MISSING | image folders | datasets=%s | root=%s | pass --image-dir LABEL=PATH",
            sorted(missing_datasets),
            extracted_root,
        )
        failures += 1

    for label, folder in image_folders.items():
        if not folder.is_dir():
            logger.error("MISSING | image folder %s | %s", label, folder)
            failures += 1
            continue
        count = image_count(folder)
        if count:
            logger.info("OK | images | dataset=%s | path=%s | count=%s", label, folder, count)
        else:
            logger.error("EMPTY | images | dataset=%s | path=%s | count=0", label, folder)
        if count == 0:
            failures += 1

    checkpoint = files["DoSSR checkpoint"]
    if checkpoint.is_file():
        try:
            keys = load_checkpoint(checkpoint)
            logger.info("OK | torch.load(map_location='cpu') | top_level_keys=%s", keys)
        except Exception as error:  # Report the load failure without dumping checkpoint contents.
            logger.exception("FAIL | torch.load(map_location='cpu') | %s", type(error).__name__)
            failures += 1

    if failures:
        logger.error("VERIFICATION FAILED | issues=%s", failures)
        return 1
    logger.info("VERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
