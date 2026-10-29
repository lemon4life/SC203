#!/usr/bin/env python3
"""Check one DoSSR image and save a bicubic-versus-DoSSR comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.env_info import DEFAULT_LOG_DIR, configure_logging, log_run_info  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Original low-resolution input image")
    parser.add_argument("--output", type=Path, required=True, help="DoSSR output image")
    parser.add_argument("--comparison-output", type=Path, required=True)
    parser.add_argument("--expected-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing comparison image")
    args = parser.parse_args()

    logger, log_path = configure_logging("inspect_output", args.log_dir)
    log_run_info(logger, args, seed=args.seed)
    logger.info("Log file: %s", log_path)

    try:
        input_path = args.input.expanduser()
        output_path = args.output.expanduser()
        comparison_path = args.comparison_output.expanduser()
        for label, path in (("input", input_path), ("output", output_path)):
            if not path.is_file():
                raise FileNotFoundError(f"{label} image not found: {path}")
        if args.expected_size < 1:
            raise ValueError("--expected-size must be positive")

        with Image.open(input_path) as image:
            input_rgb = image.convert("RGB")
        with Image.open(output_path) as image:
            output_raw = np.asarray(image).copy()
            output_rgb = image.convert("RGB")

        output_size_ok = output_rgb.size == (args.expected_size, args.expected_size)
        logger.info(
            "Output size: %s | expected: %sx%s | %s",
            output_rgb.size,
            args.expected_size,
            args.expected_size,
            "PASS" if output_size_ok else "FAIL",
        )

        finite = bool(np.isfinite(output_raw).all()) if np.issubdtype(output_raw.dtype, np.number) else False
        logger.info("Output finite/NaN check: %s", "PASS (no NaN/Inf)" if finite else "FAIL (NaN/Inf or non-numeric pixels)")

        output_array = np.asarray(output_rgb)
        all_black = bool(np.all(output_array == 0))
        all_white = bool(np.all(output_array == 255))
        logger.info("All-black check: %s", "FAIL" if all_black else "PASS")
        logger.info("All-white check: %s", "FAIL" if all_white else "PASS")

        bicubic = input_rgb.resize(output_rgb.size, Image.Resampling.BICUBIC)
        if comparison_path.exists() and not args.overwrite:
            logger.info("Comparison already exists; leaving it unchanged: %s", comparison_path)
        else:
            comparison_path.parent.mkdir(parents=True, exist_ok=True)
            comparison = Image.new("RGB", (output_rgb.width * 2, output_rgb.height))
            comparison.paste(bicubic, (0, 0))
            comparison.paste(output_rgb, (output_rgb.width, 0))
            comparison.save(comparison_path)
            logger.info("Saved comparison (bicubic left, DoSSR right): %s", comparison_path)

        if not output_size_ok or not finite or all_black or all_white:
            logger.error("OUTPUT INSPECTION FAILED")
            return 1
        logger.info("OUTPUT INSPECTION PASSED")
        return 0
    except Exception as error:
        logger.exception("OUTPUT INSPECTION FAILED | %s", type(error).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
