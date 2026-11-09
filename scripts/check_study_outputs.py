#!/usr/bin/env python3
"""Verify that a 10-seed study output root has all expected seed folders, images, and manifests.

Status: NOT TESTED on full 10-seed Drive output root (syntax and --help verified locally).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_SEEDS = list(range(42, 52))


def parse_seeds(raw: str | None) -> list[int]:
    if not raw:
        return list(DEFAULT_SEEDS)
    seeds: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start_i, end_i = int(start_s.strip()), int(end_s.strip())
            if start_i > end_i:
                raise ValueError(f"Invalid seed range {part!r}: start > end")
            seeds.extend(range(start_i, end_i + 1))
        else:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("Seed list cannot be empty")
    return seeds


def infer_expected_prompt_mode(root: Path, explicit: str | None) -> str | None:
    if explicit is not None:
        return explicit
    name_lower = root.name.lower()
    if "default-prompt" in name_lower or "default_prompt" in name_lower:
        return "default"
    if "empty-prompt" in name_lower or "empty_prompt" in name_lower:
        return "empty"
    if "unwrapped-prompt" in name_lower or "unwrapped_prompt" in name_lower:
        return "unwrapped"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Path to the 10-seed study output root containing seed_42 .. seed_51",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("default", "empty", "unwrapped"),
        default=None,
        help="Expected prompt_mode in run_manifest.json (auto-inferred from folder name if omitted)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42-51",
        help="Comma-separated list or range of expected seeds (default: 42-51)",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=100,
        help="Expected number of output images per seed folder, excluding *_compare.png (default: 100)",
    )
    parser.add_argument(
        "--expected-size",
        type=int,
        default=512,
        help="Expected square width and height of every output PNG in pixels (default: 512)",
    )
    parser.add_argument(
        "--expected-precision",
        type=str,
        default="fp16",
        help="Expected precision in run_manifest.json (default: fp16)",
    )
    parser.add_argument(
        "--allow-sdpa-enabled",
        action="store_true",
        help="Do not fail if sdpa_enabled is True in run_manifest.json (by default, requires sdpa_enabled=False and disable_sdpa=True)",
    )
    args = parser.parse_args()

    try:
        from PIL import Image
    except ImportError as exc:
        print(f"Missing required dependency Pillow: {exc}", file=sys.stderr)
        return 2

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: Study output root does not exist or is not a directory: {root}", file=sys.stderr)
        return 1

    try:
        expected_seeds = parse_seeds(args.seeds)
    except ValueError as exc:
        print(f"ERROR: Invalid --seeds value {args.seeds!r}: {exc}", file=sys.stderr)
        return 2

    expected_prompt_mode = infer_expected_prompt_mode(root, args.prompt_mode)
    expected_dim = (int(args.expected_size), int(args.expected_size))
    expected_count = int(args.expected_count)

    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    reference_names: list[str] | None = None

    for seed in expected_seeds:
        seed_dir = root / f"seed_{seed}"
        if not seed_dir.is_dir():
            errors.append(f"seed_{seed}: directory missing ({seed_dir})")
            rows.append(
                {
                    "seed": f"seed_{seed}",
                    "images": 0,
                    "size_ok": "NO",
                    "precision": "MISSING",
                    "sdpa": "MISSING",
                    "prompt_mode": "MISSING",
                    "status": "FAIL (missing dir)",
                }
            )
            continue

        manifest_path = seed_dir / "run_manifest.json"
        precision_val = "MISSING"
        sdpa_val = "MISSING"
        prompt_val = "MISSING"
        seed_errors: list[str] = []

        if not manifest_path.is_file():
            seed_errors.append("missing run_manifest.json")
        else:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                precision_val = str(manifest.get("precision", "MISSING"))
                sdpa_enabled = manifest.get("sdpa_enabled")
                disable_sdpa = manifest.get("disable_sdpa")
                sdpa_val = f"sdpa={sdpa_enabled},disable={disable_sdpa}"
                prompt_val = str(manifest.get("prompt_mode", "MISSING"))

                if int(manifest.get("seed", -1)) != seed:
                    seed_errors.append(f"manifest seed={manifest.get('seed')!r} != {seed}")
                if precision_val != args.expected_precision:
                    seed_errors.append(
                        f"precision={precision_val!r} != {args.expected_precision!r}"
                    )
                if not args.allow_sdpa_enabled:
                    if sdpa_enabled is not False or disable_sdpa is not True:
                        seed_errors.append(
                            f"expected sdpa_enabled=False & disable_sdpa=True, got sdpa_enabled={sdpa_enabled!r}, disable_sdpa={disable_sdpa!r}"
                        )
                if expected_prompt_mode is not None and prompt_val != expected_prompt_mode:
                    seed_errors.append(
                        f"prompt_mode={prompt_val!r} != {expected_prompt_mode!r}"
                    )
            except Exception as exc:
                seed_errors.append(f"invalid run_manifest.json ({exc})")

        images = sorted(
            p
            for p in seed_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in IMAGE_EXTENSIONS
            and not p.name.endswith("_compare.png")
        )
        img_count = len(images)
        if img_count != expected_count:
            seed_errors.append(f"image count={img_count} != {expected_count}")

        names = [p.name for p in images]
        if reference_names is None and names:
            reference_names = names
        elif reference_names is not None and names != reference_names:
            seed_errors.append("image filenames do not match first seed folder")

        bad_sizes = 0
        for img_path in images:
            try:
                with Image.open(img_path) as im:
                    if im.size != expected_dim:
                        bad_sizes += 1
            except Exception as exc:
                bad_sizes += 1
                seed_errors.append(f"unreadable image {img_path.name} ({exc})")

        if bad_sizes > 0:
            seed_errors.append(
                f"{bad_sizes}/{img_count} images do not have size {expected_dim[0]}x{expected_dim[1]}"
            )

        size_ok_str = f"{img_count - bad_sizes}/{img_count} ({expected_dim[0]}x{expected_dim[1]})"
        status_str = "OK" if not seed_errors else "FAIL: " + "; ".join(seed_errors)
        if seed_errors:
            for err in seed_errors:
                errors.append(f"seed_{seed}: {err}")

        rows.append(
            {
                "seed": f"seed_{seed}",
                "images": img_count,
                "size_ok": size_ok_str,
                "precision": precision_val,
                "sdpa": sdpa_val,
                "prompt_mode": prompt_val,
                "status": status_str,
            }
        )

    print(f"STUDY OUTPUT CHECK | root={root}")
    print(
        f"Expected: {len(expected_seeds)} seeds ({expected_seeds[0]}..{expected_seeds[-1]}), "
        f"{expected_count} images/seed at {expected_dim[0]}x{expected_dim[1]}, "
        f"precision={args.expected_precision}, prompt_mode={expected_prompt_mode or 'any'}"
    )
    print("-" * 108)
    print(
        f"{'Seed':<10} | {'Images':<8} | {'Size Check':<20} | {'Precision':<9} | {'SDPA Flags':<24} | {'Prompt':<10} | {'Status'}"
    )
    print("-" * 108)
    for row in rows:
        print(
            f"{row['seed']:<10} | {row['images']:<8} | {row['size_ok']:<20} | "
            f"{row['precision']:<9} | {row['sdpa']:<24} | {row['prompt_mode']:<10} | {row['status']}"
        )
    print("-" * 108)

    if errors:
        print(f"CHECK FAILED | {len(errors)} error(s) found:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    total_imgs = sum(int(r["images"]) for r in rows)
    print(
        f"CHECK PASSED | seeds={len(expected_seeds)} | total_images={total_imgs} | "
        f"size={expected_dim[0]}x{expected_dim[1]} | precision={args.expected_precision} | "
        f"prompt_mode={expected_prompt_mode or rows[0]['prompt_mode']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
