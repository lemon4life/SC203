#!/usr/bin/env python3
"""Scene-level train/validation split script for the Real-CE (x4) dataset.

Follows mentor guidance strictly:
1. Split strictly within --split train (holding out test untouched).
2. Pair (LR, HR) images 1-to-1 with zero missing counterparts.
3. Group patches/crops by scene/original image ID before splitting so no scene
   appears in both train and validation, without collapsing camera IDs such as
   Canon_001 into Canon.
4. Optionally extract aligned (LR, HR) patches strictly after scene assignment.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import sys
from typing import Any


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
LR_DIR_TOKENS = {"lr", "lr_x4", "lr_x2", "lr4", "lr2", "lq", "train_lr", "val_lr"}
HR_DIR_TOKENS = {"hr", "hr_x4", "hr_x2", "hr4", "hr2", "hq", "gt", "train_hr", "val_hr"}
ROLE_SUFFIX_RE = re.compile(r"(?:_|-)(?:LR\d*|HR|LQ|HQ|GT|x\d+)$", re.IGNORECASE)
PATCH_SUFFIX_RE = re.compile(r"(?:_|-)(?:patch|crop|sub|p)[_-]?\d+$", re.IGNORECASE)


@dataclass(frozen=True)
class PairedSample:
    """One paired (LR, HR) Real-CE sample belonging to a single scene."""

    pair_id: str
    scene_key: str
    lr_path: Path
    hr_path: Path


def extract_scene_key(filename_or_stem: str) -> str:
    """Extract the root scene identifier by stripping explicit scale/patch suffixes only.

    Camera/capture identifiers such as 'Canon_001' or 'scene_001' are preserved.
    """
    stem = Path(filename_or_stem).stem
    while True:
        updated = PATCH_SUFFIX_RE.sub("", stem)
        updated = ROLE_SUFFIX_RE.sub("", updated)
        if updated == stem or not updated:
            break
        stem = updated
    return stem


def canonical_pair_key(filename_or_stem: str) -> str:
    """Strip explicit LR/HR/scale suffixes so LR and HR filenames align 1-to-1."""
    stem = Path(filename_or_stem).stem
    while True:
        updated = ROLE_SUFFIX_RE.sub("", stem)
        if updated == stem or not updated:
            break
        stem = updated
    return stem


def _resolve_split_root(data_dir: Path, split: str, scale: str) -> Path:
    """Locate the target split directory while never falling back into another split."""
    scale_clean = scale.lstrip("x")
    scale_candidates = [scale, f"Real-CE_{scale}", f"x{scale_clean}"]
    candidates = [
        *(data_dir / split / sc for sc in scale_candidates),
        data_dir / split,
        *(data_dir / sc / split for sc in scale_candidates),
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand
    raise FileNotFoundError(
        f"Could not find split directory '{split}' under {data_dir}. Checked: {[str(c) for c in candidates]}"
    )


def _classify_lr_hr(rel_parts: tuple[str, ...]) -> str | None:
    """Classify an image's role ('LR' or 'HR') from its path components below split_root."""
    lowered = [p.lower() for p in rel_parts[:-1]]
    is_lr = any(tok in LR_DIR_TOKENS or tok.startswith("lr") or tok.endswith("_lr") for tok in lowered)
    is_hr = any(tok in HR_DIR_TOKENS or tok.startswith("hr") or tok.endswith("_hr") for tok in lowered)
    if is_lr and not is_hr:
        return "LR"
    if is_hr and not is_lr:
        return "HR"
    stem_lower = Path(rel_parts[-1]).stem.lower()
    if ROLE_SUFFIX_RE.search(stem_lower):
        if any(tag in stem_lower for tag in ("_lr", "-lr", "_lq")):
            return "LR"
        if any(tag in stem_lower for tag in ("_hr", "-hr", "_gt", "_hq")):
            return "HR"
    return None


def _rel_parent_context(rel_parts: tuple[str, ...]) -> str:
    """Return non-role relative parent segments so duplicate stems in different subdirs stay distinct."""
    kept = []
    for part in rel_parts[:-1]:
        low = part.lower()
        if low in LR_DIR_TOKENS or low in HR_DIR_TOKENS or low.startswith(("lr", "hr", "gt", "lq", "hq")):
            continue
        kept.append(part)
    return "/".join(kept)


def discover_paired_samples(
    data_dir: Path,
    *,
    split: str = "train",
    scale: str = "x4",
) -> list[PairedSample]:
    """Discover and validate 1-to-1 (LR, HR) pairs within the requested split."""
    split_root = _resolve_split_root(data_dir, split, scale)
    lr_entries: dict[str, Path] = {}
    hr_entries: dict[str, Path] = {}

    other_splits = {"train", "test", "val", "valid"} - {split.lower()}
    all_files = sorted(
        p for p in split_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    if not all_files:
        raise FileNotFoundError(f"No images found inside split root: {split_root}")

    for img_path in all_files:
        rel_parts = img_path.relative_to(split_root).parts
        if any(part.lower() in other_splits for part in rel_parts[:-1]):
            continue
        role = _classify_lr_hr(rel_parts)
        if role is None:
            continue
        ctx = _rel_parent_context(rel_parts)
        base_id = canonical_pair_key(img_path.name)
        pair_id = f"{ctx}/{base_id}" if ctx else base_id
        target = lr_entries if role == "LR" else hr_entries
        if pair_id in target:
            raise ValueError(f"Duplicate {role} image mapped to pair_id '{pair_id}': {img_path} vs {target[pair_id]}")
        target[pair_id] = img_path

    if not lr_entries or not hr_entries:
        raise FileNotFoundError(
            f"Could not find both LR and HR images under {split_root} (found {len(lr_entries)} LR, {len(hr_entries)} HR)"
        )

    missing_hr = sorted(set(lr_entries) - set(hr_entries))
    missing_lr = sorted(set(hr_entries) - set(lr_entries))
    if missing_hr or missing_lr:
        raise ValueError(
            f"Unpaired images under {split_root}: missing_hr={missing_hr[:5]} (total {len(missing_hr)}), "
            f"missing_lr={missing_lr[:5]} (total {len(missing_lr)})"
        )

    pairs: list[PairedSample] = []
    for pair_id in sorted(lr_entries):
        base_name = pair_id.split("/")[-1]
        scene_root = extract_scene_key(base_name)
        ctx = "/".join(pair_id.split("/")[:-1])
        scene_key = f"{ctx}/{scene_root}" if ctx else scene_root
        pairs.append(
            PairedSample(
                pair_id=pair_id,
                scene_key=scene_key,
                lr_path=lr_entries[pair_id],
                hr_path=hr_entries[pair_id],
            )
        )
    return pairs


def group_pairs_by_scene(pairs: list[PairedSample]) -> dict[str, list[PairedSample]]:
    """Group paired samples by scene key."""
    scenes: dict[str, list[PairedSample]] = {}
    for pair in pairs:
        scenes.setdefault(pair.scene_key, []).append(pair)
    return scenes


def split_scenes(
    scenes: dict[str, list[PairedSample]],
    *,
    val_ratio: float = 0.20,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Split distinct scenes into non-overlapping train and val lists."""
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    scene_keys = sorted(scenes.keys())
    if len(scene_keys) < 2:
        raise ValueError(f"Need at least 2 scenes to split train/val, got {len(scene_keys)}")
    rng = random.Random(seed)
    rng.shuffle(scene_keys)

    num_val = min(len(scene_keys) - 1, max(1, int(round(len(scene_keys) * val_ratio))))
    val_scenes = sorted(scene_keys[:num_val])
    train_scenes = sorted(scene_keys[num_val:])

    overlap = set(train_scenes) & set(val_scenes)
    if overlap:
        raise RuntimeError(f"Data leakage detected! Overlapping scenes: {overlap}")
    return train_scenes, val_scenes


def extract_aligned_patches(
    split_pairs: dict[str, list[PairedSample]],
    *,
    patch_dir: Path,
    hr_patch_size: int = 512,
    lr_patch_size: int = 128,
    stride: int = 256,
) -> list[dict[str, Any]]:
    """Extract aligned (LR, HR) patches strictly inside each split after scene splitting."""
    from PIL import Image

    if hr_patch_size <= 0 or lr_patch_size <= 0 or stride <= 0:
        raise ValueError("Patch sizes and stride must be positive integers")
    if hr_patch_size % lr_patch_size != 0:
        raise ValueError(f"hr_patch_size ({hr_patch_size}) must be divisible by lr_patch_size ({lr_patch_size})")
    scale = hr_patch_size // lr_patch_size
    if stride % scale != 0:
        raise ValueError(f"HR stride ({stride}) must be divisible by scale ({scale})")
    lr_stride = stride // scale

    patch_records: list[dict[str, Any]] = []
    for subset in ("train", "val"):
        lr_out_dir = patch_dir / subset / "LR"
        hr_out_dir = patch_dir / subset / "HR"
        lr_out_dir.mkdir(parents=True, exist_ok=True)
        hr_out_dir.mkdir(parents=True, exist_ok=True)

        for pair in split_pairs[subset]:
            with Image.open(pair.lr_path) as lr_im_raw, Image.open(pair.hr_path) as hr_im_raw:
                lr_im = lr_im_raw.convert("RGB")
                hr_im = hr_im_raw.convert("RGB")
                lr_w, lr_h = lr_im.size
                hr_w, hr_h = hr_im.size
                if (hr_w, hr_h) != (lr_w * scale, lr_h * scale):
                    raise ValueError(
                        f"Scale mismatch for {pair.pair_id}: HR={(hr_w, hr_h)} vs LR={(lr_w, lr_h)} * {scale}"
                    )
                if hr_w < hr_patch_size or hr_h < hr_patch_size:
                    continue

                safe_stem = pair.pair_id.replace("/", "__")
                crop_idx = 0
                for top_hr in range(0, hr_h - hr_patch_size + 1, stride):
                    for left_hr in range(0, hr_w - hr_patch_size + 1, stride):
                        top_lr = top_hr // scale
                        left_lr = left_hr // scale
                        lr_crop = lr_im.crop((left_lr, top_lr, left_lr + lr_patch_size, top_lr + lr_patch_size))
                        hr_crop = hr_im.crop((left_hr, top_hr, left_hr + hr_patch_size, top_hr + hr_patch_size))
                        crop_name = f"{safe_stem}_patch{crop_idx:03d}.png"
                        lr_dst = lr_out_dir / crop_name
                        hr_dst = hr_out_dir / crop_name
                        lr_crop.save(lr_dst)
                        hr_crop.save(hr_dst)
                        patch_records.append(
                            {
                                "split": subset,
                                "scene_key": pair.scene_key,
                                "pair_id": f"{pair.pair_id}_patch{crop_idx:03d}",
                                "lr_path": str(lr_dst),
                                "hr_path": str(hr_dst),
                            }
                        )
                        crop_idx += 1
    return patch_records


def write_pair_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write split paired sample paths to CSV for LoRA training/validation scripts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "scene_key", "pair_id", "lr_path", "hr_path"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Split Real-CE train subset by scene with zero leakage.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/Real-CE"), help="Real-CE dataset root directory")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to partition (default: train; test stays untouched)")
    parser.add_argument("--scale", type=str, default="x4", help="Scale subdirectory preference (default: x4)")
    parser.add_argument("--val-ratio", type=float, default=0.20, help="Fraction of scenes assigned to validation (default: 0.20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for scene shuffling (default: 42)")
    parser.add_argument("--output-manifest", type=Path, default=Path("data/realce_splits_manifest.json"), help="Output JSON manifest")
    parser.add_argument("--output-train-csv", type=Path, default=Path("data/realce_train_manifest.csv"), help="Output paired train/val CSV")
    parser.add_argument("--output-val-boxes", type=Path, default=Path("data/realce_val_boxes.csv"), help="Output validation box CSV header if absent")
    parser.add_argument("--extract-patches", action="store_true", help="Extract aligned LR/HR patches strictly after scene split")
    parser.add_argument("--patch-dir", type=Path, default=None, help="Output directory for extracted patches (required when --extract-patches)")
    parser.add_argument("--hr-patch-size", type=int, default=512, help="HR patch size in pixels (default: 512)")
    parser.add_argument("--lr-patch-size", type=int, default=128, help="LR patch size in pixels (default: 128)")
    parser.add_argument("--stride", type=int, default=256, help="HR sliding-window stride in pixels (default: 256)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print("=== Real-CE Scene-Level Splitter ===")
    print(f"Data directory: {args.data_dir} | split={args.split} | scale={args.scale}")
    print(f"Validation ratio: {args.val_ratio:.2f} (seed={args.seed})")

    if not args.data_dir.exists():
        print(f"[WARNING] Directory {args.data_dir} does not exist locally.")
        print(f"Writing a template manifest to {args.output_manifest} for Colab execution...")
        template_manifest = {
            "meta": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "split": args.split,
                "scale": args.scale,
                "val_ratio": args.val_ratio,
                "seed": args.seed,
                "note": "Template manifest. Run with --data-dir pointing to Real-CE on Drive/Colab.",
            },
            "train_scenes": [],
            "val_scenes": [],
            "train_pairs": [],
            "val_pairs": [],
        }
        args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.output_manifest.write_text(json.dumps(template_manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote template manifest to {args.output_manifest}")
        return 0

    pairs = discover_paired_samples(args.data_dir, split=args.split, scale=args.scale)
    scenes = group_pairs_by_scene(pairs)
    print(f"Verified {len(pairs)} (LR, HR) pairs grouped into {len(scenes)} distinct scenes.")

    train_scenes, val_scenes = split_scenes(scenes, val_ratio=args.val_ratio, seed=args.seed)
    train_pairs = [p for s in train_scenes for p in scenes[s]]
    val_pairs = [p for s in val_scenes for p in scenes[s]]

    print("Partitioned into:")
    print(f"  Train: {len(train_scenes)} scenes ({len(train_pairs)} LR/HR pairs)")
    print(f"  Val:   {len(val_scenes)} scenes ({len(val_pairs)} LR/HR pairs)")

    csv_rows: list[dict[str, Any]] = [
        {"split": "train", "scene_key": p.scene_key, "pair_id": p.pair_id, "lr_path": str(p.lr_path), "hr_path": str(p.hr_path)}
        for p in train_pairs
    ] + [
        {"split": "val", "scene_key": p.scene_key, "pair_id": p.pair_id, "lr_path": str(p.lr_path), "hr_path": str(p.hr_path)}
        for p in val_pairs
    ]

    patch_records: list[dict[str, Any]] = []
    if args.extract_patches:
        patch_dir = args.patch_dir if args.patch_dir is not None else (args.output_manifest.parent / "realce_patches")
        patch_records = extract_aligned_patches(
            {"train": train_pairs, "val": val_pairs},
            patch_dir=patch_dir,
            hr_patch_size=args.hr_patch_size,
            lr_patch_size=args.lr_patch_size,
            stride=args.stride,
        )
        print(f"Extracted {len(patch_records)} aligned patches into {patch_dir}")
        if patch_records:
            csv_rows = patch_records

    manifest = {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "data_dir": str(args.data_dir),
            "split": args.split,
            "scale": args.scale,
            "val_ratio": args.val_ratio,
            "seed": args.seed,
            "total_scenes": len(scenes),
            "total_pairs": len(pairs),
            "train_scene_count": len(train_scenes),
            "val_scene_count": len(val_scenes),
            "train_pair_count": len(train_pairs),
            "val_pair_count": len(val_pairs),
            "extracted_patch_count": len(patch_records),
            "leak_check": "PASSED (scenes(train) intersect scenes(val) == empty)",
        },
        "train_scenes": train_scenes,
        "val_scenes": val_scenes,
        "train_pairs": [
            {"scene_key": p.scene_key, "pair_id": p.pair_id, "lr_path": str(p.lr_path), "hr_path": str(p.hr_path)}
            for p in train_pairs
        ],
        "val_pairs": [
            {"scene_key": p.scene_key, "pair_id": p.pair_id, "lr_path": str(p.lr_path), "hr_path": str(p.hr_path)}
            for p in val_pairs
        ],
    }

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved manifest to {args.output_manifest}")

    write_pair_manifest_csv(args.output_train_csv, csv_rows)
    print(f"Saved paired CSV to {args.output_train_csv}")

    args.output_val_boxes.parent.mkdir(parents=True, exist_ok=True)
    if not args.output_val_boxes.exists():
        args.output_val_boxes.write_text("image,box_id,x,y,w,h,transcript,readable\n", encoding="utf-8")
        print(f"Initialized validation boxes CSV header at {args.output_val_boxes}")

    print("Done. Scene-level split completed with 0 data leakage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
