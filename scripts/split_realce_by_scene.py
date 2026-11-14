#!/usr/bin/env python3
"""Scene-level train/validation split script for the Real-CE dataset.

Supports the official Real-CE directory structure:
  <Real-CE>/
    train/
      13mm/          # 4x LR inputs
      26mm/          # 2x LR inputs (ignored when --scale x4)
      52mm/          # HR ground truth
      trans_anoon/   # (or trans_annos/) ignored as image inputs
      valid.txt      # optional split manifest
    val/             # official held-out evaluation split (untouched during train split)

Also supports generic LR/HR directory layouts as fallback.

Follows mentor guidance strictly:
1. Split strictly within --split train (holding out val/test untouched).
2. Pair (LR, HR) images 1-to-1 with zero missing counterparts.
3. Group patches/crops by scene/original image ID before splitting so no scene
   appears in both train and validation, without collapsing camera IDs such as
   Canon_001 into Canon.
4. Optionally extract aligned (128x128 LR, 512x512 HR) patches strictly after scene
   assignment, supporting both 4x-ratio and pre-registered same-canvas focal captures.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.check_realce_feasibility import (  # noqa: E402
    ANNOTATION_DIR_TOKENS,
    parse_valid_txt,
    resolve_realce_root,
)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
LR_DIR_TOKENS = {"lr", "lr_x4", "lr_x2", "lr4", "lr2", "lq", "train_lr", "val_lr"}
HR_DIR_TOKENS = {"hr", "hr_x4", "hr_x2", "hr4", "hr2", "hq", "gt", "train_hr", "val_hr"}
ROLE_SUFFIX_RE = re.compile(r"(?:_|-)(?:13mm|26mm|52mm|LR\d*|HR|LQ|HQ|GT|x\d+)$", re.IGNORECASE)
PATCH_SUFFIX_RE = re.compile(r"(?:_|-)(?:patch|crop|sub|p)[_-]?\d+$", re.IGNORECASE)


@dataclass(frozen=True)
class PairedSample:
    """One paired (LR, HR) Real-CE sample belonging to a single scene."""

    pair_id: str
    scene_key: str
    lr_path: Path
    hr_path: Path


@dataclass(frozen=True)
class TextBox:
    """Axis-aligned HR ground-truth text box parsed from Real-CE annotations."""

    x: int
    y: int
    w: int
    h: int
    transcript: str


ANNO_DIR_PRIORITY: tuple[str, ...] = (
    "trans_annos_52mm_renov",
    "trans_annos_52mm",
    "annos",
    "trans_anoon",
    "trans_annos",
    "trans_annos_13mm",
)
BOXES_CSV_FIELDS = ["image", "box_id", "x", "y", "w", "h", "transcript", "readable"]


def extract_scene_key(filename_or_stem: str) -> str:
    """Extract the root scene identifier by stripping explicit focal/scale/patch suffixes only.

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
    """Strip explicit LR/HR/focal/scale suffixes so LR and HR filenames align 1-to-1."""
    stem = Path(filename_or_stem).stem
    while True:
        updated = ROLE_SUFFIX_RE.sub("", stem)
        if updated == stem or not updated:
            break
        stem = updated
    return stem


def _resolve_split_root(data_dir: Path, split: str, scale: str) -> Path:
    """Locate the target split directory while never falling back into another split."""
    root = resolve_realce_root(data_dir)
    scale_clean = scale.lstrip("x")
    scale_candidates = [scale, f"Real-CE_{scale}", f"x{scale_clean}"]
    candidates = [
        root / split,
        *(root / split / sc for sc in scale_candidates),
        *(root / sc / split for sc in scale_candidates),
    ]
    for cand in candidates:
        if cand.is_dir():
            return cand
    raise FileNotFoundError(
        f"Could not find split directory '{split}' under {data_dir} (resolved root: {root}). Checked: {[str(c) for c in candidates]}"
    )


def _classify_lr_hr(rel_parts: tuple[str, ...], scale: str = "x4") -> str | None:
    """Classify an image's role ('LR' or 'HR') from its path components below split_root."""
    lowered = [p.lower() for p in rel_parts[:-1]]
    if any(tok in ANNOTATION_DIR_TOKENS or tok.startswith("trans_") for tok in lowered):
        return None

    scale_clean = scale.lower().lstrip("x")
    lr_focal = "13mm" if scale_clean == "4" else "26mm"
    ignored_focal = "26mm" if scale_clean == "4" else "13mm"

    # Real-CE focal-length directory layout: 13mm (4x LR), 26mm (2x LR), 52mm (HR)
    if any(tok == ignored_focal for tok in lowered):
        return None
    if any(tok == lr_focal for tok in lowered):
        return "LR"
    if any(tok == "52mm" for tok in lowered):
        return "HR"

    # Generic fallback for LR/HR layouts
    is_lr = any(tok in LR_DIR_TOKENS or tok.startswith("lr") or tok.endswith("_lr") for tok in lowered)
    is_hr = any(tok in HR_DIR_TOKENS or tok.startswith("hr") or tok.endswith("_hr") for tok in lowered)
    if is_lr and not is_hr:
        return "LR"
    if is_hr and not is_lr:
        return "HR"

    stem_lower = Path(rel_parts[-1]).stem.lower()
    if ROLE_SUFFIX_RE.search(stem_lower):
        if any(tag in stem_lower for tag in (f"_{lr_focal}", f"-{lr_focal}", "_lr", "-lr", "_lq")):
            return "LR"
        if any(tag in stem_lower for tag in ("_52mm", "-52mm", "_hr", "-hr", "_gt", "_hq")):
            return "HR"
    return None


def _rel_parent_context(rel_parts: tuple[str, ...]) -> str:
    """Return non-role relative parent segments so duplicate stems in different subdirs stay distinct."""
    focal_tokens = {"13mm", "26mm", "52mm"}
    kept = []
    for part in rel_parts[:-1]:
        low = part.lower()
        if (
            low in focal_tokens
            or low in LR_DIR_TOKENS
            or low in HR_DIR_TOKENS
            or low.startswith(("lr", "hr", "gt", "lq", "hq"))
        ):
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
        role = _classify_lr_hr(rel_parts, scale=scale)
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
            f"Could not find both LR and HR images under {split_root} for scale={scale} "
            f"(found {len(lr_entries)} LR, {len(hr_entries)} HR)"
        )

    missing_hr = sorted(set(lr_entries) - set(hr_entries))
    missing_lr = sorted(set(hr_entries) - set(lr_entries))
    if missing_hr or missing_lr:
        raise ValueError(
            f"Unpaired images under {split_root}: missing_hr={missing_hr[:5]} (total {len(missing_hr)}), "
            f"missing_lr={missing_lr[:5]} (total {len(missing_lr)})"
        )

    # Apply valid.txt filter only when its entries map to discovered pair stems
    valid_entries = parse_valid_txt(split_root)
    valid_filter: set[str] | None = None
    if valid_entries:
        valid_tokens = {
            canonical_pair_key(Path(line.split()[0].split(",")[0].strip()).name).lower()
            for line in valid_entries
            if line.split()
        }
        available_stems = {canonical_pair_key(pid.split("/")[-1]).lower() for pid in lr_entries}
        overlap = valid_tokens & available_stems
        if overlap:
            valid_filter = overlap

    pairs: list[PairedSample] = []
    for pair_id in sorted(lr_entries):
        base_name = pair_id.split("/")[-1]
        if valid_filter is not None and canonical_pair_key(base_name).lower() not in valid_filter:
            continue
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
    if not pairs:
        raise FileNotFoundError(f"Zero pairs matched after valid.txt filtering under {split_root}")
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


def _is_confidence_score(token: str, *, num_parts: int) -> bool:
    """Detect optional float OCR confidence field preceding a transcript."""
    tok = token.strip()
    if not tok or num_parts < 10:
        return False
    if "." not in tok and "e" not in tok.lower() and tok not in {"0", "1"}:
        return False
    try:
        val = float(tok)
    except ValueError:
        return False
    return 0.0 <= val <= 1.0


def build_annotation_index(roots: list[Path]) -> dict[str, Path]:
    """Index annotation .txt files across split roots respecting ANNO_DIR_PRIORITY."""
    candidate_dirs: list[tuple[int, Path]] = []
    seen_dirs: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.iterdir()):
            if not p.is_dir():
                continue
            low = p.name.lower()
            if low in ANNO_DIR_PRIORITY:
                prio = ANNO_DIR_PRIORITY.index(low)
            elif low in ANNOTATION_DIR_TOKENS or "anno" in low:
                prio = len(ANNO_DIR_PRIORITY)
            else:
                continue
            resolved = p.resolve()
            if resolved not in seen_dirs:
                seen_dirs.add(resolved)
                candidate_dirs.append((prio, p))

    candidate_dirs.sort(key=lambda item: (item[0], str(item[1]).lower()))
    index: dict[str, Path] = {}
    for _, anno_dir in candidate_dirs:
        for txt_path in sorted(anno_dir.rglob("*.txt")):
            if txt_path.name.lower() == "valid.txt":
                continue
            stem_low = txt_path.stem.lower()
            clean_low = stem_low[4:] if stem_low.startswith("res_") else stem_low
            canon_low = canonical_pair_key(clean_low).lower()
            for key in (stem_low, clean_low, canon_low):
                if key:
                    index.setdefault(key, txt_path)
    return index


def _lookup_annotation_file(pair: PairedSample, anno_index: dict[str, Path]) -> Path | None:
    """Resolve annotation file for a paired sample (preferring HR 52mm annotations)."""
    hr_stem = pair.hr_path.stem.lower()
    lr_stem = pair.lr_path.stem.lower()
    candidates = [
        hr_stem,
        f"res_{hr_stem}",
        canonical_pair_key(pair.hr_path.name).lower(),
        lr_stem,
        canonical_pair_key(pair.lr_path.name).lower(),
    ]
    for key in candidates:
        if key in anno_index:
            return anno_index[key]
    return None


def load_realce_hr_boxes(anno_path: Path | None, img_w: int, img_h: int) -> list[TextBox]:
    """Parse 8-point Real-CE quadrilateral polygon annotations into clamped HR boxes."""
    if anno_path is None or not anno_path.is_file() or img_w <= 0 or img_h <= 0:
        return []
    boxes: list[TextBox] = []
    text = anno_path.read_text(encoding="utf-8-sig", errors="replace")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 9:
            continue
        try:
            coords = [float(tok.strip()) for tok in parts[:8]]
        except ValueError:
            continue
        if any(not math.isfinite(c) for c in coords):
            continue

        if len(parts) >= 10 and _is_confidence_score(parts[8], num_parts=len(parts)):
            transcript = ",".join(parts[9:]).strip()
        else:
            transcript = ",".join(parts[8:]).strip()

        if not transcript or set(transcript) <= {"#", "*"} or transcript.lower() in {"???", "n/a", "null"}:
            continue

        xs = coords[0::2]
        ys = coords[1::2]
        x1 = max(0, min(img_w, int(math.floor(min(xs)))))
        y1 = max(0, min(img_h, int(math.floor(min(ys)))))
        x2 = max(0, min(img_w, int(math.ceil(max(xs)))))
        y2 = max(0, min(img_h, int(math.ceil(max(ys)))))
        w = x2 - x1
        h = y2 - y1
        if w < 8 or h < 8:
            continue
        boxes.append(TextBox(x=x1, y=y1, w=w, h=h, transcript=transcript))
    return boxes


def pad_edge_right_bottom(im: Any, target_w: int, target_h: int) -> Any:
    """Edge-replicate an RGB image on the right/bottom when smaller than target size."""
    from PIL import Image

    w, h = im.size
    new_w = max(w, target_w)
    new_h = max(h, target_h)
    if new_w == w and new_h == h:
        return im
    near = getattr(getattr(Image, "Resampling", Image), "NEAREST")
    canvas = Image.new("RGB", (new_w, new_h))
    canvas.paste(im, (0, 0))
    if new_w > w:
        right_edge = im.crop((w - 1, 0, w, h)).resize((new_w - w, h), resample=near)
        canvas.paste(right_edge, (w, 0))
    if new_h > h:
        bottom_edge = canvas.crop((0, h - 1, new_w, h)).resize((new_w, new_h - h), resample=near)
        canvas.paste(bottom_edge, (0, h))
    return canvas


def _is_ascii_alnum_text(text: str) -> bool:
    return any(c.isascii() and c.isalnum() for c in text)


def _window_iou(a: tuple[int, int], b: tuple[int, int], size: int) -> float:
    ax, ay = a
    bx, by = b
    ix = max(0, min(ax + size, bx + size) - max(ax, bx))
    iy = max(0, min(ay + size, by + size) - max(ay, by))
    inter = ix * iy
    if inter == 0:
        return 0.0
    union = 2 * size * size - inter
    return inter / float(union)


def _enclosed_boxes(
    boxes: list[TextBox],
    left: int,
    top: int,
    patch_size: int,
) -> list[tuple[int, TextBox]]:
    right = left + patch_size
    bottom = top + patch_size
    return [
        (idx, b)
        for idx, b in enumerate(boxes)
        if b.x >= left and b.y >= top and (b.x + b.w) <= right and (b.y + b.h) <= bottom
    ]


def _clamp_snap(val: int, max_val: int, snap: int) -> int:
    clamped = max(0, min(max_val, val))
    if snap > 1:
        clamped = min(max_val, (clamped // snap) * snap)
    return clamped


def _generate_candidate_windows(
    eff_hr_w: int,
    eff_hr_h: int,
    boxes: list[TextBox],
    *,
    hr_patch_size: int,
    stride: int,
    snap: int,
    uncapped: bool,
) -> list[tuple[int, int]]:
    max_left = max(0, eff_hr_w - hr_patch_size)
    max_top = max(0, eff_hr_h - hr_patch_size)
    seen: set[tuple[int, int]] = set()
    windows: list[tuple[int, int]] = []

    def add_win(lx: int, ty: int) -> None:
        win = (_clamp_snap(lx, max_left, snap), _clamp_snap(ty, max_top, snap))
        if win not in seen:
            seen.add(win)
            windows.append(win)

    if uncapped:
        for top_hr in range(0, max_top + 1, stride):
            for left_hr in range(0, max_left + 1, stride):
                add_win(left_hr, top_hr)
        if not windows:
            add_win(0, 0)
        return windows

    for b in boxes:
        cx = b.x + b.w // 2
        cy = b.y + b.h // 2
        add_win(cx - hr_patch_size // 2, cy - hr_patch_size // 2)
        add_win(b.x - 16, b.y - 16)

    add_win((eff_hr_w - hr_patch_size) // 2, (eff_hr_h - hr_patch_size) // 2)
    add_win(0, 0)
    grid_step = stride
    if max_left > 0 and max_top > 0:
        est = ((max_left // grid_step) + 1) * ((max_top // grid_step) + 1)
        if est > 49:
            grid_step = max(stride, ((max(eff_hr_w, eff_hr_h) // 5) // snap or 1) * snap)
    for top_hr in range(0, max_top + 1, grid_step):
        for left_hr in range(0, max_left + 1, grid_step):
            add_win(left_hr, top_hr)
    return windows


@dataclass
class _WindowCandidate:
    pair: PairedSample
    lr_size: tuple[int, int]
    hr_size: tuple[int, int]
    eff_hr_size: tuple[int, int]
    exact_scale: bool
    left_hr: int
    top_hr: int
    enclosed: list[tuple[int, TextBox]]
    base_score: tuple[int, int, int, int, int]


def _score_candidate(
    cand: _WindowCandidate,
    *,
    covered_box_keys: set[tuple[str, int]],
    picked_windows_on_pair: list[tuple[int, int]],
    hr_patch_size: int,
) -> tuple[int, int, int, int, int, int, int, int, int]:
    non_overlap = 1
    if picked_windows_on_pair:
        max_iou = max(_window_iou((cand.left_hr, cand.top_hr), prev, hr_patch_size) for prev in picked_windows_on_pair)
        if max_iou > 0.35:
            non_overlap = 0

    new_ascii = 0
    new_total = 0
    for idx, box in cand.enclosed:
        if (cand.pair.pair_id, idx) not in covered_box_keys:
            new_total += 1
            if _is_ascii_alnum_text(box.transcript):
                new_ascii += 1

    eff_w, eff_h = cand.eff_hr_size
    win_cx = cand.left_hr + hr_patch_size // 2
    win_cy = cand.top_hr + hr_patch_size // 2
    center_dist = abs(win_cx - eff_w // 2) + abs(win_cy - eff_h // 2)

    all_ascii, all_total, total_chars = cand.base_score[0], cand.base_score[1], cand.base_score[2]
    return (
        non_overlap,
        new_ascii,
        new_total,
        all_ascii,
        all_total,
        total_chars,
        -center_dist,
        -cand.top_hr,
        -cand.left_hr,
    )


def extract_aligned_patches(
    split_pairs: dict[str, list[PairedSample]],
    *,
    patch_dir: Path,
    hr_patch_size: int = 512,
    lr_patch_size: int = 128,
    stride: int = 256,
    split_root: Path | None = None,
    max_patches_per_scene: int = 2,
    max_val_patches: int = 30,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Extract text-prioritized aligned patches + patch-local box CSVs strictly inside each split.

    Handles HR images smaller than hr_patch_size by right/bottom edge replication so (0, 0)
    coordinates and box annotations remain exact, supports both 4x-ratio and pre-registered
    same-canvas focal captures, and avoids dropping small sign images.
    """
    from PIL import Image

    if hr_patch_size <= 0 or lr_patch_size <= 0 or stride <= 0:
        raise ValueError("Patch sizes and stride must be positive integers")
    if hr_patch_size % lr_patch_size != 0:
        raise ValueError(f"hr_patch_size ({hr_patch_size}) must be divisible by lr_patch_size ({lr_patch_size})")
    scale = hr_patch_size // lr_patch_size
    if stride % scale != 0:
        raise ValueError(f"HR stride ({stride}) must be divisible by scale ({scale})")

    search_roots: list[Path] = []
    if split_root is not None:
        search_roots.append(split_root)
    for subset_pairs in split_pairs.values():
        for pair in subset_pairs:
            p_root = pair.hr_path.parent.parent
            if p_root not in search_roots:
                search_roots.append(p_root)
    anno_index = build_annotation_index(search_roots)

    padded_count = 0
    patch_records: list[dict[str, Any]] = []
    box_rows_by_subset: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    bicubic = getattr(getattr(Image, "Resampling", Image), "BICUBIC")

    for subset in ("train", "val"):
        lr_out_dir = patch_dir / subset / "LR"
        hr_out_dir = patch_dir / subset / "HR"
        lr_out_dir.mkdir(parents=True, exist_ok=True)
        hr_out_dir.mkdir(parents=True, exist_ok=True)

        if subset == "train":
            uncapped = max_patches_per_scene <= 0
            per_scene_limit = max_patches_per_scene
        else:
            uncapped = max_val_patches <= 0 and max_patches_per_scene <= 0
            per_scene_limit = 1 if not uncapped else 0

        pairs_by_scene: dict[str, list[PairedSample]] = {}
        for pair in split_pairs[subset]:
            pairs_by_scene.setdefault(pair.scene_key, []).append(pair)

        chosen_for_subset: list[tuple[_WindowCandidate, tuple[int, int, int, int, int]]] = []
        for scene_key in sorted(pairs_by_scene):
            scene_cands: list[_WindowCandidate] = []
            for pair in pairs_by_scene[scene_key]:
                with Image.open(pair.lr_path) as lr_hdr:
                    lr_w, lr_h = lr_hdr.size
                with Image.open(pair.hr_path) as hr_hdr:
                    hr_w, hr_h = hr_hdr.size
                if lr_w <= 0 or lr_h <= 0 or hr_w <= 0 or hr_h <= 0:
                    continue

                eff_hr_w = max(hr_w, hr_patch_size)
                eff_hr_h = max(hr_h, hr_patch_size)
                if eff_hr_w > hr_w or eff_hr_h > hr_h:
                    padded_count += 1
                eff_lr_w = max(lr_patch_size, int(round(eff_hr_w * lr_w / hr_w))) if (eff_hr_w != hr_w or eff_hr_h != hr_h) else lr_w
                eff_lr_h = max(lr_patch_size, int(round(eff_hr_h * lr_h / hr_h))) if (eff_hr_w != hr_w or eff_hr_h != hr_h) else lr_h
                exact_scale = (eff_hr_w == eff_lr_w * scale) and (eff_hr_h == eff_lr_h * scale)
                snap = scale if exact_scale else 1

                anno_path = _lookup_annotation_file(pair, anno_index)
                boxes = load_realce_hr_boxes(anno_path, hr_w, hr_h)
                windows = _generate_candidate_windows(
                    eff_hr_w,
                    eff_hr_h,
                    boxes,
                    hr_patch_size=hr_patch_size,
                    stride=stride,
                    snap=snap,
                    uncapped=uncapped,
                )
                for left_hr, top_hr in windows:
                    enclosed = _enclosed_boxes(boxes, left_hr, top_hr, hr_patch_size)
                    all_ascii = sum(1 for _, b in enclosed if _is_ascii_alnum_text(b.transcript))
                    all_total = len(enclosed)
                    total_chars = sum(min(len(b.transcript), 32) for _, b in enclosed)
                    scene_cands.append(
                        _WindowCandidate(
                            pair=pair,
                            lr_size=(lr_w, lr_h),
                            hr_size=(hr_w, hr_h),
                            eff_hr_size=(eff_hr_w, eff_hr_h),
                            exact_scale=exact_scale,
                            left_hr=left_hr,
                            top_hr=top_hr,
                            enclosed=enclosed,
                            base_score=(all_ascii, all_total, total_chars, -top_hr, -left_hr),
                        )
                    )

            if not scene_cands:
                continue

            if uncapped:
                for cand in scene_cands:
                    chosen_for_subset.append((cand, cand.base_score))
                continue

            remaining = list(scene_cands)
            covered_boxes: set[tuple[str, int]] = set()
            picked_by_pair: dict[str, list[tuple[int, int]]] = {}
            for _ in range(per_scene_limit):
                if not remaining:
                    break
                best_idx = max(
                    range(len(remaining)),
                    key=lambda idx: _score_candidate(
                        remaining[idx],
                        covered_box_keys=covered_boxes,
                        picked_windows_on_pair=picked_by_pair.get(remaining[idx].pair.pair_id, []),
                        hr_patch_size=hr_patch_size,
                    ),
                )
                best_cand = remaining.pop(best_idx)
                chosen_for_subset.append((best_cand, best_cand.base_score))
                picked_by_pair.setdefault(best_cand.pair.pair_id, []).append(
                    (best_cand.left_hr, best_cand.top_hr)
                )
                for b_idx, _ in best_cand.enclosed:
                    covered_boxes.add((best_cand.pair.pair_id, b_idx))

        if subset == "val" and max_val_patches > 0 and len(chosen_for_subset) > max_val_patches:
            chosen_for_subset.sort(
                key=lambda item: (
                    item[1][0],
                    item[1][1],
                    item[1][2],
                    item[0].pair.scene_key,
                    item[0].pair.pair_id,
                    item[0].top_hr,
                    item[0].left_hr,
                ),
                reverse=True,
            )
            chosen_for_subset = chosen_for_subset[:max_val_patches]
            chosen_for_subset.sort(key=lambda item: (item[0].pair.pair_id, item[0].top_hr, item[0].left_hr))

        grouped_to_extract: dict[str, list[_WindowCandidate]] = {}
        for cand, _ in chosen_for_subset:
            grouped_to_extract.setdefault(cand.pair.pair_id, []).append(cand)

        for pair_id in sorted(grouped_to_extract):
            cands = grouped_to_extract[pair_id]
            pair = cands[0].pair
            safe_stem = pair.pair_id.replace("/", "__")
            with Image.open(pair.lr_path) as lr_im_raw, Image.open(pair.hr_path) as hr_im_raw:
                lr_im = lr_im_raw.convert("RGB")
                hr_im = hr_im_raw.convert("RGB")
                lr_w, lr_h = lr_im.size
                hr_w, hr_h = hr_im.size
                eff_hr_w = max(hr_w, hr_patch_size)
                eff_hr_h = max(hr_h, hr_patch_size)
                if eff_hr_w != hr_w or eff_hr_h != hr_h:
                    target_lr_w = max(lr_patch_size, int(round(eff_hr_w * lr_w / hr_w)))
                    target_lr_h = max(lr_patch_size, int(round(eff_hr_h * lr_h / hr_h)))
                    hr_im = pad_edge_right_bottom(hr_im, eff_hr_w, eff_hr_h)
                    lr_im = pad_edge_right_bottom(lr_im, target_lr_w, target_lr_h)
                eff_lr_w, eff_lr_h = lr_im.size

                for crop_idx, cand in enumerate(cands):
                    left_hr = cand.left_hr
                    top_hr = cand.top_hr
                    hr_crop = hr_im.crop((left_hr, top_hr, left_hr + hr_patch_size, top_hr + hr_patch_size))
                    if cand.exact_scale:
                        top_lr = top_hr // scale
                        left_lr = left_hr // scale
                        lr_crop = lr_im.crop((left_lr, top_lr, left_lr + lr_patch_size, top_lr + lr_patch_size))
                    else:
                        left_lr = max(0, min(eff_lr_w - 1, int(round(left_hr * eff_lr_w / eff_hr_w))))
                        top_lr = max(0, min(eff_lr_h - 1, int(round(top_hr * eff_lr_h / eff_hr_h))))
                        right_lr = max(left_lr + 1, min(eff_lr_w, int(round((left_hr + hr_patch_size) * eff_lr_w / eff_hr_w))))
                        bottom_lr = max(top_lr + 1, min(eff_lr_h, int(round((top_hr + hr_patch_size) * eff_lr_h / eff_hr_h))))
                        lr_crop = lr_im.crop((left_lr, top_lr, right_lr, bottom_lr))
                        if lr_crop.size != (lr_patch_size, lr_patch_size):
                            lr_crop = lr_crop.resize((lr_patch_size, lr_patch_size), bicubic)

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
                    for box_id, (_, box) in enumerate(cand.enclosed, start=1):
                        box_rows_by_subset[subset].append(
                            {
                                "image": crop_name,
                                "box_id": box_id,
                                "x": box.x - left_hr,
                                "y": box.y - top_hr,
                                "w": box.w,
                                "h": box.h,
                                "transcript": box.transcript,
                                "readable": "yes",
                            }
                        )

    if padded_count:
        print(f"Note: edge-padded {padded_count} pair(s) smaller than HR patch size {hr_patch_size}x{hr_patch_size} on right/bottom.")
    return patch_records, box_rows_by_subset


def write_pair_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write split paired sample paths to CSV for LoRA training/validation scripts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["split", "scene_key", "pair_id", "lr_path", "hr_path"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_boxes_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write patch-local bounding boxes CSV using the text_eval / run_lora_trial schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=BOXES_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Split Real-CE train subset by scene with zero leakage.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/Real-CE"), help="Real-CE dataset root directory")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to partition (default: train; val stays untouched)")
    parser.add_argument("--scale", type=str, default="x4", help="Focal scale pair (default: x4 -> 13mm LR + 52mm HR; x2 -> 26mm LR + 52mm HR)")
    parser.add_argument("--val-ratio", type=float, default=0.20, help="Fraction of scenes assigned to validation (default: 0.20)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for scene shuffling (default: 42)")
    parser.add_argument("--output-manifest", type=Path, default=Path("data/realce_splits_manifest.json"), help="Output JSON manifest")
    parser.add_argument("--output-train-csv", type=Path, default=Path("data/realce_train_manifest.csv"), help="Output paired train/val CSV")
    parser.add_argument("--output-train-boxes", type=Path, default=Path("data/realce_train_boxes.csv"), help="Output training patch box CSV")
    parser.add_argument("--output-val-boxes", type=Path, default=Path("data/realce_val_boxes.csv"), help="Output validation patch box CSV")
    parser.add_argument("--extract-patches", action="store_true", help="Extract aligned LR/HR patches strictly after scene split")
    parser.add_argument("--patch-dir", type=Path, default=None, help="Output directory for extracted patches (required when --extract-patches)")
    parser.add_argument("--hr-patch-size", type=int, default=512, help="HR patch size in pixels (default: 512)")
    parser.add_argument("--lr-patch-size", type=int, default=128, help="LR patch size in pixels (default: 128)")
    parser.add_argument("--stride", type=int, default=256, help="HR sliding-window stride in pixels (default: 256)")
    parser.add_argument("--max-patches-per-scene", type=int, default=2, help="Max text-centered patches per train scene (0 = uncapped sliding windows; default: 2)")
    parser.add_argument("--max-val-patches", type=int, default=30, help="Max text-centered validation patches across distinct val scenes (0 = uncapped; default: 30)")
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

    split_root = _resolve_split_root(args.data_dir, args.split, args.scale)
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
    box_rows_by_subset: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    if args.extract_patches:
        patch_dir = args.patch_dir if args.patch_dir is not None else (args.output_manifest.parent / "realce_patches")
        patch_records, box_rows_by_subset = extract_aligned_patches(
            {"train": train_pairs, "val": val_pairs},
            patch_dir=patch_dir,
            hr_patch_size=args.hr_patch_size,
            lr_patch_size=args.lr_patch_size,
            stride=args.stride,
            split_root=split_root,
            max_patches_per_scene=args.max_patches_per_scene,
            max_val_patches=args.max_val_patches,
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
            "train_box_count": len(box_rows_by_subset["train"]),
            "val_box_count": len(box_rows_by_subset["val"]),
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

    if args.extract_patches:
        write_boxes_csv(args.output_train_boxes, box_rows_by_subset["train"])
        print(f"Saved {len(box_rows_by_subset['train'])} train patch box(es) to {args.output_train_boxes}")
        write_boxes_csv(args.output_val_boxes, box_rows_by_subset["val"])
        print(f"Saved {len(box_rows_by_subset['val'])} val patch box(es) to {args.output_val_boxes}")
    else:
        for box_out, label in ((args.output_train_boxes, "train"), (args.output_val_boxes, "validation")):
            box_out.parent.mkdir(parents=True, exist_ok=True)
            if not box_out.exists():
                write_boxes_csv(box_out, [])
                print(f"Initialized {label} boxes CSV header at {box_out}")

    print("Done. Scene-level split completed with 0 data leakage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
