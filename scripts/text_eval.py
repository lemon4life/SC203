#!/usr/bin/env python3
"""NOT TESTED: evaluate text restoration accuracy and seed consistency inside labeled HR text boxes.

Designed for CPU or GPU execution on Windows, macOS, or Google Colab with no hard-coded /content paths.
- Keeps the OCR backend choice in ONE place (DEFAULT_OCR_BACKEND / run_ocr_on_crop).
- Uses the exact same text normalization rules defined in docs/handoff_member2.md.
- Treats the IMAGE as the primary unit of statistical analysis (the 10 seeds of one image are not independent)
  and pairs comparisons on (image, seed) pairs when --compare-study-root is provided.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import string
import sys
import unicodedata
from pathlib import Path
from typing import Any

CSV_HEADER = ["image", "box_id", "x", "y", "w", "h", "transcript", "readable"]

# ============================================================================
# SINGLE PLACE FOR OCR CHOICE (Update DEFAULT_OCR_BACKEND once user chooses)
# Supported values: "rapidocr", "easyocr", "tesseract"
# ============================================================================
DEFAULT_OCR_BACKEND = "rapidocr"
VALID_OCR_BACKENDS = ("rapidocr", "easyocr", "tesseract")
_OCR_ENGINE_CACHE: dict[str, Any] = {}


def normalize_text(text: str, *, case_sensitive: bool = False) -> str:
    """Normalize text following docs/handoff_member2.md Section 5 rules.

    1. Unicode NFKC normalization.
    2. Strip leading/trailing whitespace and ASCII punctuation.
    3. Collapse internal whitespace runs to a single space.
    4. Lowercase by default (case_sensitive=False for primary metric; True for secondary).
    """
    norm = unicodedata.normalize("NFKC", text or "")
    norm = norm.strip().strip(string.punctuation).strip()
    norm = re.sub(r"\s+", " ", norm)
    if not case_sensitive:
        norm = norm.lower()
    return norm


def levenshtein_distance(a: str, b: str) -> int:
    """Compute exact character-level Levenshtein edit distance between strings a and b."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def compute_cer(pred: str, gt: str) -> float:
    """Compute Character Error Rate: edit_distance(pred, gt) / max(1, len(gt))."""
    return float(levenshtein_distance(pred, gt)) / float(max(1, len(gt)))


def get_ocr_engine(backend: str, device: str = "cpu") -> Any:
    """Lazy-initialize the chosen OCR engine in one place."""
    key = f"{backend}:{device}"
    if key in _OCR_ENGINE_CACHE:
        return _OCR_ENGINE_CACHE[key]

    if backend == "rapidocr":
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise RuntimeError(
                "OCR backend 'rapidocr' requires rapidocr-onnxruntime. "
                "Install with: pip install rapidocr-onnxruntime"
            ) from exc
        engine = RapidOCR()
    elif backend == "easyocr":
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError(
                "OCR backend 'easyocr' requires easyocr. "
                "Install with: pip install easyocr"
            ) from exc
        use_gpu = device.startswith("cuda")
        engine = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
    elif backend == "tesseract":
        try:
            import pytesseract
        except ImportError as exc:
            raise RuntimeError(
                "OCR backend 'tesseract' requires pytesseract and the system tesseract binary. "
                "Install with: pip install pytesseract"
            ) from exc
        engine = pytesseract
    else:
        raise ValueError(f"Unsupported OCR backend: {backend!r}")

    _OCR_ENGINE_CACHE[key] = engine
    return engine


def run_ocr_on_crop(crop_pil: Any, *, backend: str, device: str = "cpu") -> str:
    """Run the selected OCR backend on a cropped PIL RGB image and return raw text."""
    import numpy as np

    engine = get_ocr_engine(backend, device=device)
    rgb = crop_pil.convert("RGB")

    if backend == "rapidocr":
        arr = np.asarray(rgb)
        result, _ = engine(arr)
        if not result:
            return ""
        # Sort detected boxes left-to-right / top-to-bottom and join text spans
        texts = [str(item[1]) for item in result if len(item) >= 2 and item[1]]
        return " ".join(texts)

    if backend == "easyocr":
        arr = np.asarray(rgb)
        texts = engine.readtext(arr, detail=0, paragraph=True)
        if not texts:
            return ""
        return " ".join(str(t) for t in texts if t)

    if backend == "tesseract":
        # PSM 7 treats the cropped box as a single text line
        text = engine.image_to_string(rgb, config="--psm 7")
        return str(text or "").strip()

    raise ValueError(f"Unsupported OCR backend: {backend!r}")


def load_readable_boxes(csv_path: Path) -> list[dict[str, Any]]:
    """Load boxes from CSV, failing if any row has a blank readable column."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Text-box CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = [c.strip() for c in next(reader)]
        except StopIteration as exc:
            raise ValueError(f"CSV file is empty: {csv_path}") from exc
        if header != CSV_HEADER:
            raise ValueError(f"Invalid CSV header {header} in {csv_path}, expected {CSV_HEADER}")

        boxes: list[dict[str, Any]] = []
        blank_rows: list[str] = []
        for line_num, row in enumerate(reader, start=2):
            if not any(cell.strip() for cell in row):
                continue
            if len(row) != len(CSV_HEADER):
                raise ValueError(f"{csv_path}:{line_num}: expected 8 columns, got {len(row)}")
            image, box_id, x, y, w, h, transcript, readable = [c.strip() for c in row]
            r_low = readable.lower()
            if r_low == "":
                blank_rows.append(f"{image} (line {line_num})")
                continue
            if r_low not in {"yes", "no"}:
                raise ValueError(
                    f"{csv_path}:{line_num}: readable must be 'yes' or 'no', got {readable!r}"
                )
            if r_low == "yes":
                stem = Path(image).stem
                boxes.append(
                    {
                        "image": image,
                        "stem": stem,
                        "box_id": int(box_id),
                        "x": int(x),
                        "y": int(y),
                        "w": int(w),
                        "h": int(h),
                        "transcript": transcript,
                    }
                )

    if blank_rows:
        raise ValueError(
            f"CSV {csv_path} still has {len(blank_rows)} unlabeled row(s) with blank 'readable': "
            f"{blank_rows[:10]}. Complete labeling first."
        )
    if not boxes:
        raise ValueError(f"No rows with readable='yes' found in {csv_path}")
    return boxes


def discover_seed_dirs(study_root: Path) -> list[tuple[int, Path]]:
    if not study_root.is_dir():
        raise FileNotFoundError(f"Study root not found: {study_root}")
    seed_dirs: list[tuple[int, Path]] = []
    for child in study_root.iterdir():
        if child.is_dir() and child.name.startswith("seed_"):
            suffix = child.name.removeprefix("seed_")
            if suffix.isdigit():
                seed_dirs.append((int(suffix), child))
    seed_dirs.sort(key=lambda item: item[0])
    if not seed_dirs:
        raise ValueError(f"No seed_<N> directories found in {study_root}")
    return seed_dirs


def resolve_pred_image(seed_dir: Path, stem: str) -> Path:
    candidates = [
        seed_dir / f"{stem}_0.png",
        seed_dir / f"{stem}.png",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"Missing output image for stem {stem!r} in {seed_dir} (tried {[c.name for c in candidates]})"
    )


def evaluate_study(
    study_root: Path,
    boxes: list[dict[str, Any]],
    *,
    ocr_backend: str,
    device: str,
    expected_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate all (box, seed) crops and aggregate to image level."""
    import numpy as np
    from PIL import Image

    seed_dirs = discover_seed_dirs(study_root)
    per_box_seed_rows: list[dict[str, Any]] = []
    crops_by_box: dict[tuple[str, int], list[Any]] = {}
    preds_by_box: dict[tuple[str, int], list[str]] = {}

    for box in boxes:
        stem = box["stem"]
        bid = box["box_id"]
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        gt_raw = box["transcript"]
        gt_norm = normalize_text(gt_raw, case_sensitive=False)
        gt_cs = normalize_text(gt_raw, case_sensitive=True)

        crops_by_box[(stem, bid)] = []
        preds_by_box[(stem, bid)] = []

        for seed, seed_dir in seed_dirs:
            img_path = resolve_pred_image(seed_dir, stem)
            with Image.open(img_path) as im:
                if expected_size > 0 and im.size != (expected_size, expected_size):
                    raise ValueError(
                        f"Output image {img_path} has size {im.size}, expected {(expected_size, expected_size)}"
                    )
                rgb = im.convert("RGB")
                crop = rgb.crop((x, y, x + w, y + h))
                crop_arr = np.asarray(crop, dtype=np.float32)

            raw_ocr = run_ocr_on_crop(crop, backend=ocr_backend, device=device)
            pred_norm = normalize_text(raw_ocr, case_sensitive=False)
            pred_cs = normalize_text(raw_ocr, case_sensitive=True)

            wrong_char = 0 if pred_norm == gt_norm else 1
            wrong_char_cs = 0 if pred_cs == gt_cs else 1
            cer = compute_cer(pred_norm, gt_norm)
            cer_cs = compute_cer(pred_cs, gt_cs)

            crops_by_box[(stem, bid)].append(crop_arr)
            preds_by_box[(stem, bid)].append(pred_norm)

            per_box_seed_rows.append(
                {
                    "image": box["image"],
                    "stem": stem,
                    "box_id": bid,
                    "seed": seed,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "gt_transcript": gt_raw,
                    "gt_norm": gt_norm,
                    "ocr_raw": raw_ocr,
                    "ocr_norm": pred_norm,
                    "wrong_char": wrong_char,
                    "wrong_char_yes_no": "yes" if wrong_char else "no",
                    "cer": cer,
                    "wrong_char_case_sensitive": wrong_char_cs,
                    "cer_case_sensitive": cer_cs,
                }
            )

    # Compute across-seed disagreement per box, then aggregate to IMAGE level
    box_disagreement: dict[tuple[str, int], dict[str, float]] = {}
    for key, norm_preds in preds_by_box.items():
        n = len(norm_preds)
        pairs = 0
        diff_pairs = 0
        pixel_diffs: list[float] = []
        arrs = crops_by_box[key]
        for i in range(n):
            for j in range(i + 1, n):
                pairs += 1
                if norm_preds[i] != norm_preds[j]:
                    diff_pairs += 1
                pixel_diffs.append(float(np.abs(arrs[i] - arrs[j]).mean()))
        box_disagreement[key] = {
            "ocr_seed_disagreement_rate": (diff_pairs / pairs) if pairs > 0 else 0.0,
            "num_distinct_ocr_across_seeds": float(len(set(norm_preds))),
            "crop_pixel_seed_spread_l1": (sum(pixel_diffs) / len(pixel_diffs)) if pixel_diffs else 0.0,
        }

    # Group by IMAGE (statistical unit)
    rows_by_image: dict[str, list[dict[str, Any]]] = {}
    for row in per_box_seed_rows:
        rows_by_image.setdefault(row["stem"], []).append(row)

    image_summary_rows: list[dict[str, Any]] = []
    for stem in sorted(rows_by_image):
        img_rows = rows_by_image[stem]
        box_ids = sorted({r["box_id"] for r in img_rows})
        mean_wrong = sum(r["wrong_char"] for r in img_rows) / len(img_rows)
        mean_cer = sum(r["cer"] for r in img_rows) / len(img_rows)
        mean_disagree = sum(
            box_disagreement[(stem, bid)]["ocr_seed_disagreement_rate"] for bid in box_ids
        ) / len(box_ids)
        mean_distinct = sum(
            box_disagreement[(stem, bid)]["num_distinct_ocr_across_seeds"] for bid in box_ids
        ) / len(box_ids)
        mean_px_spread = sum(
            box_disagreement[(stem, bid)]["crop_pixel_seed_spread_l1"] for bid in box_ids
        ) / len(box_ids)

        image_summary_rows.append(
            {
                "image": img_rows[0]["image"],
                "stem": stem,
                "num_boxes": len(box_ids),
                "num_seeds": len(seed_dirs),
                "wrong_char_rate": mean_wrong,
                "cer_mean": mean_cer,
                "ocr_seed_disagreement_rate": mean_disagree,
                "num_distinct_ocr_mean": mean_distinct,
                "crop_pixel_seed_spread_l1": mean_px_spread,
            }
        )

    return per_box_seed_rows, image_summary_rows


def write_csv_dicts(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_and_se(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    m = sum(values) / len(values)
    if len(values) < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
    return m, math.sqrt(var / len(values))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-root",
        type=Path,
        required=True,
        help="Path to a seed study output folder containing seed_<N>/ directories",
    )
    parser.add_argument(
        "--boxes-csv",
        type=Path,
        default=Path("data/text_boxes_realsr.csv"),
        help="Path to the labeled text-box CSV (default: data/text_boxes_realsr.csv)",
    )
    parser.add_argument(
        "--compare-study-root",
        type=Path,
        default=None,
        help="Optional second seed study root for paired (image, seed) comparison (e.g., empty vs default prompt)",
    )
    parser.add_argument(
        "--ocr-backend",
        choices=VALID_OCR_BACKENDS,
        default=DEFAULT_OCR_BACKEND,
        help=f"OCR backend (default: {DEFAULT_OCR_BACKEND})",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for OCR backend ('cpu' or 'cuda'; default: 'cpu')",
    )
    parser.add_argument(
        "--expected-size",
        type=int,
        default=512,
        help="Expected output image width/height in pixels (default: 512)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write text evaluation CSVs (defaults to --study-root)",
    )
    args = parser.parse_args()

    study_root = args.study_root.expanduser().resolve()
    boxes_csv = args.boxes_csv.expanduser().resolve()
    out_dir = (args.output_dir or study_root).expanduser().resolve()

    boxes = load_readable_boxes(boxes_csv)
    per_seed_rows, image_rows = evaluate_study(
        study_root,
        boxes,
        ocr_backend=args.ocr_backend,
        device=args.device,
        expected_size=args.expected_size,
    )

    per_seed_csv = out_dir / "text_eval_per_box_seed.csv"
    image_csv = out_dir / "text_eval_summary_by_image.csv"
    write_csv_dicts(per_seed_csv, per_seed_rows)
    write_csv_dicts(image_csv, image_rows)

    n_img = len(image_rows)
    wrong_m, wrong_se = mean_and_se([r["wrong_char_rate"] for r in image_rows])
    cer_m, cer_se = mean_and_se([r["cer_mean"] for r in image_rows])
    dis_m, dis_se = mean_and_se([r["ocr_seed_disagreement_rate"] for r in image_rows])

    print(
        f"TEXT EVAL SUMMARY (unit = IMAGE, n_images={n_img}, boxes={len(boxes)}, backend={args.ocr_backend})"
    )
    print(f"  Share with >=1 wrong char : {wrong_m:.4f} +/- {wrong_se:.4f} (SE across images)")
    print(f"  Character Error Rate (CER): {cer_m:.4f} +/- {cer_se:.4f} (SE across images)")
    print(f"  Seed OCR disagreement rate: {dis_m:.4f} +/- {dis_se:.4f} (SE across images)")
    print(f"  Wrote: {per_seed_csv}")
    print(f"  Wrote: {image_csv}")

    if args.compare_study_root is not None:
        comp_root = args.compare_study_root.expanduser().resolve()
        comp_seed_rows, comp_img_rows = evaluate_study(
            comp_root,
            boxes,
            ocr_backend=args.ocr_backend,
            device=args.device,
            expected_size=args.expected_size,
        )
        # Pair on exact (stem, box_id, seed) first, then average within each IMAGE (unit of analysis)
        base_map = {(r["stem"], r["box_id"], r["seed"]): r for r in per_seed_rows}
        comp_map = {(r["stem"], r["box_id"], r["seed"]): r for r in comp_seed_rows}
        shared_keys = sorted(set(base_map).intersection(comp_map))
        if len(shared_keys) != len(base_map) or len(shared_keys) != len(comp_map):
            raise ValueError(
                "Paired comparison requires identical (image, box_id, seed) pairs in both study roots"
            )

        diffs_by_img: dict[str, list[tuple[float, float]]] = {}
        for key in shared_keys:
            rb = base_map[key]
            rc = comp_map[key]
            diffs_by_img.setdefault(rb["stem"], []).append(
                (rc["wrong_char"] - rb["wrong_char"], rc["cer"] - rb["cer"])
            )

        paired_rows: list[dict[str, Any]] = []
        for stem in sorted(diffs_by_img):
            pairs = diffs_by_img[stem]
            d_wrong = sum(p[0] for p in pairs) / len(pairs)
            d_cer = sum(p[1] for p in pairs) / len(pairs)
            paired_rows.append(
                {
                    "stem": stem,
                    "num_paired_box_seeds": len(pairs),
                    "delta_wrong_char_rate_compare_minus_base": d_wrong,
                    "delta_cer_compare_minus_base": d_cer,
                }
            )
        paired_csv = out_dir / "text_eval_paired_comparison_by_image.csv"
        write_csv_dicts(paired_csv, paired_rows)
        dw_m, dw_se = mean_and_se([r["delta_wrong_char_rate_compare_minus_base"] for r in paired_rows])
        dc_m, dc_se = mean_and_se([r["delta_cer_compare_minus_base"] for r in paired_rows])
        print(f"\nPAIRED COMPARISON (compare - base, paired on image-seed, unit = IMAGE, n_images={len(paired_rows)})")
        print(f"  Delta (>=1 wrong char): {dw_m:+.4f} +/- {dw_se:.4f}")
        print(f"  Delta CER             : {dc_m:+.4f} +/- {dc_se:.4f}")
        print(f"  Wrote: {paired_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
