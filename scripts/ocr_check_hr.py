#!/usr/bin/env python3
"""NOT TESTED: check OCR accuracy on sharp test_HR crops (1x, 2x, or 4x upscale) against ground-truth transcripts.

Reads a labeled text-box CSV (such as data/text_boxes_realsr.csv or data/text_boxes_realsr_m3.csv),
crops each readable=yes box from the 512x512 test_HR image, optionally bicubic-enlarges the crop
by 1x, 2x, or 4x (--upscale), runs the chosen OCR backend (--ocr-backend rapidocr|easyocr|tesseract),
compares against the ground-truth transcript using the exact normalization rules in scripts/text_eval.py,
prints the exact-match box count out of total boxes (plus CER), and writes a per-box CSV.

CPU-only by default, pathlib-based, no hardcoded /content paths. Works on Windows, macOS, and Free Colab.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from text_eval import (
    DEFAULT_OCR_BACKEND,
    VALID_OCR_BACKENDS,
    VALID_UPSCALE_FACTORS,
    evaluate_single_folder_reference,
    load_readable_boxes,
    mean_and_se,
    write_csv_dicts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--boxes-csv",
        "--csv",
        dest="boxes_csv",
        type=Path,
        default=Path("data/text_boxes_realsr.csv"),
        help="Path to the labeled text-box CSV (for example, data/text_boxes_realsr_m3.csv or data/text_boxes_realsr.csv)",
    )
    parser.add_argument(
        "--hr-dir",
        type=Path,
        required=True,
        help="Path to the 512x512 test_HR folder (for example, .../RealSRVal_crop128/test_HR)",
    )
    parser.add_argument(
        "--ocr-backend",
        choices=VALID_OCR_BACKENDS,
        default=DEFAULT_OCR_BACKEND,
        help=f"OCR backend to test (default: {DEFAULT_OCR_BACKEND})",
    )
    parser.add_argument(
        "--upscale",
        type=int,
        choices=VALID_UPSCALE_FACTORS,
        default=1,
        help="Bicubic upscale factor for each cropped text box before OCR (1, 2, or 4; default: 1)",
    )
    parser.add_argument(
        "--output-csv",
        "--out",
        dest="output_csv",
        type=Path,
        default=None,
        help=(
            "Path to save the per-box OCR check CSV "
            "(default: outputs/ocr_check_hr_<backend>_x<upscale>.csv)"
        ),
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
        help="Expected HR image width and height in pixels (default: 512)",
    )
    args = parser.parse_args()

    boxes_csv = args.boxes_csv.expanduser().resolve()
    hr_dir = args.hr_dir.expanduser().resolve()
    out_csv = (
        args.output_csv
        if args.output_csv is not None
        else Path(f"outputs/ocr_check_hr_{args.ocr_backend}_x{args.upscale}.csv")
    ).expanduser().resolve()

    boxes = load_readable_boxes(boxes_csv)
    per_box_rows, image_rows = evaluate_single_folder_reference(
        hr_dir,
        boxes,
        ocr_backend=args.ocr_backend,
        device=args.device,
        expected_hr_size=args.expected_size,
        upscale_crop=args.upscale,
        pre_upscale_image_to=None,
        reference_label="HR_CHECK",
    )

    write_csv_dicts(out_csv, per_box_rows)

    total_boxes = len(per_box_rows)
    exact_ci = sum(int(r["exact_match"]) for r in per_box_rows)
    exact_cs = sum(1 - int(r["wrong_char_case_sensitive"]) for r in per_box_rows)
    box_cer_mean = sum(float(r["cer"]) for r in per_box_rows) / max(1, total_boxes)
    img_wrong_m, img_wrong_se = mean_and_se([float(r["wrong_char_rate"]) for r in image_rows])
    img_cer_m, img_cer_se = mean_and_se([float(r["cer_mean"]) for r in image_rows])

    print(
        f"OCR HR CHECK DONE | backend={args.ocr_backend} | upscale={args.upscale}x "
        f"| exact_right={exact_ci}/{total_boxes} ({100.0 * exact_ci / max(1, total_boxes):.1f}%) "
        f"| exact_case_sensitive={exact_cs}/{total_boxes} ({100.0 * exact_cs / max(1, total_boxes):.1f}%) "
        f"| box_mean_cer={box_cer_mean:.4f} | images={len(image_rows)} "
        f"| img_wrong_char_rate={img_wrong_m:.4f}+/-{img_wrong_se:.4f} "
        f"| img_cer={img_cer_m:.4f}+/-{img_cer_se:.4f} | csv={out_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
