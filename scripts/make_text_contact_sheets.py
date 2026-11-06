#!/usr/bin/env python3
"""NOT TESTED: generate numbered contact sheets from an HR image folder and create a text-box CSV template.

Designed for CPU execution on Windows, macOS, or free Google Colab with no GPU required
and no hard-coded /content paths.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
CSV_HEADER = ["image", "box_id", "x", "y", "w", "h", "transcript", "readable"]


def discover_images(hr_dir: Path, expected_size: int | None) -> list[Path]:
    from PIL import Image

    if not hr_dir.is_dir():
        raise FileNotFoundError(f"HR image directory does not exist: {hr_dir}")
    subfolders = sorted(p.name for p in hr_dir.iterdir() if p.is_dir())
    if subfolders:
        raise ValueError(
            f"Input directory {hr_dir} contains subfolders {subfolders}. "
            "Pass the leaf ground-truth folder (for example, RealSRVal_crop128/test_HR)."
        )
    files = sorted(
        p for p in hr_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise ValueError(f"No supported images found in {hr_dir}")

    if expected_size is not None:
        target = (expected_size, expected_size)
        for path in files:
            with Image.open(path) as img:
                if img.size != target:
                    raise ValueError(
                        f"Image {path} has size {img.size}, expected {target}"
                    )
    return files


def write_csv_template(
    csv_path: Path,
    image_paths: list[Path] | None = None,
    overwrite: bool = False,
) -> bool:
    if csv_path.exists() and not overwrite:
        return False
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        if image_paths:
            for img_path in image_paths:
                writer.writerow([img_path.name, 0, "", "", "", "", "", "no"])
    return True


def make_contact_sheets(
    image_paths: list[Path],
    output_dir: Path,
    *,
    cols: int,
    rows: int,
    thumb_size: int,
) -> list[Path]:
    from PIL import Image, ImageDraw

    output_dir.mkdir(parents=True, exist_ok=True)
    per_sheet = cols * rows
    total_sheets = math.ceil(len(image_paths) / per_sheet)
    header_h = 28
    pad = 12
    cell_w = thumb_size + pad * 2
    cell_h = thumb_size + header_h + pad * 2
    sheet_w = cols * cell_w
    sheet_h = rows * cell_h

    saved_sheets: list[Path] = []
    for sheet_idx in range(total_sheets):
        start = sheet_idx * per_sheet
        end = min(len(image_paths), start + per_sheet)
        batch = image_paths[start:end]

        canvas = Image.new("RGB", (sheet_w, sheet_h), color=(245, 245, 245))
        draw = ImageDraw.Draw(canvas)

        for local_idx, img_path in enumerate(batch):
            global_idx = start + local_idx + 1
            r = local_idx // cols
            c = local_idx % cols
            x0 = c * cell_w + pad
            y0 = r * cell_h + pad

            label = f"#{global_idx:03d} {img_path.name}"
            draw.text((x0 + 2, y0 + 4), label, fill=(20, 20, 20))

            img_y0 = y0 + header_h
            with Image.open(img_path) as im:
                rgb = im.convert("RGB")
                if rgb.size != (thumb_size, thumb_size):
                    resample = getattr(Image, "Resampling", Image).BICUBIC
                    rgb = rgb.resize((thumb_size, thumb_size), resample=resample)
                canvas.paste(rgb, (x0, img_y0))

            # Draw a thin border and 25% / 50% / 75% ruler ticks to help estimate (x, y, w, h)
            draw.rectangle(
                [x0, img_y0, x0 + thumb_size - 1, img_y0 + thumb_size - 1],
                outline=(80, 80, 80),
                width=1,
            )
            for frac in (0.25, 0.5, 0.75):
                tx = x0 + int(round(thumb_size * frac))
                ty = img_y0 + int(round(thumb_size * frac))
                draw.line([(tx, img_y0), (tx, img_y0 + 6)], fill=(220, 40, 40), width=1)
                draw.line([(x0, ty), (x0 + 6, ty)], fill=(220, 40, 40), width=1)

        sheet_name = (
            f"contact_sheet_{sheet_idx + 1:02d}_"
            f"imgs_{start + 1:03d}_to_{end:03d}.png"
        )
        sheet_path = output_dir / sheet_name
        canvas.save(sheet_path)
        saved_sheets.append(sheet_path)

    return saved_sheets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hr-dir",
        type=Path,
        required=True,
        help="Path to the ground-truth HR folder (for example, .../RealSRVal_crop128/test_HR)",
    )
    parser.add_argument(
        "--output-dir",
        "--out-dir",
        dest="output_dir",
        type=Path,
        required=True,
        help="Directory where contact sheet PNGs will be saved",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("data/text_boxes_realsr_template.csv"),
        help="Path to write the blank CSV template if it does not already exist",
    )
    parser.add_argument(
        "--overwrite-csv",
        action="store_true",
        help="Overwrite --csv-out even if it already exists",
    )
    parser.add_argument(
        "--expected-size",
        type=int,
        default=512,
        help="Expected width/height of each HR image in pixels (default: 512; pass 0 to skip check)",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=5,
        help="Number of columns per contact sheet (default: 5)",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=4,
        help="Number of rows per contact sheet (default: 4 -> 20 images per sheet)",
    )
    parser.add_argument(
        "--thumb-size",
        type=int,
        default=512,
        help="Tile size in pixels per image on the contact sheet (default: 512 for 1:1 pixel coordinates)",
    )
    args = parser.parse_args()

    if args.cols <= 0 or args.rows <= 0 or args.thumb_size <= 0:
        raise ValueError("--cols, --rows, and --thumb-size must all be positive integers")

    hr_dir = args.hr_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    csv_out = args.csv_out.expanduser().resolve()
    expected_size = args.expected_size if args.expected_size > 0 else None

    image_paths = discover_images(hr_dir, expected_size)
    sheets = make_contact_sheets(
        image_paths,
        output_dir,
        cols=args.cols,
        rows=args.rows,
        thumb_size=args.thumb_size,
    )
    wrote_csv = write_csv_template(csv_out, image_paths=image_paths, overwrite=args.overwrite_csv)

    print(
        f"DONE | total_images={len(image_paths)} | sheets_written={len(sheets)} | csv={csv_out}"
    )
    print(f"Verified {len(image_paths)} HR image(s) in {hr_dir}")
    print(f"Saved {len(sheets)} contact sheet PNG(s) in {output_dir}:")
    for sheet in sheets:
        print(f"  - {sheet}")
    if wrote_csv:
        print(f"Wrote CSV template ({','.join(CSV_HEADER)}): {csv_out}")
    else:
        print(f"CSV file already exists (kept untouched; pass --overwrite-csv to replace): {csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
