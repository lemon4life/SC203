#!/usr/bin/env python3
"""NOT TESTED: generate numbered contact sheets from an HR image folder, create a text-box CSV template, and check CSV labeling status.

Designed for CPU execution on Windows, macOS, or free Google Colab with no GPU required
and no hard-coded /content paths.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
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
    """Write the CSV header and one blank-readable row per image (blank = not yet labeled)."""
    if csv_path.exists() and not overwrite:
        return False
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        if image_paths:
            for img_path in image_paths:
                # readable is left blank ("") by default to indicate 'not yet labeled'
                writer.writerow([img_path.name, 0, "", "", "", "", "", ""])
    return True


def check_csv_labels(csv_path: Path) -> int:
    """Verify that no rows have blank readable values and that readable=yes rows are complete."""
    if not csv_path.is_file():
        print(f"CSV CHECK FAILED | file not found: {csv_path}", file=sys.stderr)
        return 1

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = [col.strip() for col in next(reader)]
        except StopIteration:
            print(f"CSV CHECK FAILED | file is empty: {csv_path}", file=sys.stderr)
            return 1

        if header != CSV_HEADER:
            print(
                f"CSV CHECK FAILED | invalid header {header}, expected {CSV_HEADER}",
                file=sys.stderr,
            )
            return 1

        blank_images: list[str] = []
        invalid_rows: list[str] = []
        total_rows = 0
        readable_yes = 0
        readable_no = 0
        images_seen: set[str] = set()
        images_with_yes: set[str] = set()

        for line_num, row in enumerate(reader, start=2):
            if not any(cell.strip() for cell in row):
                continue
            total_rows += 1
            if len(row) != len(CSV_HEADER):
                invalid_rows.append(
                    f"line {line_num}: expected {len(CSV_HEADER)} columns, found {len(row)}"
                )
                continue
            image, box_id, x, y, w, h, transcript, readable = [c.strip() for c in row]
            images_seen.add(image)
            r_low = readable.lower()
            if r_low == "":
                blank_images.append(f"{image} (line {line_num})")
            elif r_low == "yes":
                readable_yes += 1
                images_with_yes.add(image)
                try:
                    bid = int(box_id)
                    xi, yi, wi, hi = int(x), int(y), int(w), int(h)
                    if bid < 1 or wi <= 0 or hi <= 0 or xi < 0 or yi < 0:
                        invalid_rows.append(
                            f"line {line_num} ({image}): readable=yes requires box_id>=1, x>=0, y>=0, w>0, h>0"
                        )
                except ValueError:
                    invalid_rows.append(
                        f"line {line_num} ({image}): readable=yes requires integer box_id,x,y,w,h"
                    )
                if not transcript:
                    invalid_rows.append(
                        f"line {line_num} ({image}): readable=yes requires non-empty transcript"
                    )
            elif r_low == "no":
                readable_no += 1
            else:
                invalid_rows.append(
                    f"line {line_num} ({image}): readable must be 'yes' or 'no', got '{readable}'"
                )

    if total_rows == 0:
        print(
            f"CSV CHECK FAILED | {csv_path} has header only (0 data rows)",
            file=sys.stderr,
        )
        return 1

    if blank_images or invalid_rows:
        print(
            f"CSV CHECK FAILED | total_rows={total_rows} | unlabeled_blank={len(blank_images)} "
            f"| invalid_rows={len(invalid_rows)} | file={csv_path}",
            file=sys.stderr,
        )
        if blank_images:
            print("Images with blank 'readable' (not yet labeled):", file=sys.stderr)
            for item in blank_images:
                print(f"  - {item}", file=sys.stderr)
        if invalid_rows:
            print("Invalid rows:", file=sys.stderr)
            for item in invalid_rows:
                print(f"  - {item}", file=sys.stderr)
        return 1

    print(
        f"CSV CHECK PASSED | total_rows={total_rows} | unique_images={len(images_seen)} "
        f"| images_with_readable_text={len(images_with_yes)} | readable_yes_boxes={readable_yes} "
        f"| readable_no_rows={readable_no} | unlabeled_blank=0 | file={csv_path}"
    )
    return 0


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
        "--check",
        nargs="?",
        const="",
        default=None,
        metavar="CSV_PATH",
        help=(
            "Check a CSV file for blank 'readable' rows (not yet labeled) or invalid fields, "
            "and fail (exit 1) if any exist. Uses CSV_PATH if given, else --csv-out."
        ),
    )
    parser.add_argument(
        "--hr-dir",
        type=Path,
        default=None,
        help="Path to the ground-truth HR folder (for example, .../RealSRVal_crop128/test_HR)",
    )
    parser.add_argument(
        "--output-dir",
        "--out-dir",
        dest="output_dir",
        type=Path,
        default=None,
        help="Directory where contact sheet PNGs will be saved",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("data/text_boxes_realsr.csv"),
        help="Path to write or check the CSV file (default: data/text_boxes_realsr.csv)",
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

    if args.check is not None:
        target_csv = Path(args.check) if args.check else args.csv_out
        return check_csv_labels(target_csv.expanduser().resolve())

    if args.hr_dir is None or args.output_dir is None:
        parser.error("--hr-dir and --output-dir/--out-dir are required unless --check is passed")

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
        print(
            f"Wrote CSV ({','.join(CSV_HEADER)}) with 'readable' left blank (not yet labeled): {csv_out}"
        )
    else:
        print(f"CSV file already exists (kept untouched; pass --overwrite-csv to replace): {csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
