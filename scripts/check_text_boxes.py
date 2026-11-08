#!/usr/bin/env python3
"""NOT TESTED: validate a labeler's text-box CSV against test_HR and optionally save box preview images.

Checks enforced:
1. Header must be exactly: image,box_id,x,y,w,h,transcript,readable
2. readable must be 'yes' or 'no' (no blanks).
3. When readable == 'yes':
   - box_id must be an integer >= 1.
   - x, y, w, h must be integers inside the 512x512 HR image (0 <= x < 512, 0 <= y < 512,
     w > 0, h > 0, x + w <= 512, y + h <= 512).
   - transcript must be non-empty.
4. When readable == 'no':
   - box_id must be an integer >= 0 (0 for an image with no readable text, or >= 1 for an
     unreadable text box). If x, y, w, h are provided, they must be valid integers inside 512x512.
5. No duplicate (image, box_id) pairs, and no image may mix box_id=0 (no readable text) with
   box_id >= 1 rows.
6. Every image name in the CSV must exist in --hr-dir (test_HR), and every image in the
   labeler's assigned 1-based range (--range START-END, e.g. 1-50 or 51-100) must have at
   least one row in the CSV (with no images outside that range).
7. Optional --preview-dir draws each box on its 512x512 HR image and saves a PNG preview.

CPU-only, pathlib-based, no hardcoded /content paths. Works on Windows, macOS, and Free Colab.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
CSV_HEADER = ["image", "box_id", "x", "y", "w", "h", "transcript", "readable"]


def discover_hr_images(hr_dir: Path, image_size: int) -> list[Path]:
    from PIL import Image

    if not hr_dir.is_dir():
        raise FileNotFoundError(f"HR directory does not exist: {hr_dir}")
    subfolders = sorted(p.name for p in hr_dir.iterdir() if p.is_dir())
    if subfolders:
        raise ValueError(
            f"HR directory {hr_dir} contains subfolders {subfolders}. "
            "Pass the leaf test_HR folder (for example, RealSRVal_crop128/test_HR)."
        )
    files = sorted(
        p for p in hr_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise ValueError(f"No image files found in {hr_dir}")
    if image_size > 0:
        expected = (image_size, image_size)
        for path in files:
            with Image.open(path) as im:
                if im.size != expected:
                    raise ValueError(
                        f"HR image {path} has size {im.size}, expected {expected}"
                    )
    return files


def resolve_range(range_arg: str | None, csv_path: Path, total_hr: int) -> tuple[int, int]:
    """Return 1-based inclusive (start_idx, end_idx)."""
    if range_arg is None:
        name_low = csv_path.name.lower()
        if name_low.endswith("_m2.csv"):
            return (1, min(50, total_hr))
        if name_low.endswith("_m3.csv") and total_hr >= 51:
            return (51, total_hr)
        return (1, total_hr)
    cleaned = range_arg.strip().replace(":", "-")
    parts = cleaned.split("-")
    if len(parts) != 2 or not parts[0].strip().isdigit() or not parts[1].strip().isdigit():
        raise ValueError(
            f"Invalid --range '{range_arg}'. Expected START-END (for example 1-50, 51-100, or 1-100)."
        )
    start_idx, end_idx = int(parts[0].strip()), int(parts[1].strip())
    if start_idx < 1 or end_idx < start_idx or end_idx > total_hr:
        raise ValueError(
            f"--range {start_idx}-{end_idx} is out of bounds for {total_hr} HR images (valid: 1-{total_hr})."
        )
    return (start_idx, end_idx)


def parse_and_validate_box_coords(
    x_str: str,
    y_str: str,
    w_str: str,
    h_str: str,
    image_size: int,
) -> tuple[tuple[int, int, int, int] | None, str | None]:
    try:
        xi, yi, wi, hi = int(x_str), int(y_str), int(w_str), int(h_str)
    except ValueError:
        return None, f"x,y,w,h must be integers, got ({x_str!r}, {y_str!r}, {w_str!r}, {h_str!r})"
    if wi <= 0 or hi <= 0:
        return None, f"w and h must be > 0, got w={wi}, h={hi}"
    if xi < 0 or yi < 0 or xi >= image_size or yi >= image_size:
        return None, f"x and y must be inside 0..{image_size - 1}, got x={xi}, y={yi}"
    if xi + wi > image_size or yi + hi > image_size:
        return (
            None,
            f"box (x={xi}, y={yi}, w={wi}, h={hi}) exceeds {image_size}x{image_size} "
            f"(x+w={xi + wi}, y+h={yi + hi})",
        )
    return (xi, yi, wi, hi), None


def render_previews(
    hr_by_name: dict[str, Path],
    boxes_by_image: dict[str, list[dict[str, object]]],
    preview_dir: Path,
    only_readable_images: bool = True,
) -> int:
    from PIL import Image, ImageDraw

    preview_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    for image_name, rows in sorted(boxes_by_image.items()):
        draw_rows = [r for r in rows if r.get("coords") is not None]
        if only_readable_images and not draw_rows:
            continue
        hr_path = hr_by_name.get(image_name)
        if hr_path is None:
            continue
        with Image.open(hr_path) as im:
            canvas = im.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        for row in draw_rows:
            xi, yi, wi, hi = row["coords"]  # type: ignore[misc]
            r_low = str(row["readable"])
            bid = row["box_id"]
            transcript = str(row["transcript"])
            color = (0, 220, 60) if r_low == "yes" else (240, 160, 0)
            draw.rectangle([xi, yi, xi + wi - 1, yi + hi - 1], outline=color, width=2)
            label = f"#{bid}:{transcript}" if transcript else f"#{bid}:({r_low})"
            text_y = max(0, yi - 14)
            draw.rectangle(
                [xi, text_y, min(canvas.width - 1, xi + max(48, len(label) * 6 + 6)), text_y + 13],
                fill=(20, 20, 20),
            )
            draw.text((xi + 3, text_y + 1), label, fill=color)
        out_path = preview_dir / f"{Path(image_name).stem}_boxes.png"
        canvas.save(out_path)
        saved_count += 1
    return saved_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Path to the labeler CSV file (for example, data/text_boxes_realsr_m2.csv)",
    )
    parser.add_argument(
        "--hr-dir",
        type=Path,
        required=True,
        help="Path to the 512x512 test_HR folder (for example, .../RealSRVal_crop128/test_HR)",
    )
    parser.add_argument(
        "--range",
        dest="image_range",
        type=str,
        default=None,
        help=(
            "1-based inclusive image range expected in this CSV (for example 1-50, 51-100, or 1-100). "
            "Auto-detected for *_m2.csv (1-50) and *_m3.csv (51-100) if omitted."
        ),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Expected HR image width and height in pixels (default: 512)",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=None,
        help="Optional directory to save HR images with drawn bounding boxes (<stem>_boxes.png)",
    )
    parser.add_argument(
        "--preview-all",
        action="store_true",
        help="When --preview-dir is set, also save previews for images with 0 boxes",
    )
    args = parser.parse_args()

    csv_path = args.csv.expanduser().resolve()
    hr_dir = args.hr_dir.expanduser().resolve()

    if not csv_path.is_file():
        print(f"CHECK FAILED | CSV file not found: {csv_path}", file=sys.stderr)
        return 1

    try:
        hr_files = discover_hr_images(hr_dir, args.image_size)
        start_idx, end_idx = resolve_range(args.image_range, csv_path, len(hr_files))
    except Exception as exc:
        print(f"CHECK FAILED | {exc}", file=sys.stderr)
        return 1

    hr_by_name = {p.name: p for p in hr_files}
    expected_range_names = [p.name for p in hr_files[start_idx - 1 : end_idx]]
    expected_range_set = set(expected_range_names)

    errors: list[str] = []
    seen_pairs: set[tuple[str, int]] = set()
    box_ids_by_image: dict[str, list[int]] = defaultdict(list)
    rows_by_image: dict[str, list[dict[str, object]]] = defaultdict(list)
    images_with_yes: set[str] = set()
    total_rows = 0
    readable_yes_boxes = 0
    readable_no_rows = 0
    blank_rows = 0

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = [col.strip() for col in next(reader)]
        except StopIteration:
            print(f"CHECK FAILED | CSV file is empty: {csv_path}", file=sys.stderr)
            return 1

        if header != CSV_HEADER:
            print(
                f"CHECK FAILED | invalid header {header}, expected {CSV_HEADER}",
                file=sys.stderr,
            )
            return 1

        for line_num, raw_row in enumerate(reader, start=2):
            if not any(cell.strip() for cell in raw_row):
                continue
            total_rows += 1
            if len(raw_row) != len(CSV_HEADER):
                errors.append(
                    f"line {line_num}: expected {len(CSV_HEADER)} columns, found {len(raw_row)}"
                )
                continue

            image, box_id_s, x_s, y_s, w_s, h_s, transcript, readable = [
                cell.strip() for cell in raw_row
            ]

            if not image:
                errors.append(f"line {line_num}: empty image column")
                continue
            if image not in hr_by_name:
                errors.append(f"line {line_num} ({image}): image does not exist in test_HR ({hr_dir})")
            elif image not in expected_range_set:
                errors.append(
                    f"line {line_num} ({image}): image is outside expected range {start_idx}-{end_idx}"
                )

            try:
                bid = int(box_id_s)
            except ValueError:
                errors.append(f"line {line_num} ({image}): box_id must be an integer, got {box_id_s!r}")
                bid = -1

            if bid >= 0:
                pair = (image, bid)
                if pair in seen_pairs:
                    errors.append(f"line {line_num} ({image}): duplicate (image, box_id)=({image!r}, {bid})")
                seen_pairs.add(pair)
                box_ids_by_image[image].append(bid)

            r_low = readable.lower()
            coords: tuple[int, int, int, int] | None = None
            if r_low == "":
                blank_rows += 1
                errors.append(f"line {line_num} ({image}): readable is blank (must be 'yes' or 'no')")
            elif r_low == "yes":
                readable_yes_boxes += 1
                images_with_yes.add(image)
                if bid < 1:
                    errors.append(f"line {line_num} ({image}): readable=yes requires box_id >= 1, got {bid}")
                coords, coord_err = parse_and_validate_box_coords(
                    x_s, y_s, w_s, h_s, args.image_size
                )
                if coord_err is not None:
                    errors.append(f"line {line_num} ({image}, box_id={bid}): {coord_err}")
                if not transcript:
                    errors.append(
                        f"line {line_num} ({image}, box_id={bid}): readable=yes requires non-empty transcript"
                    )
            elif r_low == "no":
                readable_no_rows += 1
                if bid < 0:
                    errors.append(f"line {line_num} ({image}): readable=no requires box_id >= 0")
                if any(s != "" for s in (x_s, y_s, w_s, h_s)):
                    coords, coord_err = parse_and_validate_box_coords(
                        x_s, y_s, w_s, h_s, args.image_size
                    )
                    if coord_err is not None:
                        errors.append(f"line {line_num} ({image}, box_id={bid}): {coord_err}")
            else:
                errors.append(
                    f"line {line_num} ({image}): readable must be 'yes' or 'no', got {readable!r}"
                )

            rows_by_image[image].append(
                {
                    "box_id": bid,
                    "coords": coords,
                    "transcript": transcript,
                    "readable": r_low,
                }
            )

    for image_name, bids in sorted(box_ids_by_image.items()):
        if 0 in bids and len(bids) > 1:
            errors.append(
                f"{image_name}: cannot mix box_id=0 (no readable text) with other box_id rows {bids}"
            )

    missing_in_range = [name for name in expected_range_names if name not in rows_by_image]
    for name in missing_in_range:
        errors.append(f"missing required image in range {start_idx}-{end_idx}: {name}")

    if errors:
        print(
            f"CHECK FAILED | file={csv_path} | range={start_idx}-{end_idx} "
            f"| total_rows={total_rows} | blank_readable={blank_rows} | errors={len(errors)}",
            file=sys.stderr,
        )
        for item in errors[:50]:
            print(f"  - {item}", file=sys.stderr)
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more error(s)", file=sys.stderr)
        return 1

    previews_saved = 0
    if args.preview_dir is not None:
        preview_dir = args.preview_dir.expanduser().resolve()
        previews_saved = render_previews(
            hr_by_name,
            rows_by_image,
            preview_dir,
            only_readable_images=not args.preview_all,
        )

    print(
        f"CHECK PASSED | file={csv_path} | range={start_idx}-{end_idx} ({len(expected_range_names)} images) "
        f"| total_rows={total_rows} | images_with_readable_text={len(images_with_yes)} "
        f"| readable_yes_boxes={readable_yes_boxes} | readable_no_rows={readable_no_rows} "
        f"| previews_saved={previews_saved}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
