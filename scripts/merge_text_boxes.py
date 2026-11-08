#!/usr/bin/env python3
"""NOT TESTED: merge data/text_boxes_realsr_m2.csv (images 1-50) and data/text_boxes_realsr_m3.csv (images 51-100) into data/text_boxes_realsr.csv.

Fails with a clear error on:
- Invalid CSV header (must be image,box_id,x,y,w,h,transcript,readable)
- Any blank 'readable' field or invalid 'readable' value (must be 'yes' or 'no')
- Invalid box_id or x,y,w,h coordinates (must be integers inside 512x512 with w>0, h>0)
- Empty transcript when readable == 'yes'
- Duplicate (image, box_id) within a file or mixing box_id=0 with box_id>=1
- Any image overlap between the _m2 and _m3 CSV files
- Missing images (requires 50 unique images in _m2 and 50 unique images in _m3, and if --hr-dir is passed, exact match with test_HR images 1-50 and 51-100)

Prints the number of images with readable text (and total readable boxes).
CPU-only, pathlib-based, no hardcoded /content paths. Works on Windows, macOS, and Free Colab.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from check_text_boxes import (
    CSV_HEADER,
    discover_hr_images,
    parse_and_validate_box_coords,
)


def load_and_validate_part(
    csv_path: Path,
    part_label: str,
    image_size: int,
) -> tuple[list[list[str]], set[str], set[str], int, list[str]]:
    """Return (normalized_rows, images_seen, images_with_yes, yes_boxes, errors)."""
    errors: list[str] = []
    rows_out: list[list[str]] = []
    images_seen: set[str] = set()
    images_with_yes: set[str] = set()
    seen_pairs: set[tuple[str, int]] = set()
    bids_by_image: dict[str, list[int]] = defaultdict(list)
    yes_boxes = 0

    if not csv_path.is_file():
        return [], set(), set(), 0, [f"{part_label}: file not found: {csv_path}"]

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = [col.strip() for col in next(reader)]
        except StopIteration:
            return [], set(), set(), 0, [f"{part_label}: file is empty: {csv_path}"]

        if header != CSV_HEADER:
            return (
                [],
                set(),
                set(),
                0,
                [f"{part_label}: invalid header {header}, expected {CSV_HEADER}"],
            )

        for line_num, raw_row in enumerate(reader, start=2):
            if not any(cell.strip() for cell in raw_row):
                continue
            if len(raw_row) != len(CSV_HEADER):
                errors.append(
                    f"{part_label} line {line_num}: expected {len(CSV_HEADER)} columns, found {len(raw_row)}"
                )
                continue

            image, box_id_s, x_s, y_s, w_s, h_s, transcript, readable = [
                c.strip() for c in raw_row
            ]
            if not image:
                errors.append(f"{part_label} line {line_num}: empty image name")
                continue

            images_seen.add(image)
            try:
                bid = int(box_id_s)
            except ValueError:
                errors.append(
                    f"{part_label} line {line_num} ({image}): box_id must be integer, got {box_id_s!r}"
                )
                bid = -1

            if bid >= 0:
                pair = (image, bid)
                if pair in seen_pairs:
                    errors.append(
                        f"{part_label} line {line_num} ({image}): duplicate (image, box_id)=({image!r}, {bid})"
                    )
                seen_pairs.add(pair)
                bids_by_image[image].append(bid)

            r_low = readable.lower()
            if r_low == "":
                errors.append(
                    f"{part_label} line {line_num} ({image}): blank 'readable' (not yet labeled)"
                )
            elif r_low == "yes":
                yes_boxes += 1
                images_with_yes.add(image)
                if bid < 1:
                    errors.append(
                        f"{part_label} line {line_num} ({image}): readable=yes requires box_id >= 1, got {bid}"
                    )
                _, coord_err = parse_and_validate_box_coords(x_s, y_s, w_s, h_s, image_size)
                if coord_err is not None:
                    errors.append(
                        f"{part_label} line {line_num} ({image}, box_id={bid}): {coord_err}"
                    )
                if not transcript:
                    errors.append(
                        f"{part_label} line {line_num} ({image}, box_id={bid}): readable=yes requires non-empty transcript"
                    )
            elif r_low == "no":
                if bid < 0:
                    errors.append(
                        f"{part_label} line {line_num} ({image}): readable=no requires box_id >= 0"
                    )
                if any(s != "" for s in (x_s, y_s, w_s, h_s)):
                    _, coord_err = parse_and_validate_box_coords(x_s, y_s, w_s, h_s, image_size)
                    if coord_err is not None:
                        errors.append(
                            f"{part_label} line {line_num} ({image}, box_id={bid}): {coord_err}"
                        )
            else:
                errors.append(
                    f"{part_label} line {line_num} ({image}): readable must be 'yes' or 'no', got {readable!r}"
                )

            rows_out.append([image, str(bid), x_s, y_s, w_s, h_s, transcript, r_low])

    for image_name, bids in sorted(bids_by_image.items()):
        if 0 in bids and len(bids) > 1:
            errors.append(
                f"{part_label} ({image_name}): cannot mix box_id=0 with box_id>=1 rows {bids}"
            )

    return rows_out, images_seen, images_with_yes, yes_boxes, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--m2",
        type=Path,
        default=Path("data/text_boxes_realsr_m2.csv"),
        help="Path to Member 2 CSV (images 1-50, default: data/text_boxes_realsr_m2.csv)",
    )
    parser.add_argument(
        "--m3",
        type=Path,
        default=Path("data/text_boxes_realsr_m3.csv"),
        help="Path to Member 3 CSV (images 51-100, default: data/text_boxes_realsr_m3.csv)",
    )
    parser.add_argument(
        "--out",
        "--output",
        dest="output_csv",
        type=Path,
        default=Path("data/text_boxes_realsr.csv"),
        help="Merged output CSV path (default: data/text_boxes_realsr.csv)",
    )
    parser.add_argument(
        "--hr-dir",
        type=Path,
        default=None,
        help="Optional path to test_HR folder to verify exact image names and 1-50 / 51-100 partition",
    )
    parser.add_argument(
        "--expected-m2-images",
        type=int,
        default=50,
        help="Expected number of unique images in --m2 (default: 50)",
    )
    parser.add_argument(
        "--expected-m3-images",
        type=int,
        default=50,
        help="Expected number of unique images in --m3 (default: 50)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=512,
        help="Expected HR image width and height in pixels (default: 512)",
    )
    args = parser.parse_args()

    m2_path = args.m2.expanduser().resolve()
    m3_path = args.m3.expanduser().resolve()
    out_path = args.output_csv.expanduser().resolve()

    rows_m2, imgs_m2, yes_imgs_m2, yes_boxes_m2, errs_m2 = load_and_validate_part(
        m2_path, "M2(1-50)", args.image_size
    )
    rows_m3, imgs_m3, yes_imgs_m3, yes_boxes_m3, errs_m3 = load_and_validate_part(
        m3_path, "M3(51-100)", args.image_size
    )

    errors = list(errs_m2) + list(errs_m3)

    overlap = sorted(imgs_m2 & imgs_m3)
    if overlap:
        errors.append(
            f"OVERLAP ERROR: {len(overlap)} image(s) appear in BOTH --m2 and --m3: {overlap[:10]}"
        )

    if len(imgs_m2) != args.expected_m2_images:
        errors.append(
            f"M2(1-50): expected {args.expected_m2_images} unique images, found {len(imgs_m2)}"
        )
    if len(imgs_m3) != args.expected_m3_images:
        errors.append(
            f"M3(51-100): expected {args.expected_m3_images} unique images, found {len(imgs_m3)}"
        )

    if args.hr_dir is not None:
        try:
            hr_files = discover_hr_images(args.hr_dir.expanduser().resolve(), args.image_size)
            hr_names = [p.name for p in hr_files]
            expected_m2_set = set(hr_names[: args.expected_m2_images])
            expected_m3_set = set(
                hr_names[args.expected_m2_images : args.expected_m2_images + args.expected_m3_images]
            )
            missing_m2 = sorted(expected_m2_set - imgs_m2)
            extra_m2 = sorted(imgs_m2 - expected_m2_set)
            missing_m3 = sorted(expected_m3_set - imgs_m3)
            extra_m3 = sorted(imgs_m3 - expected_m3_set)
            if missing_m2:
                errors.append(f"M2(1-50) missing {len(missing_m2)} image(s): {missing_m2[:10]}")
            if extra_m2:
                errors.append(f"M2(1-50) has {len(extra_m2)} unexpected image(s): {extra_m2[:10]}")
            if missing_m3:
                errors.append(f"M3(51-100) missing {len(missing_m3)} image(s): {missing_m3[:10]}")
            if extra_m3:
                errors.append(f"M3(51-100) has {len(extra_m3)} unexpected image(s): {extra_m3[:10]}")
        except Exception as exc:
            errors.append(f"HR directory validation failed: {exc}")

    if errors:
        print(
            f"MERGE FAILED | errors={len(errors)} | m2={m2_path} | m3={m3_path}",
            file=sys.stderr,
        )
        for item in errors[:50]:
            print(f"  - {item}", file=sys.stderr)
        if len(errors) > 50:
            print(f"  ... and {len(errors) - 50} more error(s)", file=sys.stderr)
        return 1

    merged_rows = sorted(rows_m2 + rows_m3, key=lambda r: (r[0], int(r[1])))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        writer.writerows(merged_rows)

    total_unique_images = len(imgs_m2 | imgs_m3)
    total_readable_images = len(yes_imgs_m2 | yes_imgs_m3)
    total_yes_boxes = yes_boxes_m2 + yes_boxes_m3

    print(
        f"MERGE PASSED | total_images={total_unique_images} "
        f"| images_with_readable_text={total_readable_images} "
        f"({_m2_m3_breakdown(len(yes_imgs_m2), len(yes_imgs_m3))}) "
        f"| readable_yes_boxes={total_yes_boxes} (m2={yes_boxes_m2}, m3={yes_boxes_m3}) "
        f"| total_rows={len(merged_rows)} | output={out_path}"
    )
    return 0


def _m2_m3_breakdown(m2_yes: int, m3_yes: int) -> str:
    return f"m2={m2_yes}, m3={m3_yes}"


if __name__ == "__main__":
    raise SystemExit(main())
