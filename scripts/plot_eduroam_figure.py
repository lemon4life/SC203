#!/usr/bin/env python3
"""NOT TESTED: build the Canon_001 (eduroam) 4-panel comparison figure with zoomed text region.

Designed for CPU execution on Windows, macOS, or free Google Colab with no GPU required
and no hard-coded /content paths.
Panels (left to right):
  1. LR (128x128 upscaled x4 to 512x512 for visual alignment)
  2. HR Ground Truth (512x512)
  3. DoSSR Default Prompt ("clean, high-resolution, 8k", 512x512)
  4. DoSSR Empty Prompt ("", 512x512)
Top row: full 512x512 image with the text crop box highlighted.
Bottom row: the exact same zoomed text crop (x, y, w, h) scaled to panel width.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_crop_arg(crop_str: str) -> tuple[int, int, int, int]:
    parts = [p.strip() for p in crop_str.split(",")]
    if len(parts) != 4:
        raise ValueError(f"--crop must be 'x,y,w,h' (4 comma-separated integers), got {crop_str!r}")
    x, y, w, h = (int(p) for p in parts)
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise ValueError(f"--crop requires x>=0, y>=0, w>0, h>0, got {(x, y, w, h)}")
    return x, y, w, h


def load_crop_from_csv(csv_path: Path, image_name: str, box_id: int) -> tuple[int, int, int, int]:
    target_stem = Path(image_name).stem
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            img_val = (row.get("image") or "").strip()
            if Path(img_val).stem == target_stem and int(row.get("box_id") or 0) == box_id:
                return (
                    int(row["x"]),
                    int(row["y"]),
                    int(row["w"]),
                    int(row["h"]),
                )
    raise ValueError(
        f"Could not find box_id={box_id} for image {image_name!r} in {csv_path}"
    )


def build_figure(
    *,
    lr_path: Path,
    hr_path: Path,
    default_path: Path,
    empty_path: Path,
    crop_box: tuple[int, int, int, int],
    output_path: Path,
    panel_size: int = 512,
) -> Path:
    from PIL import Image, ImageDraw

    x, y, w, h = crop_box
    resample_bicubic = getattr(Image, "Resampling", Image).BICUBIC
    resample_nearest = getattr(Image, "Resampling", Image).NEAREST

    panels = [
        ("LR Input (128x128)", lr_path),
        ("HR Reference (512x512)", hr_path),
        ("DoSSR Default Prompt", default_path),
        ("DoSSR Empty Prompt", empty_path),
    ]

    loaded_full: list[tuple[str, Image.Image]] = []
    for title, path in panels:
        if not path.is_file():
            raise FileNotFoundError(f"Missing image for panel '{title}': {path}")
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            if rgb.size != (panel_size, panel_size):
                # Upsample 128x128 LR to 512x512 so the same (x, y, w, h) box aligns across all 4 panels
                rgb = rgb.resize((panel_size, panel_size), resample=resample_bicubic)
            loaded_full.append((title, rgb))

    if x + w > panel_size or y + h > panel_size:
        raise ValueError(
            f"Crop box {(x, y, w, h)} exceeds {panel_size}x{panel_size} image bounds"
        )

    zoom_w = panel_size
    zoom_h = max(120, int(round(panel_size * (h / float(w)))))
    header_h = 34
    sub_h = 26
    pad = 14

    total_w = len(loaded_full) * panel_size + (len(loaded_full) + 1) * pad
    total_h = header_h + panel_size + sub_h + zoom_h + pad * 3

    canvas = Image.new("RGB", (total_w, total_h), color=(250, 250, 250))
    draw = ImageDraw.Draw(canvas)

    for idx, (title, full_im) in enumerate(loaded_full):
        x0 = pad + idx * (panel_size + pad)
        y0 = header_h

        draw.text((x0 + 4, 8), title, fill=(20, 20, 20))

        full_annotated = full_im.copy()
        full_draw = ImageDraw.Draw(full_annotated)
        for offset in range(3):
            full_draw.rectangle(
                [x - offset, y - offset, x + w + offset, y + h + offset],
                outline=(230, 30, 30),
                width=1,
            )
        canvas.paste(full_annotated, (x0, y0))

        crop_im = full_im.crop((x, y, x + w, y + h))
        zoomed_crop = crop_im.resize((zoom_w, zoom_h), resample=resample_nearest)

        zoom_y0 = y0 + panel_size + sub_h
        draw.text(
            (x0 + 4, y0 + panel_size + 5),
            f"Zoomed crop (x={x}, y={y}, w={w}, h={h})",
            fill=(180, 20, 20),
        )
        canvas.paste(zoomed_crop, (x0, zoom_y0))
        draw.rectangle(
            [x0, zoom_y0, x0 + zoom_w - 1, zoom_y0 + zoom_h - 1],
            outline=(230, 30, 30),
            width=2,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lr-image", type=Path, required=True, help="Path to Canon_001_LR4.png in test_LR (128x128)")
    parser.add_argument("--hr-image", type=Path, required=True, help="Path to Canon_001_LR4.png in test_HR (512x512)")
    parser.add_argument(
        "--default-image",
        type=Path,
        required=True,
        help="Path to DoSSR default-prompt output Canon_001_LR4_0.png (512x512)",
    )
    parser.add_argument(
        "--empty-image",
        type=Path,
        required=True,
        help="Path to DoSSR empty-prompt output Canon_001_LR4_0.png (512x512)",
    )
    parser.add_argument(
        "--crop",
        type=str,
        default=None,
        help="Crop box 'x,y,w,h' in 512x512 HR coordinates (for example, '135,10,121,55')",
    )
    parser.add_argument(
        "--boxes-csv",
        type=Path,
        default=None,
        help="Optional CSV file to read (x,y,w,h) from if --crop is not passed",
    )
    parser.add_argument(
        "--image-name",
        type=str,
        default="Canon_001_LR4.png",
        help="Image name in --boxes-csv (default: Canon_001_LR4.png)",
    )
    parser.add_argument(
        "--box-id",
        type=int,
        default=1,
        help="box_id in --boxes-csv (default: 1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/figures/canon_001_eduroam_default_vs_empty.png"),
        help="Output figure PNG path",
    )
    args = parser.parse_args()

    if args.crop is not None:
        crop_box = parse_crop_arg(args.crop)
    elif args.boxes_csv is not None:
        crop_box = load_crop_from_csv(
            args.boxes_csv.expanduser().resolve(), args.image_name, args.box_id
        )
    else:
        parser.error("Provide either --crop x,y,w,h or --boxes-csv PATH")

    saved = build_figure(
        lr_path=args.lr_image.expanduser().resolve(),
        hr_path=args.hr_image.expanduser().resolve(),
        default_path=args.default_image.expanduser().resolve(),
        empty_path=args.empty_image.expanduser().resolve(),
        crop_box=crop_box,
        output_path=args.output.expanduser().resolve(),
    )
    print(f"EDUROAM FIGURE SAVED | crop={crop_box} | path={saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
