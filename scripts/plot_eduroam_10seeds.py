#!/usr/bin/env python3
"""Build a publication-quality 10-seed comparison figure for the Canon_001 (eduroam) text region.

Panels displayed:
- Reference row:
  1. LR Input (128x128 upscaled x4)
  2. HR Ground Truth ("eduroam")
  3. Bicubic x4 Baseline ("eurotin")
- 10-Seed DoSSR outputs:
  Seeds 42, 43, 44, 45, 46, 47, 48, 49, 50, 51 (or custom seed list)
  Each panel displays the zoomed crop and the exact OCR transcript recognized.

Usage:
    # Run on Colab / environment with access to Drive outputs:
    python scripts/plot_eduroam_10seeds.py \
        --outputs-root /content/drive/MyDrive/lemmepass/outputs/seed-study-realsr-100img-10seed-default-prompt-sdpa-off-fp16 \
        --hr-image /content/drive/MyDrive/lemmepass/data/StableSR_testsets/StableSR_testsets/RealSRVal_crop128/test_HR/Canon_001_LR4.png \
        --lr-image /content/drive/MyDrive/lemmepass/data/StableSR_testsets/StableSR_testsets/RealSRVal_crop128/test_LR/Canon_001_LR4.png \
        --output outputs/figures/canon_001_eduroam_10seeds_comparison.png
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Ground-truth crop box for eduroam sign on Canon_001_LR4 (512x512 coordinates)
DEFAULT_EDUROAM_CROP = (310, 5, 202, 85)   # Full sign box (x, y, w, h)
WORD_BOX = (320, 38, 192, 46)              # Word bounding box (x, y, w, h)

# Pre-extracted OCR transcripts from results/text_eval_per_box_seed.csv
EDUROAM_OCR_RESULTS = {
    42: "odureani",
    43: "aduronm",
    44: "odareum",
    45: "oduroant",
    46: "edureain",
    47: "cduroniv",
    48: "adaroam",
    49: "Odoroam",
    50: "Ddaroain",
    51: "aduronm",
}


def build_10seed_figure(
    outputs_root: Path,
    hr_path: Optional[Path],
    lr_path: Optional[Path],
    output_png: Path,
    crop_box: Tuple[int, int, int, int] = DEFAULT_EDUROAM_CROP,
    seeds: Optional[List[int]] = None,
) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    if seeds is None:
        seeds = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]

    x, y, w, h = crop_box
    zoom_w = 260
    zoom_h = int(round(zoom_w * (h / float(w))))
    card_w = zoom_w + 16
    card_h = zoom_h + 60
    pad = 12

    # Grid layout: 4 columns x 3 rows (or 12 items: 2 references + 10 seeds)
    items = []
    if hr_path and hr_path.is_file():
        items.append(("HR Ground Truth", hr_path, "eduroam (GT)", (30, 140, 30)))
    if lr_path and lr_path.is_file():
        items.append(("LR Input (x4)", lr_path, "blurry / unreadable", (120, 120, 120)))

    for s in seeds:
        # Search for Canon_001_LR4_0.png in seed_<seed> directory
        cand1 = outputs_root / f"seed_{s}" / "Canon_001_LR4_0.png"
        cand2 = outputs_root / f"seed_{s}" / "Canon_001_LR4.png"
        p = cand1 if cand1.is_file() else cand2
        ocr_text = EDUROAM_OCR_RESULTS.get(s, "N/A")
        items.append((f"DoSSR Seed {s}", p, f"OCR: \"{ocr_text}\"", (200, 30, 30)))

    cols = 4
    rows = (len(items) + cols - 1) // cols
    canvas_w = pad + cols * (card_w + pad)
    canvas_h = 50 + pad + rows * (card_h + pad)

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(245, 247, 250))
    draw = ImageDraw.Draw(canvas)

    # Header Title
    draw.text((pad + 6, 14), "Canon_001 Text Region ('eduroam') — Stochastic Degradation Across 10 Seeds", fill=(20, 30, 45))

    for idx, (title, img_path, subtitle, color) in enumerate(items):
        r = idx // cols
        c = idx % cols
        x0 = pad + c * (card_w + pad)
        y0 = 50 + pad + r * (card_h + pad)

        # Card background
        draw.rectangle([x0, y0, x0 + card_w, y0 + card_h], fill=(255, 255, 255), outline=(215, 220, 228), width=1)

        # Title
        draw.text((x0 + 8, y0 + 6), title, fill=(20, 20, 20))

        # Crop & zoom
        img_x = x0 + 8
        img_y = y0 + 26
        if img_path and img_path.is_file():
            with Image.open(img_path) as im:
                rgb = im.convert("RGB")
                if rgb.size != (512, 512):
                    rgb = rgb.resize((512, 512), resample=getattr(Image, "Resampling", Image).BICUBIC)
                crop_im = rgb.crop((x, y, x + w, y + h))
                zoomed = crop_im.resize((zoom_w, zoom_h), resample=getattr(Image, "Resampling", Image).NEAREST)
                canvas.paste(zoomed, (img_x, img_y))
        else:
            # Placeholder if file missing
            draw.rectangle([img_x, img_y, img_x + zoom_w, img_y + zoom_h], fill=(235, 238, 242))
            draw.text((img_x + 20, img_y + zoom_h // 2 - 8), f"File not found:\n{img_path.name if img_path else 'N/A'}", fill=(120, 120, 120))

        # Crop border
        draw.rectangle([img_x, img_y, img_x + zoom_w, img_y + zoom_h], outline=(200, 205, 215), width=1)

        # OCR Subtitle
        draw.text((x0 + 8, y0 + 26 + zoom_h + 6), subtitle, fill=color)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_png)
    return output_png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("/content/drive/MyDrive/lemmepass/outputs/seed-study-realsr-100img-10seed-default-prompt-sdpa-off-fp16"),
        help="Root folder containing seed_42, seed_43, ... folders",
    )
    parser.add_argument(
        "--hr-image",
        type=Path,
        default=Path("/content/drive/MyDrive/lemmepass/data/StableSR_testsets/StableSR_testsets/RealSRVal_crop128/test_HR/Canon_001_LR4.png"),
        help="Path to HR Canon_001 image",
    )
    parser.add_argument(
        "--lr-image",
        type=Path,
        default=Path("/content/drive/MyDrive/lemmepass/data/StableSR_testsets/StableSR_testsets/RealSRVal_crop128/test_LR/Canon_001_LR4.png"),
        help="Path to LR Canon_001 image",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/figures/canon_001_eduroam_10seeds_comparison.png"),
        help="Output PNG path",
    )
    args = parser.parse_args()

    saved = build_10seed_figure(
        outputs_root=args.outputs_root,
        hr_path=args.hr_image,
        lr_path=args.lr_image,
        output_png=args.output,
    )
    print(f"Successfully generated 10-seed comparison figure: {saved}")


if __name__ == "__main__":
    main()
