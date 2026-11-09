# Results

Curated summary CSVs, whole-image metrics tables, and 10-seed text evaluation tables belong here in Git. Large raw image outputs (`2,000` `512x512` 10-seed PNGs, contact sheets, and checkpoints) stay on shared Google Drive (`/content/drive/MyDrive/lemmepass/outputs/`).

## Tracked Summary CSVs in `results/`

### 1. Combined 100-Image (`25` Readable Images, `193` Boxes, `1,930` Box-Seed Crops) Text Evaluation (`VERIFIED`)
- `text_eval_summary_by_image.csv` — Per-image 10-seed `default` prompt text restoration accuracy & cross-seed disagreement (`25` readable images).
- `text_eval_paired_comparison_by_image.csv` — Paired `(image, seed)` comparison (`empty` minus `default` prompt, `25` readable images, `1,930` box-seed pairs).
- `text_eval_hr_ceiling_by_image.csv` & `text_eval_hr_ceiling_per_box.csv` — OCR upper bound on sharp `512x512` `test_HR` ground-truth crops (`136 / 193 = 70.5%` exact match, `CER = 0.1648`).
- `text_eval_bicubic_floor_by_image.csv` & `text_eval_bicubic_floor_per_box.csv` — OCR lower bound on `128x128 -> 512x512` bicubic $\times 4$ `test_LR` crops (`39 / 193 = 20.2%` exact match, `CER = 0.7423`).
- `text_eval_per_box_seed.csv` — All `1,930` `(box, seed)` OCR readings and character error rates across `seed_42`..`seed_51`.

### 2. Per-Camera Subset Text Evaluation Tables (`VERIFIED`)
- **Canon (`images 1–50`, `10` readable images, `50` boxes — Member 2, Gia Hân):** `results/text_eval_10seed_canon/*.csv` and `results/text_eval_canon_*.csv`.
- **Nikon (`images 51–100`, `15` readable images, `143` boxes — Member 3, Quang Minh):** `results/text_eval_nikon_*.csv`.

### 3. Whole-Image Metrics (`scripts/compute_metrics.py`)
- Save 10-seed whole-image `PSNR`, `SSIM`, `LPIPS`, and pairwise seed spread summary CSVs here as:
  - `results/metrics_10seed_default_summary_by_image.csv` & `results/metrics_10seed_default_per_seed.csv`
  - `results/metrics_10seed_empty_summary_by_image.csv` & `results/metrics_10seed_empty_per_seed.csv`
