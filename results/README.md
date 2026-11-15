# Results

Curated summary CSVs, whole-image metrics tables, and 10-seed text evaluation tables belong here in Git. Large raw image outputs (`2,000` `512x512` 10-seed PNGs, contact sheets, and checkpoints) stay on shared Google Drive (`/content/drive/MyDrive/lemmepass/outputs/`).

## Tracked Summary CSVs in `results/`

### 1. Combined 100-Image (`25` Readable Images, `193` Boxes, `1,930` Box-Seed Crops) Text Evaluation (`VERIFIED`)
- `text_eval_summary_by_image.csv` — Per-image 10-seed `default` prompt text restoration accuracy & cross-seed disagreement (`25` readable images).
- `text_eval_paired_comparison_by_image.csv` — Paired `(image, seed)` comparison (`empty` minus `default` prompt, `25` readable images, `1,930` box-seed pairs).
- `text_eval_hr_ceiling_by_image.csv` & `text_eval_hr_ceiling_per_box.csv` — OCR upper bound on sharp `512x512` `test_HR` ground-truth crops (`136 / 193 = 70.5%` exact match, `CER = 0.1648`).
- `text_eval_bicubic_floor_by_image.csv` & `text_eval_bicubic_floor_per_box.csv` — OCR lower bound on `128x128 -> 512x512` bicubic $\times 4$ `test_LR` crops (`39 / 193 = 20.2%` exact match, `CER = 0.7423`).
- `text_eval_per_box_seed.csv` — All `1,930` `(box, seed)` OCR readings and character error rates across `seed_42`..`seed_51`.
- `text_eval_a5_cohort_breakdown.csv` & `text_eval_a5_summary.json` — A5 baseline cohort breakdown separating all `193` labeled boxes, the `136` HR-solvable boxes (`HR CER = 0`), and the `57` HR-unsolvable boxes, plus `seed 42` source PNG audit (`scripts/analyze_text_eval_a5.py`).

### 2. Per-Camera Subset Text Evaluation Tables (`VERIFIED`)
- **Canon (`images 1–50`, `10` readable images, `50` boxes — Member 2, Gia Hân):** `results/text_eval_canon_*.csv`.
- **Nikon (`images 51–100`, `15` readable images, `143` boxes — Member 3, Quang Minh):** `results/text_eval_nikon_*.csv`.

### 3. Whole-Image Metrics (`scripts/compute_metrics.py`)
- Save 10-seed whole-image `PSNR`, `SSIM`, `LPIPS`, and pairwise seed spread summary CSVs here as:
  - `results/metrics_10seed_default_summary_by_image.csv` & `results/metrics_10seed_default_per_seed.csv`
  - `results/metrics_10seed_empty_summary_by_image.csv` & `results/metrics_10seed_empty_per_seed.csv`

### 4. LoRA 2 vs. LoRA 3 Pilot Summary on Real-CE $\times 4$ (`VERIFIED`, Nov 15, 2026)
- `results/lora_pilot_summary.json` — Verified controlled `400`-step Colab pilot results (`notebooks/03_lora_pilot_colab.ipynb` `CELL 05`–`07`, `30` scene-disjoint val patches $\times$ `10` seeds `42..51`, `5` inference steps):
  - **Training Cost & VRAM (`1,659,904` trainable params):** LoRA 2 `4.67 s/step`, `8.04 GiB` allocated (`8.52 GiB` reserved); LoRA 3 `5.38 s/step` (`+15.20%`), **`8.04 GiB` allocated (`0.00 GiB` extra VRAM)**.
  - **10-Seed Whole-Patch Metrics (`LoRA 3 - LoRA 2`):** `PSNR_Y` `17.6980 -> 18.2828` (`+0.5848 dB`), `SSIM_Y` `0.6442 -> 0.6742` (`+0.0300`), `LPIPS` `0.4231 -> 0.4711` (`+0.0480`), `pair_px` `9.2715 -> 3.9405` (**`-57.50%`**), `pair_lpips` `0.2020 -> 0.0726` (**`-64.06%`**).
  - **Text-Box OCR (default RapidOCR `det+rec`, `--upscale 1`, `n=30`):** OCR `seed_disagree` `0.3456 -> 0.2470` (**`-28.53%`**), `wrong_char` `+0.0019` (`0.9838 -> 0.9857`), `CER` `+0.0147` (`0.8908 -> 0.9055`); see `docs/after_03_lora_pilot_discovery.md` for diagnostic notes and `--rec-only --rotate-vertical --upscale 2` follow-up.
