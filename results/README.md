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
- Save 10-seed and deterministic whole-image `PSNR`, `SSIM`, `LPIPS`, and pairwise seed spread summary CSVs here as:
  - `results/metrics_10seed_default_summary_by_image.csv` & `results/metrics_10seed_default_per_seed.csv`
  - `results/metrics_10seed_empty_summary_by_image.csv` & `results/metrics_10seed_empty_per_seed.csv`
  - `results/metrics_realsr_bicubic_x4_summary.csv` — Deterministic Bicubic $\times 4$ whole-image floor on `RealSRVal_crop128` (`100` images: `PSNR_RGB = 25.48 dB`, `SSIM_RGB = 0.7189`, `PSNR_Y = 27.11 dB`, `SSIM_Y = 0.7561`, `LPIPS = 0.4563`).

### 4. LoRA 2 vs. LoRA 3 Pilot Summary on Real-CE $\times 4$ (`VERIFIED`, Nov 15–16, 2026)
- `results/lora_pilot_summary.json` — Controlled `400`-step Colab pilot results (`notebooks/03_lora_pilot_colab.ipynb` `CELL 05`–`07`, `30` scene-disjoint val patches $\times$ `10` seeds `42..51`, `5` inference steps):
  - **Training Cost & VRAM (`1,659,904` trainable params):** LoRA 2 `4.67 s/step`, `8.04 GiB` allocated (`8.52 GiB` reserved); LoRA 3 `5.38 s/step` (`+15.20%`), **`8.04 GiB` allocated (`0.00 GiB` extra VRAM)**.
  - **10-Seed Whole-Patch Metrics (`LoRA 3 - LoRA 2`):** `PSNR_Y` `17.6980 -> 18.2828` (`+0.5848 dB`), `SSIM_Y` `0.6442 -> 0.6742` (`+0.0300`), `LPIPS` `0.4231 -> 0.4711` (`+0.0480`), `pair_px` `9.2715 -> 3.9405` (**`-57.50%`**), `pair_lpips` `0.2020 -> 0.0726` (**`-64.06%`**).
  - **Corrected OCR Re-Score (`results/pilot_lora2_vs_lora3_reconly/`, `results/pilot_lora3_reconly/` — RapidOCR `--rec-only --rotate-vertical --upscale 2`, `30` patches / `154` boxes, M3 on Nov 16):**
    - `HR Ceiling` CER improved from `0.7334` (`det+rec`) to `0.2722` (`0.5791` wrong-char), confirming the earlier `det+rec` run suffered detector misses on tight crops.
    - **Image-mean (`n=30`):** LoRA 3 beats LoRA 2 across all text metrics: `wrong_char` `0.9141 -> 0.8878` (**`-0.0262 ± 0.0084`**), `CER` `0.6914 -> 0.6650` (**`-0.0264 ± 0.0122`**), `seed_disagree` `0.5197 -> 0.4479` (**`-13.82%`**), `crop_pixel_seed_spread_l1` `15.8839 -> 7.1348` (**`-55.08%`**).
    - **HR-solvable cohort (`63 / 154` boxes where HR wrong-char == 0):** `Bicubic x4` `0.5556` (`CER 0.4184`), `LoRA 2` `0.8222` (`CER 0.6144`), `LoRA 3` `0.7603` (`CER 0.5803`, **`-0.0619`** error rate delta).
  - **Visual Crop Grids (`results/pilot_lora2_vs_lora3_crops/`, notes in `docs/M3_3_pilot_reconly_and_crops_notes.md`):** 5 multi-seed grids comparing `LR bicubic`, `LoRA 2`, `LoRA 3`, and `HR`.
