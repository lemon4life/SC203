# Implementation Plan 03: Record Verified `CELL 01`–`CELL 13` `test_LR` (`128x128 -> 512x512`) Results & Add `scripts/compute_metrics.py`

## 1. Context & Verified Findings from Colab (`CELL 01`–`CELL 13`, Commit `251d4a5`)

All 13 cells of [`notebooks/01_dossr_baseline.ipynb`](file:///Users/lemon4life/Projects/SC203/notebooks/01_dossr_baseline.ipynb) (`CELL 01`–`CELL 13`) have now been executed and verified on Colab using the true `RealSRVal_crop128/test_LR` (`128x128` $\rightarrow$ `512x512` at `sr_scale: 4`) inputs:

1. **Dataset & Pairing (`CELL 04` — `VERIFIED`):**
   - `RealSRVal_crop128/test_LR`: 100 images, all `(128, 128)`.
   - `RealSRVal_crop128/test_HR`: 100 images, all `(512, 512)`.
   - Verified 1-to-1 filename pairing (`Canon_001_LR4.png` .. `Nikon_050_LR4.png`) across all 100 images.
   - Quarantined all 3 old `2048x2048` folders (`baseline-smoke-vae-sdpa-fp16`, `seed-sanity-repeat42-fp16`, `seed-sanity-seed43-fp16`) with `README_DO_NOT_USE_AS_BASELINE.txt`.
2. **1-Image `test_LR` (`128x128 -> 512x512`) SDPA ON vs. SDPA OFF (`CELL 08` & `CELL 09` — `VERIFIED`):**
   - **SDPA ON:** `peak_allocated = 7.32 GiB` (`6.44 GiB` model weights + `0.88 GiB` activations), `peak_reserved = 8.29 GiB`, `sampler = 1.24 s`, `image = 1.75 s`.
   - **SDPA OFF (`--disable-sdpa`):** `peak_allocated = 7.32 GiB`, `peak_reserved = 8.28 GiB`, `sampler = 1.42 s`, `image = 1.92 s`.
   - **Pixel diff (SDPA ON vs. OFF, 8-bit `0..255`):** `max_abs_diff = 2.0000`, `mean_abs_diff = 0.0334`.
   - **Hardware conclusion:** Because `peak_reserved = 8.29 GiB`, an A100 is not required for `128x128 -> 512x512` runs; both **L4 (`22.5 GiB` VRAM)** and **T4 (`14.56 GiB` VRAM, with High-RAM enabled)** fit comfortably.
3. **Output Inspection (`CELL 10` — `VERIFIED`):**
   - `inspect_output.py --expected-size 512` passed (`Canon_001_LR4_0.png` is `512x512`, finite, non-degenerate).
   - Wrote `/content/sc203_status/cell10_inspection_status.json` (`passed: true`).
4. **Resume Check & 3-Image `test_LR` Seed Sensitivity (`CELL 11` — `VERIFIED`):**
   - Resume path verified (`OUTPUT CACHE | hits=1 | misses=0`).
   - 3-image (`Canon_001_LR4`, `Canon_002_LR4`, `Canon_003_LR4`) seed check (`512x512` outputs):
     - `Canon_001_LR4_0.png`: `(42 vs 42-repeat)` `max=0.0, mean=0.0` | `(42 vs 43)` `max=185.0, mean=5.1717`
     - `Canon_002_LR4_0.png`: `(42 vs 42-repeat)` `max=0.0, mean=0.0` | `(42 vs 43)` `max=223.0, mean=11.9893`
     - `Canon_003_LR4_0.png`: `(42 vs 42-repeat)` `max=0.0, mean=0.0` | `(42 vs 43)` `max=255.0, mean=9.3100`
     - **Aggregate (3 images):** `(42 vs 42-repeat)` `max=0.0000, mean=0.0000` | `(42 vs 43)` `max=255.0000, mean=8.8237`
   - Wrote `/content/sc203_status/cell11_seed_sanity_status.json` (`passed: true`).
5. **5-Image Batch & 5-Seed Study (`CELL 12` & `CELL 13` — `VERIFIED`):**
   - Disk gate check passed (`HEAD=251d4a5`).
   - Staged 5 `test_LR` images (`(128, 128)`) in `/content/realsr_batch_5_lr128`.
   - Completed 5-image baseline batch (`baseline-realsr-5img-lr128-fp16/seed_42`) and 5-seed study (`seed-study-realsr-5img-lr128-fp16/seed_42` through `seed_46`), verifying all 25 generated PNGs have size `(512, 512)` and `precision=fp16`.

---

## 2. Proposed Changes

### A. Create [`scripts/compute_metrics.py`](file:///Users/lemon4life/Projects/SC203/scripts/compute_metrics.py) (`NOT TESTED` until run on Colab)
1. **Inputs & Pairing Rules:**
   - `--output-root`: Root folder containing `seed_<N>/` subdirectories (e.g., `/content/drive/MyDrive/lemmepass/outputs/seed-study-realsr-5img-lr128-fp16`).
   - `--gt-dir`: Ground-truth HR directory (e.g., `.../RealSRVal_crop128/test_HR`).
   - `--expected-size`: Expected width and height (default `512`). Fails loudly if any predicted `<stem>_0.png` or ground-truth `<stem>.png` does not match `(expected_size, expected_size)` or if any predicted image is missing its ground-truth pair in `--gt-dir`.
   - `--seeds`: Optional comma-separated list of seeds to require/filter (e.g. `42,43,44,45,46`).
   - `--crop-border`: Edge border pixels to crop before PSNR/SSIM (default `0`, configurable).
   - `--test-y-channel`: Compute Y-channel PSNR/SSIM in addition to RGB PSNR/SSIM.
   - `--lpips-net`: LPIPS backbone (`alex` by default, or `vgg`).
   - `--device`: `auto` (default, uses `cuda` if available else `cpu`), `cuda`, or `cpu`.
   - `--per-image-csv`: Path to write per-(image, seed) metrics CSV (`image_stem, seed, psnr_rgb, ssim_rgb, psnr_y, ssim_y, lpips, pred_path, gt_path`).
   - `--summary-csv`: Path to write per-image across-seed summary CSV (`mean`, `std`, `min`, `max`, `range`, best/worst seed per metric, plus mean/max pairwise cross-seed pixel diff and pairwise cross-seed LPIPS).
2. **Metrics Computed:**
   - **PSNR (dB):** Standard float64 MSE-based PSNR on `[0, 1]` RGB and Y-channel (YCbCr), matching [`external/DOSSR/utils/metrics.py`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/utils/metrics.py#L9-L39).
   - **SSIM:** Standard Wang et al. (2004) / BasicSR `11x11` Gaussian window (`sigma=1.5`, $C_1 = 0.01^2$, $C_2 = 0.03^2$) implemented in PyTorch `conv2d` on float64 `[0, 1]` tensors (both RGB and Y-channel) with zero extra pip dependencies.
   - **LPIPS:** Full-reference perceptual distance (`lpips.LPIPS`, normalized from `[0, 1]` to `[-1, 1]`), plus **pairwise cross-seed LPIPS** and **pairwise cross-seed 8-bit pixel diff** across all $\binom{5}{2} = 10$ seed pairs per image.

### B. Update [`notebooks/01_dossr_baseline.ipynb`](file:///Users/lemon4life/Projects/SC203/notebooks/01_dossr_baseline.ipynb)
- Update `CELL 04`, `CELL 08`, `CELL 09`, `CELL 10`, `CELL 11`, `CELL 12`, and `CELL 13` markdown headers from `NOT TESTED` to `VERIFIED` (recording the exact `251d4a5` Colab results).
- Add **`CELL 14` (`Status: NOT TESTED`)**: Runs `scripts/compute_metrics.py` on `OUTPUT_BASE / "seed-study-realsr-5img-lr128-fp16"` against `REALSR_HR_DIR`, saves `metrics_per_seed.csv` and `metrics_summary_by_image.csv` into the seed-study folder on Drive, and prints the summary table.
- Add **`CELL 15` (`Status: NOT TESTED`)**: Packs the metrics CSVs, status JSONs (`cell10_inspection_status.json`, `cell11_seed_sanity_status.json`), `Canon_001_LR4_compare.png`, and latest log excerpts (`.txt`) into `/content/sc203_baseline_results_bundle.zip` and copies it to Drive so it can be downloaded and committed to GitHub for teammates.

### C. Update Project Documentation with Verified `CELL 01`–`CELL 13` Results
- Update [`context.md`](file:///Users/lemon4life/Projects/SC203/context.md), [`docs/after_01_dossr_baseline_discovery.md`](file:///Users/lemon4life/Projects/SC203/docs/after_01_dossr_baseline_discovery.md), [`docs/phase2_plan.md`](file:///Users/lemon4life/Projects/SC203/docs/phase2_plan.md), and [`docs/setup_notes.md`](file:///Users/lemon4life/Projects/SC203/docs/setup_notes.md) to mark `CELL 01`–`CELL 13` as `VERIFIED` (`251d4a5`), while keeping `scripts/compute_metrics.py`, `CELL 14`, and `CELL 15` explicitly marked `NOT TESTED` until you share the `CELL 14` Colab output.
- Archive this plan to `docs/plans/03_compute_metrics_and_record_lr128_results.md` **only after** you approve this plan.

---

## 3. Constraints Checklist
- Do **not** modify anything under `external/DOSSR` or `external/InstructIR`.
- Keep all unexecuted code (`scripts/compute_metrics.py`, `CELL 14`, `CELL 15`) labeled `NOT TESTED` until Colab logs are provided.
- Save this plan to `docs/plans/03_compute_metrics_and_record_lr128_results.md` only after user approval.
