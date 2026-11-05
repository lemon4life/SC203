# Implementation Plan 02: Fix `test_LR` vs `test_HR` Input Path, Add SDPA On/Off Profile & Record Verified `2048x2048` Results

**Status:** Approved & Implemented

## 1. Objective & Context
From running `CELL 01`–`CELL 11` of `notebooks/01_dossr_baseline.ipynb` at commit `5e6a777` on Colab (`NVIDIA A100-SXM4-80GB`):
1. **Verified VAE `AttnBlock` SDPA Memory Profile (`2048x2048` output from `512x512` input):**
   - `peak_allocated` dropped from `36.04 GiB` -> **`20.51 GiB`** (`-15.53 GiB` / `-43%`).
   - `peak_reserved` dropped from `52.70 GiB` -> **`34.71 GiB`** (`-17.99 GiB` / `-34%`).
   - Sampler time: `5.28 s`, image time: `8.33 s`.
2. **Verified 1-Image Seed Sensitivity & Resume Control (`CELL 11` on `2048x2048` output):**
   - Resume check (`--skip_if_exist` + `run_manifest.json`) passed (`hits=1, misses=0`).
   - Same-seed repeat (`42 vs 42-repeat`): `max_abs_diff = 0.0`, `mean_abs_diff = 0.0` (bit-exact determinism).
   - Cross-seed spread (`42 vs 43`): `max_abs_diff = 168.0`, `mean_abs_diff = 2.9309` (8-bit pixel levels `0..255`).
3. **Discovered `RealSRVal_crop128` Folder Structure (`CELL 04` + follow-up):**
   - `RealSRVal_crop128/test_LR` contains **100 low-resolution (`128x128`) input images**.
   - `RealSRVal_crop128/test_HR` contains **100 high-resolution (`512x512`) ground-truth reference images** with matching filenames (`Canon_001_LR4.png`, etc.).
   - Because `configs/baseline_realsr.yaml` pointed to the parent directory `RealSRVal_crop128`, `rglob("*")` sorted `test_HR/` before `test_LR/` alphabetically and fed `512x512` `test_HR` images into `sr_scale: 4` (`2048x2048` output).

---

## 2. Approved Changes Implemented

1. **Harden `scripts/prepare_baseline_smoke.py`:**
   - Rejects any `input_dir` that contains subdirectories (such as `test_HR` and `test_LR`).
   - Rejects any `input_dir` with mixed image sizes across files.
   - Logs and prints the pixel size (`WxH`) of every staged image.
2. **Add Input (`128x128`) and Output (`512x512`) Size Checks in `CELL 08`, `CELL 12`, `CELL 13`, and `scripts/run_seed_study.py`:**
   - Fails loudly if any input image is not `(128, 128)` or if any generated output image is not `(512, 512)`.
3. **Protect Against `test_HR` in `src/run_dossr.py` and `configs/baseline_realsr.yaml`:**
   - Added a clear warning comment to `configs/baseline_realsr.yaml` (keeping other old configs untouched).
   - `src/run_dossr.py` refuses to run if `input_dir` is named `test_HR` or contains a `test_HR` subdirectory.
4. **Confirm `gt_dir` Behavior:**
   - `src/run_dossr.py` accepts optional extra YAML keys such as `gt_dir` in `load_config()` and does **not** pass `gt_dir` to DoSSR in `build_command()`.
5. **`README_DO_NOT_USE_AS_BASELINE.txt` Quarantine Guidance:**
   - Documented in `docs/after_01_dossr_baseline_discovery.md`, `docs/phase2_plan.md`, `docs/setup_notes.md`, and `context.md`, and automatically written by `CELL 04` into any old `2048x2048` (`test_HR`-sourced) output directories.
6. **Label Estimates Clearly & Fix OOM Wording:**
   - Any performance expectations for `test_LR` (`128x128 -> 512x512`, such as `~7-9 GiB`, `T4`, `~1-2 s per image`) are explicitly labeled **estimates (`NOT TESTED`)**.
   - Fixed the documentation wording to state accurately that **only the first, unpatched FP16 run hit the 80 GiB OOM**; subsequent `2048x2048` runs after the `CrossAttention` SDPA patch succeeded (`36.04 GiB` allocated before the VAE patch, `20.51 GiB` allocated after the VAE patch).
7. **Add SDPA Patch On/Off Comparison on 1 `test_LR` Image (`CELL 08`–`CELL 09`):**
   - Added `--disable-sdpa` (and `--disable-vae-sdpa`) flags to `src/dossr_inference_compat.py` and `src/run_dossr.py`.
   - `CELL 09` runs 1 `test_LR` image (`128x128 -> 512x512`) with SDPA patches ON and with `--disable-sdpa` (SDPA patches OFF), comparing `peak_allocated`, `peak_reserved`, sampler/image time, and `float32` pixel diff (`max_abs_diff`, `mean_abs_diff` in 8-bit `0..255` levels).
8. **Run Seed Sanity Check (`42`, `42-repeat`, `43`) on 3 `test_LR` Images (`CELL 11`):**
   - Stages 3 `test_LR` images (`Canon_001_LR4`..`Canon_003_LR4`, verified `128x128`), runs `seed 42`, `seed 42-repeat`, and `seed 43` into separate `output_root`s (`--fp16`), verifies all outputs are `512x512`, and checks the spread gate across all 3 images.
9. **Next Task Recorded in Roadmap:**
   - Recorded `scripts/compute_metrics.py` (pairing `<stem>_0.png` with `test_HR/<stem>.png`, outputting a CSV with PSNR, SSIM, and LPIPS per image and seed, plus mean and std across seeds) as the immediate next task after the `test_LR` Colab logs arrive.
