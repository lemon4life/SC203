# context.md: read this first

You are joining Team 1 (LEMMEPASS) in SC203 Scientific Methods (class 25A02). This file tells you what the project is, what is done, what is not, and what to do next. Facts below come from our real logs (Oct 29 & Nov 5, 2026). Anything not confirmed is marked **TODO: confirm**. Do not guess it. Ask the user.

## 1. Project

- **Topic 3:** seed-robust, instruction-conditioned real-world super-resolution.
- **Base model:** DoSSR (public code and pretrained checkpoint). We add text conditioning later. InstructIR is a reference for the instruction idea.
- **Research question:** how much does the random seed change the output, and does text conditioning help or hurt stability?
- **Team:** 4 members. The user runs things on Google Colab. The user works on an Apple Silicon Mac (no NVIDIA GPU, no PyTorch locally). Teammates use Windows. So **you cannot run GPU inference locally**. The user runs your commands in Colab and sends back logs.

## 2. Repo

```bash
git clone https://github.com/lemon4life/SC203
cd SC203
git checkout feat/baseline-runner   # all current work is on this branch
```

- Work branch: `feat/baseline-runner` (as of `251d4a5`; before it: `5e6a777`, `dfc8d96`, `cdc9a71`, `d066ade`, `1af0178`). TODO: confirm if `main` has been updated (`origin/main` was at initial commit `f08e303` when checked locally).
- `external/DOSSR/` is the upstream DoSSR code. **Never edit it.** It is ignored in `.gitignore` (plain clone, not a submodule); `scripts/colab_setup.py` clones `https://github.com/AMD-AIG-AIMA/DOSSR.git` automatically when `external/DOSSR` is absent on a fresh machine.

Files we know about:

| Path | What it is |
|---|---|
| `notebooks/01_dossr_baseline.ipynb` | Standardized team Colab notebook: **`CELL 01`–`CELL 13` are VERIFIED** (`251d4a5`: setup, `test_LR` vs `test_HR` verification, GPU SDPA check, 1-image `test_LR` `128x128 -> 512x512` SDPA ON vs OFF profile, `512x512` output inspection, 3-image seed sensitivity control, and 5-image / 5-seed study `seeds 42..46`). **`CELL 14` (`compute_metrics.py`) and `CELL 15` (results zip bundle) are NOT TESTED.** |
| `notebooks/colab_l4_smoke.ipynb`, `notebooks/colab_runner.ipynb`, `notebooks/kaggle_runner.ipynb` | Earlier smoke and runner notebooks (kept until team confirms cleanup). |
| `scripts/run_baseline.sh` | Entry point. Calls the runner. |
| `src/run_dossr.py` | Runner. Reads a YAML config (accepts optional `gt_dir` without passing it to DoSSR; refuses `input_dir` pointing to or containing `test_HR`; supports `--disable-sdpa` and `--disable-vae-sdpa`), builds the DoSSR command, writes logs, saves outputs per seed. |
| `src/dossr_inference_compat.py` | Launcher around upstream `inference.py`. Turns on fp16 autocast, loads the checkpoint with mmap, skips a missing author-local OpenCLIP file, installs our `CrossAttention` and `VAE AttnBlock` patches (unless `--disable-sdpa` / `--disable-vae-sdpa` is passed), supports `--profile-memory`. |
| `src/attention_compat.py` | Our SDPA patches for `CrossAttention` and VAE `AttnBlock`. |
| `scripts/check_attention_sdpa.py` | Small test: old einsum `CrossAttention` and bmm `VAE AttnBlock` (tiny `2x16x6x5` and larger `1x512x64x64`) vs SDPA on random fp16 tensors (`--device auto|cpu|cuda`). |
| `scripts/inspect_output.py` | Checks output size, NaN/Inf, all-black, all-white. Saves a side-by-side (bicubic vs DoSSR). |
| `scripts/prepare_baseline_smoke.py` | Copies N images from a flat leaf folder into a staging folder (stops with an error if the source folder has subfolders or mixed image sizes; logs each staged image's `WxH` size) and writes a YAML. |
| `scripts/run_seed_study.py` | Runs several seeds into separate folders and verifies input (`128x128`) and output (`512x512`) image sizes. **VERIFIED on `test_LR` (`251d4a5`).** |
| `scripts/compute_metrics.py` | Pairs `<stem>_0.png` in each `seed_<N>` folder with `test_HR/<stem>.png`, verifies `512x512` sizes, computes PSNR, SSIM, and LPIPS per image and seed plus across-seed summary and pairwise cross-seed spread, and writes `metrics_per_seed.csv` and `metrics_summary_by_image.csv`. **NOT TESTED.** |
| `scripts/verify_assets.py` | Checks that checkpoint, weights and test bundle are on Drive. |
| `configs/baseline_realsr_lr128.yaml`, `configs/baseline_realsr_5img_lr128.yaml`, `configs/seed_study_realsr_5img_lr128.yaml` | Corrected configs pointing `input_dir` to `RealSRVal_crop128/test_LR` (`128x128` -> `512x512` at x4) and recording `gt_dir: .../RealSRVal_crop128/test_HR` (`512x512`). **VERIFIED (`251d4a5`).** |
| `configs/baseline_realsr.yaml`, `configs/baseline_realsr_5img.yaml`, `configs/seed_study_realsr_5img.yaml` | Legacy configs (kept untouched except a warning comment in `baseline_realsr.yaml`). |
| `docs/after_01_dossr_baseline_discovery.md`, `docs/plans/` | Discovery report (`2026-11-05`) and numbered implementation plans archive. |
| `docs/setup_notes.md`, `docs/phase2_plan.md`, `docs/colab_workflow.md` | Notes, workflow guide, and phase 2 plan. |

## 3. Environment (Colab)

- torch 2.11.0+cu130, Python 3.13.16, Linux. **No xformers.**
- **Confirmed GPU (`CELL 03`, Nov 5, 2026):** `NVIDIA A100-SXM4-80GB` (`79.25 GiB` total VRAM).
- Paths on Colab:
  - Drive base: `/content/drive/MyDrive/lemmepass/`
  - Checkpoint: `.../weights/dossr_default.ckpt`
  - Logs: `.../logs/`
  - Test data (`RealSRVal_crop128` contains two subfolders with identical filenames `Canon_001_LR4.png`..):
    - **LR inputs (`128x128`, 100 images):** `.../data/StableSR_testsets/StableSR_testsets/RealSRVal_crop128/test_LR`
    - **HR ground truth (`512x512`, 100 images):** `.../data/StableSR_testsets/StableSR_testsets/RealSRVal_crop128/test_HR`
  - Outputs: `/content/drive/MyDrive/lemmepass/outputs/...` (or `/content/outputs/...` which is wiped when Colab restarts)
- Baseline settings: steps 5, solver_order 3, `sr_scale` 4, color fix wavelet, preprocessor RealESRNet, seed 42, no condition-encoder offload.

## 4. What is done

1. Assets on Drive, `verify_assets.py` passed.
2. First run (unpatched FP16 on `512x512 -> 2048x2048`) failed: `torch.OutOfMemoryError`, 80 GiB request, in `external/DOSSR/ldm/modules/attention.py` line 178 (`einsum('b i d, b j d -> b i j', q, k)`). Only that first, unpatched FP16 run hit the 80 GiB OOM.
3. Upstream `--tiled --tile_size --tile_stride` **do not do real tiling** (checked in the code). Do not suggest them as a memory fix.
4. Fix 1 (`CrossAttention` SDPA patch) + Fix 2 (`VAE AttnBlock` SDPA patch, `cdc9a71` / `5e6a777`):
   - Verified on `512x512 -> 2048x2048` (`test_HR` input stress test):

| Precision & Patches (`512x512 -> 2048x2048`) | `peak_allocated` | `peak_reserved` | Sampler time | Image time |
|---|---|---|---|---|
| `fp32` (`CrossAttention` SDPA only) | `49.29 GiB` | `60.48 GiB` | `18.17 s` | `21.05 s` |
| `fp16` (`CrossAttention` SDPA only, before VAE patch) | `36.04 GiB` | `52.70 GiB` | `4.61 s` | `7.68 s` |
| **`fp16` (`CrossAttention` + `VAE AttnBlock` SDPA, `5e6a777`)** | **`20.51 GiB`** | **`34.71 GiB`** | **`5.28 s`** | **`8.33 s`** |

5. Verified resume check (`OUTPUT CACHE | hits=1 | misses=0`) and 1-image seed control on `2048x2048` (`5e6a777`):
   - `42 vs 42-repeat`: `max_abs_diff = 0.0`, `mean_abs_diff = 0.0` (8-bit pixel levels `0..255`).
   - `42 vs 43`: `max_abs_diff = 168.0`, `mean_abs_diff = 2.9309`.
6. **Critical Dataset Discovery (`CELL 04`, Nov 5, 2026):**
   - Earlier configs pointed `input_dir` at the parent `RealSRVal_crop128` directory. Because `rglob("*")` sorted paths alphabetically, `test_HR` (`512x512`) was picked before `test_LR` (`128x128`) and upscaled by `sr_scale: 4` to `2048x2048`.
   - **Quarantine rule:** All old `2048x2048` output folders (`baseline-smoke-l4-fp16`, `baseline-smoke-vae-sdpa-fp16`, `seed-sanity-repeat42-fp16`, `seed-sanity-seed43-fp16`, `baseline-realsr-5img`) have been marked with `README_DO_NOT_USE_AS_BASELINE.txt` (`CELL 04`, `251d4a5`) and must **not** be used as the RealSR baseline.
7. **Verified True `test_LR` (`128x128 -> 512x512` at `sr_scale: 4`) Runs (`CELL 04`–`CELL 13`, `251d4a5`):**
   - **1-image `test_LR` SDPA ON vs. SDPA OFF (`--disable-sdpa`) (`CELL 08`–`CELL 09`):**
     - **SDPA ON:** `peak_allocated = 7.32 GiB`, `peak_reserved = 8.29 GiB`, `sampler = 1.24 s`, `image = 1.75 s`.
     - **SDPA OFF (`--disable-sdpa`):** `peak_allocated = 7.32 GiB`, `peak_reserved = 8.28 GiB`, `sampler = 1.42 s`, `image = 1.92 s`.
     - **Pixel diff (SDPA ON vs. OFF, 8-bit `0..255`):** `max_abs_diff = 2.0000`, `mean_abs_diff = 0.0334`.
     - **Hardware fit:** At `8.29 GiB` peak reserved, `128x128 -> 512x512` runs fit comfortably on standard Colab **T4 (`14.56 GiB`)** or **L4 (`22.5 GiB`)** GPUs (with High-RAM enabled for host checkpoint loading) without requiring an A100.
   - **3-image `test_LR` (`Canon_001`..`Canon_003`) seed sensitivity check (`CELL 11`):**
     - `42 vs 42-repeat`: `max_abs_diff = 0.0000`, `mean_abs_diff = 0.0000` across all 3 images.
     - `42 vs 43`: `max_abs_diff = 255.0000`, `mean_abs_diff = 8.8237` (`5.1717`, `11.9893`, `9.3100` on `Canon_001`..`003`).
   - **5-image / 5-seed study (`seeds 42..46` on `test_LR` in `CELL 12`–`CELL 13`):**
     - All 25 outputs (`5 images x 5 seeds`) in `seed-study-realsr-5img-lr128-fp16` verified at `(512, 512)` in FP16.

## 5. What is NOT done

- **Running `scripts/compute_metrics.py` (`CELL 14` & `CELL 15` of `notebooks/01_dossr_baseline.ipynb`, `NOT TESTED`):**
  - `scripts/compute_metrics.py` is written locally (`NOT TESTED` on Colab) to pair `<stem>_0.png` (`512x512`) with `test_HR/<stem>.png` (`512x512`), compute **PSNR**, **SSIM**, and **LPIPS** per image and seed plus across-seed summary and pairwise cross-seed spread, and write `metrics_per_seed.csv` and `metrics_summary_by_image.csv`.
- **Scaling up beyond 5 images:** 20-image and 100-image `test_LR` runs across seeds are not run yet.
- Text conditioning is **not designed or built**. The method is undecided.

## 6. Next tasks, in order

1. **Run `CELL 14` and `CELL 15` of `notebooks/01_dossr_baseline.ipynb` (`NOT TESTED`):**
   - Pull the latest commit (`CELL 02`), run `CELL 14` (`scripts/compute_metrics.py` on the 25 verified images in `seed-study-realsr-5img-lr128-fp16`), and run `CELL 15` to package `sc203_baseline_results_bundle.zip`.
2. **Plot & analyze seed variance (Teammate 2):** build boxplots/tables from `metrics_per_seed.csv` and `metrics_summary_by_image.csv`, plus per-pixel variance heatmaps across seeds.
3. **Scale up (Member 1):** test 20 `test_LR` images first and measure real time, then the full 100-image `test_LR` set.
4. **Text conditioning & instruction prompts (Teammates 3 & 4):** prepare restoration instructions (`RealSRVal_crop128/test_LR`) and propose/prototype 2 lightweight instruction-conditioning designs (referencing InstructIR). Do not build the full training pipeline until the team and mentor choose.
5. **Docs:** keep `docs/`, `docs/plans/`, and the weekly report updated.

## 7. Rules

- Always follow the **Team Notebook & Experimentation Guidelines**: whenever a task requires running code on Google Colab, create or update the `.ipynb` notebook in `notebooks/` rather than giving isolated snippets.
- Before coding or modifying anything, create a numbered implementation plan (`<NN>_<brief_context>.md`) for the user to review, ask questions if needed, and wait for explicit permission. Only add the plan file into `docs/plans/` in the repo **after** the user approves/proceeds.
- Do not edit anything under `external/`. Use runtime patches or our own files.
- Do not change a working config. Make a new one, and use a **new `output_root` for every run** (the runner keeps a manifest and `--skip_if_exist`).
- fp16 only happens with `--fp16` (or `precision: fp16` in the config). Check that the log has `enabling CUDA float16 autocast`. Check that the log has `using PyTorch scaled_dot_product_attention`.
- Never claim a result you did not see in a log. Say **NOT TESTED** for code that was not run. If you did not run it (no GPU here), say so.
- Do not invent numbers, dates, paper details or file contents. Write `TODO: confirm` and ask.
- Keep changes small. Show a diff, the exact Colab instructions/notebook cells, the expected success signal, and what to send back if it fails (last 50 log lines and the full traceback).
- End each task with: commit hash, changed files, and the push command (`git push origin feat/baseline-runner`). The user pushes and pulls by hand.
- Write docs and explanations in **simple English, short sentences**. The user asked for simpler words several times.
- Credit clearly: the model, the method, the checkpoint and `external/DOSSR` are the DoSSR authors'. Our code is the runner, the compatibility launcher, the patches, the checks and the study scripts. Say if AI agents helped write code when asked.

## 8. Open questions for the user (ask, do not guess)

1. ~~Exact GPU model in Colab?~~ **Confirmed:** `NVIDIA A100-SXM4-80GB` (`79.25 GiB`), and verified that `test_LR` (`128x128 -> 512x512`) uses only `7.32 GiB` allocated / `8.29 GiB` reserved (`1.75 s/img`), so T4 or L4 also fits easily.
2. ~~Do we have HR images for RealSR?~~ **Confirmed:** `RealSRVal_crop128/test_HR` (`100` images, `512x512`). TODO: confirm if the course or mentor wants any additional metrics beyond PSNR, SSIM, and LPIPS.
3. ~~How many images are in the full RealSR folder, and what are their sizes?~~ **Confirmed:** `100` LR images (`128x128`) in `test_LR` and `100` HR images (`512x512`) in `test_HR`.
4. Course deadlines and the plan per stage?
5. Who on the team does what?
6. Has the mentor replied, and what did they advise?

## 9. Quick Colab execution

Since `CELL 01`–`CELL 13` are already verified and all 25 images are saved on Drive (`seed-study-realsr-5img-lr128-fp16`), open `notebooks/01_dossr_baseline.ipynb` in Google Colab and run `CELL 01` $\rightarrow$ `CELL 02` (`git pull`) $\rightarrow$ `CELL 14` (`compute_metrics.py`) $\rightarrow$ `CELL 15` (export `.zip`).
