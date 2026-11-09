# Seed-robust RealSR phase: preparation plan

> **Note (`2026-11-08`):** This earlier preparation plan is superseded by **[`docs/ke_hoach_nhom_A4_A5.md`](file:///Users/lemon4life/Projects/SC203/docs/ke_hoach_nhom_A4_A5.md)**, **[`docs/plans/05_a4_a5_team_support.md`](file:///Users/lemon4life/Projects/SC203/docs/plans/05_a4_a5_team_support.md)**, and **[`docs/fact_sheet_for_report.md`](file:///Users/lemon4life/Projects/SC203/docs/fact_sheet_for_report.md)**.

No batch or seed-study inference has been run. The existing one-image smoke config was not changed.

## Prepared files

- `notebooks/01_dossr_baseline.ipynb` is the standardized Colab notebook for the `RealSRVal_crop128/test_LR` (`128x128 -> 512x512`) baseline. **`CELL 01`–`CELL 13` are VERIFIED on Colab (`251d4a5`):** GPU SDPA check (`--device cuda`), 1-image SDPA ON vs. OFF (`--disable-sdpa`) memory/time/pixel comparison (`7.32 GiB` allocated, `8.29 GiB` reserved, `1.75 s/img`, `mean_abs_diff = 0.0334 px`), output inspection (`--expected-size 512`), 3-image same-seed control vs. cross-seed spread check (`42 vs 42-repeat`: `0.0000`, `42 vs 43`: `mean_abs_diff = 8.8237 px`, `max = 255.0`), and the 5-image / 5-seed study (`seeds 42..46`, 25 `512x512` PNGs). **`CELL 14` (`scripts/compute_metrics.py`) and `CELL 15` (results zip bundle) are NOT TESTED** until Colab logs are shared.
- `scripts/compute_metrics.py` (**`NOT TESTED`**) pairs `<stem>_0.png` (`512x512`) in each `seed_<N>` folder with `RealSRVal_crop128/test_HR/<stem>.png` (`512x512`), computes full-reference **PSNR**, **SSIM**, and **LPIPS** per image and seed plus across-seed `mean`, `std`, `min`, `max`, `range`, and pairwise cross-seed spread, and writes `metrics_per_seed.csv` and `metrics_summary_by_image.csv`.
- `configs/baseline_realsr_lr128.yaml`, `configs/baseline_realsr_5img_lr128.yaml`, and `configs/seed_study_realsr_5img_lr128.yaml` point to `RealSRVal_crop128/test_LR` (`128x128` inputs) and record `gt_dir: .../RealSRVal_crop128/test_HR` (`512x512` ground truth). `src/run_dossr.py` accepts `gt_dir` without passing it to DoSSR and refuses any `input_dir` that points to or contains `test_HR`.
- `scripts/prepare_baseline_smoke.py` now stops with an error if the source directory contains subfolders or mixed image sizes, and logs the size (`WxH`) of staged images.
- `scripts/run_seed_study.py` now verifies that all prepared input images are `128x128` and all generated output images are `512x512` (**`VERIFIED`** on `251d4a5`).
- `scripts/check_attention_sdpa.py` now accepts `--device` (`auto`, `cpu`, `cuda`) and tests `CrossAttention` plus both tiny (`2x16x6x5`) and larger (`1x512x64x64`) VAE `AttnBlock` shapes.
- `scripts/inspect_output.py` checks a saved result's size, finite pixel values, and all-black/all-white cases. It saves a bicubic-upscaled input beside the DoSSR output.
- `scripts/prepare_baseline_smoke.py` now accepts `--image-count` and `--source-dir`. Defaults remain one image and the source directory in the YAML.
- `configs/baseline_realsr_5img.yaml` is a five-image batch template. It uses seed 42, FP16, CUDA, five sampling steps, x4 scale, no condition-encoder offload, and a new output root. It expects five staged inputs in `/content/realsr_batch_5`.
- `configs/seed_study_realsr_5img.yaml` and `scripts/run_seed_study.py` run seeds 42–46 into separate `seed_<N>` folders. The runner does not pass `--overwrite`; its existing `--skip_if_exist` and run-manifest checks protect completed outputs and reject mismatched runs.
- `src/run_dossr.py` now logs each image's inference duration. `--profile-memory` reports current and peak allocated/reserved CUDA memory. The peak counter resets at the start of the profiled child process, before model creation. These diagnostics are **UNTESTED on Colab**.

## Exact Colab commands

Pull the branch and run the tiny FP16 attention comparison:

```bash
cd /content/SC203
git fetch origin
git checkout feat/baseline-runner
git pull --ff-only origin feat/baseline-runner
python scripts/check_attention_sdpa.py --log-dir /content/drive/MyDrive/lemmepass/logs
```

Inspect the existing one-image result. This does not rerun DoSSR:

```bash
python scripts/inspect_output.py \
  --input /content/realsr_smoke_l4/Canon_001_LR4.png \
  --output /content/outputs/baseline-smoke-l4-fp16/seed_42/Canon_001_LR4_0.png \
  --comparison-output /content/outputs/baseline-smoke-l4-fp16/seed_42/Canon_001_LR4_compare.png \
  --expected-size 2048 --seed 42 \
  --log-dir /content/drive/MyDrive/lemmepass/logs
```

To profile the working FP16 smoke settings, copy the smoke YAML to a new config, explicitly retain FP16, and use a new output root. This leaves the working smoke config and output folder alone:

```bash
python - <<'PY'
from pathlib import Path
import yaml

source = Path('/content/baseline_smoke_l4_fp16.yaml')
target = Path('/content/baseline_smoke_l4_fp16_profile.yaml')
config = yaml.safe_load(source.read_text())
config['precision'] = 'fp16'
config['output_root'] = '/content/outputs/baseline-smoke-l4-fp16-profile-fp16'
target.write_text(yaml.safe_dump(config, sort_keys=False))
print(target)
PY
cd /content/SC203
bash scripts/run_baseline.sh --config /content/baseline_smoke_l4_fp16_profile.yaml --profile-memory --fp16
```

**Profile observed on `512x512 -> 2048x2048` (`test_HR` input, 2026-10-29 & 2026-11-05) and true `128x128 -> 512x512` (`test_LR` input, `251d4a5`):**
- **FP32 on `2048x2048` (without `--fp16`, `CrossAttention` SDPA only):** `peak_allocated=49.29 GiB`, `peak_reserved=60.48 GiB`, sampler time `18.17 s`, image time `21.05 s`, runner time `61.94 s`.
- **FP16 on `2048x2048` (with `--fp16`, before VAE `AttnBlock` patch):** `peak_allocated=36.04 GiB`, `peak_reserved=52.70 GiB`, sampler time `4.61 s`.
- **FP16 on `2048x2048` (with VAE `AttnBlock` SDPA patch, `5e6a777`):** `peak_allocated=20.51 GiB`, `peak_reserved=34.71 GiB`, sampler time `5.28 s`, image time `8.33 s`.
- **FP16 on true `RealSRVal_crop128/test_LR` (`128x128 -> 512x512`, `251d4a5`, `VERIFIED`):**
  - **SDPA ON:** `peak_allocated=7.32 GiB`, `peak_reserved=8.29 GiB`, sampler time `1.24 s`, image time `1.75 s`.
  - **SDPA OFF (`--disable-sdpa`):** `peak_allocated=7.32 GiB`, `peak_reserved=8.28 GiB`, sampler time `1.42 s`, image time `1.92 s`.
  - **Pixel diff (SDPA ON vs. OFF, 8-bit `0..255`):** `max_abs_diff=2.0000`, `mean_abs_diff=0.0334`.
  - **3-image seed check (`Canon_001`..`003`):** `42 vs 42-repeat` `max=0.0000, mean=0.0000` | `42 vs 43` `max=255.0000, mean=8.8237`.
  - **5-image / 5-seed study (`seeds 42..46`):** All 25 `512x512` PNGs verified in `seed-study-realsr-5img-lr128-fp16`.
- **Important quarantine rule for old `2048x2048` outputs:** Only the first, unpatched FP16 run on `2048x2048` hit the 80 GiB OOM; subsequent `2048x2048` runs succeeded, but they used `test_HR` (`512x512`) as input. `README_DO_NOT_USE_AS_BASELINE.txt` has been written into all old `2048x2048` output folders (`CELL 04`, `251d4a5`); do not use them as the RealSR baseline.

Stage the five lexically first RealSR images and write a temporary config from the template:

```bash
cd /content/SC203
python scripts/prepare_baseline_smoke.py \
  --config configs/baseline_realsr_5img.yaml \
  --source-dir /content/drive/MyDrive/lemmepass/data/StableSR_testsets/StableSR_testsets/RealSRVal_crop128 \
  --image-count 5 \
  --smoke-dir /content/realsr_batch_5 \
  --config-out /content/baseline_realsr_5img.yaml \
  --output-root /content/outputs/baseline-realsr-5img \
  --log-dir /content/drive/MyDrive/lemmepass/logs
```

Run the five-image batch:

```bash
bash scripts/run_baseline.sh --config /content/baseline_realsr_5img.yaml
```

After the batch completes, run the five-seed study on those same staged images:

```bash
python scripts/run_seed_study.py \
  --config configs/seed_study_realsr_5img.yaml \
  --baseline-config /content/baseline_realsr_5img.yaml
```

The seed outputs go to `/content/outputs/seed-study-realsr-5img/seed_42` through `seed_46`. Do not add `--overwrite`. If a seed folder has matching completed images, DoSSR skips them. A mismatched manifest stops the run.

Copy results to Drive before the Colab runtime ends; `/content` is temporary:

```bash
mkdir -p /content/drive/MyDrive/lemmepass/outputs
cp -Rn /content/outputs/baseline-realsr-5img /content/drive/MyDrive/lemmepass/outputs/
cp -Rn /content/outputs/seed-study-realsr-5img /content/drive/MyDrive/lemmepass/outputs/
```

## Metrics: current code and gap

- `external/DOSSR/utils/metrics.py` contains PyTorch PSNR and an LPIPS wrapper. It does not contain SSIM or no-reference metric code. No project-level metric runner or per-seed summary was found.
- PSNR, SSIM, and LPIPS require a correctly paired HR reference for each LR image. The repository config points to an LR input folder and has no GT directory or pairing map. **TODO: confirm** that this RealSR bundle contains matching HR files and how names pair.
- `scikit-image==0.26.0` and `pyiqa==0.1.16` are listed in the analysis requirements. Their installation in the Colab DoSSR environment is not confirmed. Candidate no-reference metrics include NIQE and BRISQUE through an IQA package; **TODO: confirm** metric IDs and runtime dependencies before implementation.
- Proposed `scripts/compute_metrics.py`: accept an output root, optional GT directory, and metric list; match by image stem; write one CSV row per image and seed; then report per-image variation and per-seed mean/std/range. It should refuse full-reference metrics if a GT match is missing. No metrics script was added because the GT pairing is unknown.

## VAE attention patch

The VAE `AttnBlock` in `external/DOSSR/ldm/modules/diffusionmodules/model.py` builds a `[batch, HW, HW]` table with `bmm`. `src/attention_compat.py` now runtime-patches `AttnBlock.forward` with single-head PyTorch SDPA (`scale = channels**-0.5`), keeping `norm`, `q/k/v` projections, `proj_out`, residual addition, and the original `bmm` fallback. This patch is **UNTESTED on Colab** until the FP16 memory profile is rerun.

## Ordered checklist and estimates

| Order | Step | Approximate time and Colab compute | Manual action |
|---|---|---|---|
| 1 | Pull branch; run output inspection and tiny attention check | A few minutes; CPU is enough for the tiny check, no inference GPU needed | Open the side-by-side PNG and confirm it looks reasonable |
| 2 | Profile one 512×512 input with a fresh output root | About 49 seconds based on the successful smoke run; one GPU. This run records the missing peak memory. | Record peak allocated/reserved values from the log |
| 3 | Stage and run five RealSR images | Estimate about 1 minute with a warm checkpoint: observed 49 seconds for one image plus roughly 3 seconds for each of four additional images. One GPU. Actual time is unmeasured. | Confirm five staged filenames and inspect outputs |
| 4 | Run five seeds over the same five images | Estimate about 5 minutes: five separate model runs, each approximately the five-image estimate. One GPU, sequentially. Actual time is unmeasured. | Review seed folders and logs; do not overwrite mismatched outputs |
| 5 | Compute metrics | **TODO: estimate** after confirming HR pairs and Colab metric dependencies. | Locate/verify HR files and name mapping |
| 6 | Preserve outputs | Transfer time depends on output size. No GPU compute required. | Copy `/content/outputs` results into Drive before runtime shutdown |

The time estimates assume the same runtime as the successful smoke and a warm checkpoint cache. The earlier cold checkpoint deserialization took 123.8 seconds, so the first run after a fresh runtime may take longer. No five-image or multi-seed job has been run.
