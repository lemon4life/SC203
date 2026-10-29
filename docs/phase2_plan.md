# Seed-robust RealSR phase: preparation plan

No batch or seed-study inference has been run. The existing one-image smoke config was not changed.

## Prepared files

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

To obtain the missing 512×512-input peak-memory measurement, copy the working smoke YAML to a new profile config and use a new output root. This leaves the working smoke config and output folder alone:

```bash
python - <<'PY'
from pathlib import Path
import yaml

source = Path('/content/baseline_smoke_l4_fp16.yaml')
target = Path('/content/baseline_smoke_l4_fp16_profile.yaml')
config = yaml.safe_load(source.read_text())
config['output_root'] = '/content/outputs/baseline-smoke-l4-fp16-profile'
target.write_text(yaml.safe_dump(config, sort_keys=False))
print(target)
PY
cd /content/SC203
bash scripts/run_baseline.sh --config /content/baseline_smoke_l4_fp16_profile.yaml --profile-memory
```

The supplied run facts do not include a peak-memory number. **TODO: confirm** it from this profile log. Expected fields include `peak_allocated=... GiB` and `peak_reserved=... GiB`. Each image also gets an `IMAGE PROFILE` line.

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

## VAE attention risk

The VAE `AttnBlock` in `external/DOSSR/ldm/modules/diffusionmodules/model.py` builds a `[batch, HW, HW]` table with `bmm`. It completed the supplied 2048×2048 output smoke run, but its table grows quadratically with spatial token count. Larger images may exceed GPU memory. No threshold is known. If a larger run fails there, apply an SDPA rewrite to that block with one head and scale `channels**-0.5`; do not change it without approval.

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
