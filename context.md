# context.md: read this first

You are joining Team 1 (LEMMEPASS) in SC203 Scientific Methods (class 25A02). This file tells you what the project is, what is done, what is not, and what to do next. Facts below come from our real logs (Oct 29, 2026). Anything not confirmed is marked **TODO: confirm**. Do not guess it. Ask the user.

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

- Work branch: `feat/baseline-runner` (as of `cdc9a71`; before it: `d066ade`, `1af0178`, `75baf86`, `da3bcbd`). TODO: confirm if `main` has been updated (`origin/main` was at initial commit `f08e303` when checked locally).
- `external/DOSSR/` is the upstream DoSSR code. **Never edit it.** It is ignored in `.gitignore` (plain clone, not a submodule); `scripts/colab_setup.py` clones `https://github.com/AMD-AIG-AIMA/DOSSR.git` automatically when `external/DOSSR` is absent on a fresh machine. TODO: confirm if any additional steps were in the original `SC203_Phase1_RealSR.ipynb`.

Files we know about:

| Path | What it is |
|---|---|
| `notebooks/01_dossr_baseline.ipynb` | Standardized team Colab notebook (`CELL 01`–`CELL 13`): setup, GPU SDPA check, 1-image FP16 memory profile, output inspection, 1-image seed sensitivity control (`42 vs 42-repeat` and `42 vs 43`), and `RUN_SEED_STUDY`-gated 5-seed study. **Written, NOT TESTED.** |
| `notebooks/colab_l4_smoke.ipynb`, `notebooks/colab_runner.ipynb`, `notebooks/kaggle_runner.ipynb` | Earlier smoke and runner notebooks (kept until team confirms cleanup). |
| `scripts/run_baseline.sh` | Entry point. Calls the runner. |
| `src/run_dossr.py` | Runner. Reads a YAML config, builds the DoSSR command, writes logs, saves outputs per seed. |
| `src/dossr_inference_compat.py` | Launcher around upstream `inference.py`. Turns on fp16 autocast, loads the checkpoint with mmap, skips a missing author-local OpenCLIP file, installs our `CrossAttention` and `VAE AttnBlock` patches, supports `--profile-memory`. |
| `src/attention_compat.py` | Our SDPA patches for `CrossAttention` and VAE `AttnBlock` (`patch_vae_attn_block` added in `cdc9a71`, **NOT TESTED** on Colab). |
| `scripts/check_attention_sdpa.py` | Small test: old einsum `CrossAttention` and bmm `VAE AttnBlock` (tiny `2x16x6x5` and larger `1x512x64x64`) vs SDPA on random fp16 tensors (`--device auto|cpu|cuda`). **VAE check and `--device cuda` are NOT TESTED.** |
| `scripts/inspect_output.py` | Checks output size, NaN/Inf, all-black, all-white. Saves a side-by-side (bicubic vs DoSSR). |
| `scripts/prepare_baseline_smoke.py` | Copies N images into a staging folder and writes a YAML. |
| `scripts/run_seed_study.py` | Runs several seeds into separate folders. **Written, NOT TESTED.** |
| `scripts/verify_assets.py` | Checks that checkpoint, weights and test bundle are on Drive. |
| `configs/baseline_realsr.yaml` | Default config. **Its default is `precision: fp32`.** |
| `configs/baseline_realsr_5img.yaml`, `configs/seed_study_realsr_5img.yaml` | 5-image run and seed study (seeds 42 to 46). |
| `docs/setup_notes.md`, `docs/phase2_plan.md`, `docs/colab_workflow.md` | Notes, workflow guide, and the plan written by earlier agents. |

## 3. Environment (Colab)

- torch 2.11.0+cu130, Python 3.13.16, Linux. **No xformers.**
- GPU in our runs has about 79 GiB. TODO: confirm the exact model (`nvidia-smi`). Configs are named "l4", but a real L4 has 24 GB and would probably fail.
- Paths on Colab:
  - Drive base: `/content/drive/MyDrive/lemmepass/`
  - Checkpoint: `.../weights/dossr_default.ckpt`
  - Logs: `.../logs/`
  - Test data: `.../data/StableSR_testsets/StableSR_testsets/RealSRVal_crop128`
  - Outputs: `/content/outputs/...` (**wiped when Colab restarts**; copy to Drive or set `output_root` under `/content/drive/MyDrive/lemmepass/outputs/`)
- Baseline settings: steps 5, solver_order 3, `sr_scale` 4, color fix wavelet, preprocessor RealESRNet, seed 42, no condition-encoder offload.

## 4. What is done

1. Assets on Drive, `verify_assets.py` passed.
2. First run failed: `torch.OutOfMemoryError`, 80 GiB request, in `external/DOSSR/ldm/modules/attention.py` line 178 (`einsum('b i d, b j d -> b i j', q, k)`). Cause: full attention table on the 2048×2048 image (512×512 input, x4). Model weights are only about 6.5 GiB.
3. Upstream `--tiled --tile_size --tile_stride` **do not do real tiling** (checked in the code). Do not suggest them as a memory fix. The runner has optional `tiled` keys, default false, harmless.
4. Fix: runtime patch using `torch.nn.functional.scaled_dot_product_attention` for `CrossAttention` (old code is the fallback). Check passed: max abs error 0.000977 (rtol = atol = 0.01).
5. One-image smoke test (Canon_001_LR4, fp16): passed. Runner time 48.99 s, image time 7.68 s.
6. Output inspection passed: 2048×2048, no NaN/Inf, not black, not white.
7. 5-image batch (Canon_001 to 005, fp16, seed 42): passed. 77.32 s total, 6.72 to 7.79 s per image.
8. Memory profile, one image, 512×512 input (before VAE `AttnBlock` patch):

| Precision | peak_allocated | peak_reserved | Sampler time |
|---|---|---|---|
| fp32 | 49.29 GiB | 60.48 GiB | 18.17 s |
| fp16 | 36.04 GiB | 52.70 GiB | 4.61 s |

## 5. What is NOT done

- VAE `AttnBlock` SDPA patch is **written in code (`cdc9a71`), NOT TESTED on Colab**. `AttnBlock` in `external/DOSSR/ldm/modules/diffusionmodules/model.py` (around line 181) previously built a full `[batch, HW, HW]` table with `bmm`. **Hypothesis, not measured:** it explains much of the 36 GiB peak.
- 1-image same-seed control (`42 vs 42-repeat`) vs cross-seed (`42 vs 43`) check (`CELL 11` of `notebooks/01_dossr_baseline.ipynb`) is **NOT TESTED**.
- Seed study (`seeds 42..46`) is **not run (NOT TESTED)**.
- No quality metrics. No project-level metric code. It is **not confirmed** that HR reference images exist (TODO: confirm). PSNR, SSIM and LPIPS need HR. Upstream `external/DOSSR/utils/metrics.py` has PSNR and an LPIPS wrapper.
- The visual quality of the outputs was not reviewed by a human yet (TODO: user to look at `Canon_001_LR4_compare.png`).
- Exact input size of the batch images is not confirmed (folder is named `crop128`, but the output was 2048×2048 at x4).
- Text conditioning is **not designed or built**. The method is undecided.
- The number of images in the full RealSR folder is 200 according to the Oct 28 `verify_assets.py` log in `docs/setup_notes.md` (TODO: confirm in `CELL 04`). Time estimate: about 40 s model load + 7 s per image.

## 6. Next tasks, in order

1. **Test the VAE `AttnBlock` SDPA patch on Colab (`notebooks/01_dossr_baseline.ipynb`, `CELL 01`–`CELL 09`, currently `NOT TESTED`):** run the GPU SDPA check (`scripts/check_attention_sdpa.py --device cuda`), then re-run the fp16 profile (`--fp16 --profile-memory`) with a fresh `output_root` and compare to `36.04 GiB`.
2. **Re-check the output (`CELL 10`, `NOT TESTED`):** run `inspect_output.py`, copy the pre-patch output to Drive as reference, and print the informational pixel diff (`max_abs_diff` and `mean_abs_diff` in 8-bit levels).
3. **Run 1-image seed sensitivity & resume sanity check (`CELL 11`, `NOT TESTED`):** verify resume (`hits=1`), run `seed 42` again (`42-repeat`) and `seed 43` into separate `output_root`s with `--fp16`, and confirm `mean_abs_diff(42 vs 43)` passes the gate (`> 0` if `42 vs 42-repeat == 0`, else `>= 3.0 * mean_abs_diff(42 vs 42-repeat)`).
4. **Run the 5-seed study (`CELL 12`–`CELL 13`, `NOT TESTED`):** after steps 1–3 pass and are reviewed, set `RUN_SEED_STUDY = True` and run seeds 42 to 46 on the same 5 images.
5. **Metrics:** after the user confirms HR availability and the metric set, write a script that gives per-image and per-seed values, the mean, the std across seeds, the worst seed, and pairwise LPIPS between seeds. Save a CSV.
6. **Scale up:** test 10 to 20 images first and measure the real time. Then the full RealSR set.
7. **Text conditioning:** propose 2 or 3 designs (use InstructIR as a reference), with cost and risk. Do not build one until the user and mentor choose.
8. **Docs:** keep `docs/` and the weekly report updated. TODO: weekly report file is `report_W01.md` (kept by the user; an untracked `docs/Week01.tex` is also present locally).

## 7. Rules

- Always follow the **Team Notebook & Experimentation Guidelines**: whenever a task requires running code on Google Colab, create or update the `.ipynb` notebook in `notebooks/` rather than giving isolated snippets.
- Before coding or modifying anything, create an `implementation_plan` for the user to review, ask questions if needed, and wait for explicit permission.
- Do not edit anything under `external/`. Use runtime patches or our own files.
- Do not change a working config. Make a new one, and use a **new `output_root` for every run** (the runner keeps a manifest and `--skip_if_exist`).
- fp16 only happens with `--fp16` (or `precision: fp16` in the config). Check that the log has `enabling CUDA float16 autocast`. Check that the log has `using PyTorch scaled_dot_product_attention`.
- Never claim a result you did not see in a log. Say **NOT TESTED** (or **UNTESTED**) for code that was not run. If you did not run it (no GPU here), say so.
- Do not invent numbers, dates, paper details or file contents. Write `TODO: confirm` and ask.
- Keep changes small. Show a diff, the exact Colab instructions/notebook cells, the expected success signal, and what to send back if it fails (last 50 log lines and the full traceback).
- End each task with: commit hash, changed files, and the push command (`git push origin feat/baseline-runner`). The user pushes and pulls by hand.
- Write docs and explanations in **simple English, short sentences**. The user asked for simpler words several times.
- Credit clearly: the model, the method, the checkpoint and `external/DOSSR` are the DoSSR authors'. Our code is the runner, the compatibility launcher, the patches, the checks and the study scripts. Say if AI agents helped write code when asked.

## 8. Open questions for the user (ask, do not guess)

1. Exact GPU model in Colab?
2. Do we have HR images for RealSR? Which metrics does the course or mentor want?
3. How many images are in the full RealSR folder, and what are their sizes?
4. Course deadlines and the plan per stage?
5. Who on the team does what?
6. Has the mentor replied, and what did they advise?

## 9. Quick Colab execution

Open `notebooks/01_dossr_baseline.ipynb` in Google Colab and run `CELL 01` through `CELL 11` in order (`RUN_SEED_STUDY = False` by default). Or from a Colab terminal:

```bash
cd /content/SC203
git fetch origin && git checkout feat/baseline-runner && git pull --ff-only origin feat/baseline-runner

# GPU attention check (CrossAttention + tiny and 1x512x64x64 VAE AttnBlock)
python scripts/check_attention_sdpa.py --device cuda --log-dir /content/drive/MyDrive/lemmepass/logs

# smoke test (set a NEW output_root in the YAML first)
bash scripts/run_baseline.sh --config /content/baseline_smoke_l4_fp16.yaml --fp16

# memory profile
bash scripts/run_baseline.sh --config /content/baseline_smoke_l4_fp16_profile.yaml --fp16 --profile-memory

# save results before the session ends
cp -r /content/outputs /content/drive/MyDrive/lemmepass/outputs_backup
```
