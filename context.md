# context.md: read this first

You are joining Team 1 (LEMMEPASS) in SC203 Scientific Methods (class 25A02). This file tells you what the project is, what is done, what is not, and what to do next. Facts below come from our real logs (Oct 29 & Nov 5, 2026) and direct source inspection of `external/DOSSR`. Anything not confirmed is marked **TODO: confirm**. Do not guess it. Ask the user.

## 1. Project & New Mentor Direction (Nov 2026)

- **Topic 3:** seed-robust real-world super-resolution with text-region consistency.
- **Base model:** DoSSR (public code and pretrained checkpoint).
- **Updated Research Question (Nov 2026 Mentor Direction):**
  - *"Does adding a loss during LoRA training help DoSSR restore text correctly more consistently when the seed changes?"*
  - The baseline is tentatively accepted. The instruction-conditioning part (InstructIR / prompt injection) is **deferred** until the text-region consistency loss is validated. **Do not work on prompts/InstructIR/text injection now.**
- **Mentor's 5 Core Requests:**
  1. **Fixed 10 Seeds & Prompt Comparison:** Use a fixed list of 10 seeds (`configs/seeds_10.yaml`, default `42..51`), committed before running, for every RealSR image (`RealSRVal_crop128/test_LR`, `128x128 -> 512x512`). Compare `default` prompt (`"clean, high-resolution, 8k"`) vs `empty` prompt (`""`) on the exact same `(image, seed)` pairs, 5 steps, FP16, **SDPA OFF (`--disable-sdpa`)**. For `Canon_001_LR4` (`eduroam`), make a side-by-side figure with `LR`, `HR`, and the two outputs with the same text region zoomed.
  2. **Text-Region Consistency Loss for LoRA Training:** For an LR-HR pair and the same timestep $t$, draw two independent noise samples $\epsilon_1, \epsilon_2$. Use a mask from the HR text boxes and penalize the difference between the two clean-latent predictions ($\hat{x}_0^{(1)}$ and $\hat{x}_0^{(2)}$) inside the text region. Keep the original DoSSR loss.
  3. **Three Fair-Comparison LoRA Versions `(A, B, C)`:** Same data, prompt, number of noise samples (2 noise samples per step for all three so `C` gets no extra signal), and training budget:
     - `(A)` Original loss (averaged over 2 noise samples).
     - `(B)` Higher text-region weight (averaged over 2 noise samples).
     - `(C)` `(B)` plus the cross-noise text-region consistency loss. (`B` vs `C` isolates what the new loss adds.)
  4. **Evaluation Metrics:** Main metric is the share of text-region restorations with at least one wrong character vs the HR transcription. Also compute Character Error Rate (CER, edit distance), disagreement between seeds, and whole-image SSIM/LPIPS. Treat the **IMAGE** as the statistical unit (the 10 seeds of one image are not independent) and pair comparisons on `(image, seed)` pairs. If RealSR has too few readable text regions, use `Real-CE x4` for text evaluation.
  5. **Next Report Deliverables:** (1) 10-seed baseline table, (2) one-page comparison with **CCSR**, **PNFS**, and **TADiSR**, and (3) a trial LoRA run on Colab with VRAM recorded.

## 2. Team Split (4 Members), Branches & PR Rule

- **Names:** `TODO: confirm` mapping of `Hoàng Nguyên Anh (25125002)`, `Nguyễn Phan Minh Nhật (25125033)`, `Huỳnh Trần Gia Hân (25125042)`, `Hồ Quang Minh (25125060)` to Members 1–4.
- **Merge & Status Rule:** **Members 2, 3, and 4 must NOT edit `context.md` directly.** Members report progress in the team chat, and Member 1 (via the agent) updates `context.md`. **Member 1 merges all teammate branches (`feat/text-eval`, `feat/lora-loss`, `feat/report-figures`) into `feat/baseline-runner` via Pull Requests (PRs).**

| Member | Role | Branch | Owned Files | Hardware |
|---|---|---|---|---|
| **Member 1 (User)** | GPU & Pipeline Lead | `feat/baseline-runner` | `src/run_dossr.py`, `src/dossr_inference_compat.py`, `src/attention_compat.py`, `scripts/run_seed_study.py`, `configs/seeds_10.yaml`, `configs/*_10seed_*.yaml`, `notebooks/01_dossr_baseline.ipynb`, `context.md` | Colab Pro GPU (runs on Apple Silicon Mac locally without GPU) |
| **Member 2** | Text Evaluation Lead | `feat/text-eval` | `scripts/make_text_contact_sheets.py`, `scripts/text_eval.py`, `data/text_boxes_realsr_template.csv`, `data/text_boxes_realsr.csv`, `docs/handoff_member2.md` | CPU laptop (Windows/Mac) or Free Colab CPU |
| **Member 3** | LoRA & Loss Lead | `feat/lora-loss` | `docs/lora_feasibility_and_loss_design.md`, `docs/handoff_member3.md`, future `src/lora_text_loss.py`, `configs/lora_*.yaml` | CPU laptop for design; Member 1 runs GPU trial on Colab Pro |
| **Member 4** | Report, Metrics & Figures Lead | `feat/report-figures` | `scripts/compute_metrics.py`, `scripts/plot_eduroam_figure.py`, `docs/related_methods_ccsr_pnfs_tadisr.md`, `docs/handoff_member4.md` | CPU laptop (Windows/Mac) or Free Colab CPU |

## 3. Task Status & Dependency Table

> **Rule:** Only Member 1 / the agent updates this table. Everything stays `NOT TESTED` until Colab or teammate execution logs are sent back.

| Task | Description | Owner | Status | Blocked By |
|---|---|---|---|---|
| **B.5** | Contact sheet generator (`scripts/make_text_contact_sheets.py`, with `--check` mode and blank default `readable`) & CSV template (`data/text_boxes_realsr_template.csv`) | Member 2 | `NOT TESTED` (Script & template ready) | None — ready for Member 2 today |
| **B.5-Label** | Run `make_text_contact_sheets.py` on `test_HR`, fill `data/text_boxes_realsr.csv`, run `--check`, check if `Real-CE x4` is needed | Member 2 | `NOT TESTED` | B.5 |
| **C.9** | Read-only audit of DoSSR training entry point (`scripts/train.py`, `configs/train_cldm.yaml`), training prompt (`""` in `realesrgan.py:179`), loss & `pred_x0` (`model/cldm.py`, `ddpm.py`), and queue/no_grad caveats (`docs/lora_feasibility_and_loss_design.md`) | Member 3 | **Done (`VERIFIED` in `external/DOSSR` source)** | None — ready for Member 3 today |
| **C.10** | LoRA trial parameter, batch/patch size, VRAM & logging estimates, training data options, and Phase 1 target (`short trial run of version C with VRAM logged`, `docs/lora_feasibility_and_loss_design.md`) | Member 3 | **`ESTIMATE — NOT TESTED`** (Documented in `docs/lora_feasibility_and_loss_design.md`) | None — ready for Member 3 today |
| **Handoffs** | `docs/handoff_member2.md`, `docs/handoff_member3.md`, `docs/handoff_member4.md` | Member 1 | **Done** | None — ready to hand off today |
| **A.1** | Create `configs/seeds_10.yaml` (`42..51`) | Member 1 | `NOT TESTED` (Created & committed before any 10-seed run) | None |
| **A.2** | Audit `external/DOSSR` inference vs training prompt (`"clean, high-resolution, 8k"` vs `""`) | Member 1 | **Done (`VERIFIED` in `external/DOSSR` source)** | None |
| **A.3** | Add `--prompt-mode {default,empty,unwrapped}`, log prompt & SDPA mode in `run_manifest.json`, and stop if `precision != "fp16"` | Member 1 | **Done (`VERIFIED` in `CELL 16`, Nov 7, 2026)** | None |
| **A.4** | Prepare 100-image $\times$ 10-seed $\times$ 2-prompt configs (`SDPA OFF`) & `CELL 16`–`CELL 17` in `notebooks/01_dossr_baseline.ipynb` | Member 1 | **`CELL 16` VERIFIED (`default == unwrapped` diff `0.0`; `empty != default` max diff `50.0`, mean `0.3704`); `CELL 17` prepared (`NOT TESTED`)** | None |
| **A.4-Run** | Run `CELL 17` (100-image $\times$ 10-seed $\times$ 2-prompt study) on Colab Pro | Member 1 | `NOT TESTED` | None — ready to run when Member 1 enables `RUN_100IMG_10SEED_PROMPT_STUDY = True` |
| **B.6** | Choose OCR option & write `scripts/text_eval.py` (using exact text rules in `docs/handoff_member2.md`, `IMAGE` as statistical unit) | Member 2 | `NOT TESTED` (`scripts/text_eval.py` created; awaiting OCR choice confirmation & `B.5-Label`) | OCR choice; final run blocked by B.5-Label & A.4-Run |
| **B.7** | Whole-image metrics (`scripts/compute_metrics.py`: PSNR, SSIM, LPIPS) | Member 4 | **`VERIFIED` on 5-seed study in `CELL 14`–`15` (`810e844`, `metrics_per_seed.csv` & `metrics_summary_by_image.csv` packaged in `sc203_baseline_results_bundle.zip`); 10-seed run `NOT TESTED`** | Final 10-seed table blocked by A.4-Run |
| **B.8** | Write `scripts/plot_eduroam_figure.py` (`Canon_001_LR4` side-by-side zoomed figure) & `CELL 18` | Member 4 | `NOT TESTED` (`scripts/plot_eduroam_figure.py` and `CELL 18` created; `CELL 16` outputs ready on Drive) | Exact crop box `(x,y,w,h)` from User/Member 2 |
| **C-Trial** | Implement & run short Phase 1 trial of LoRA version `C` on Colab Pro with VRAM logged (and full `A, B, C` comparison) | Member 3 & Member 1 | `NOT TESTED` | Training data decision + Member 3 plan + B.5-Label |
| **D-Report** | 10-seed baseline table, 1-page CCSR/PNFS/TADiSR comparison (`docs/related_methods_ccsr_pnfs_tadisr.md`), and trial LoRA `C` VRAM report | Member 4 & All | `NOT TESTED` (Template `docs/related_methods_ccsr_pnfs_tadisr.md` created) | A.4-Run, B.6, B.7, B.8, C-Trial |

## 4. Repo & Known Files

```bash
git clone https://github.com/lemon4life/SC203
cd SC203
git checkout feat/baseline-runner   # main integration branch managed by Member 1
```

- `external/DOSSR/` is the upstream DoSSR code. **Never edit it.** It is ignored in `.gitignore`; `scripts/colab_setup.py` clones `https://github.com/AMD-AIG-AIMA/DOSSR.git` automatically when `external/DOSSR` is absent.

| Path | What it is |
|---|---|
| `notebooks/01_dossr_baseline.ipynb` | Standardized team Colab notebook: **`CELL 01`–`CELL 13` are VERIFIED** (`251d4a5`). **`CELL 14` (`compute_metrics.py`), `CELL 15` (results zip), `CELL 16` (1-image prompt wrapper check: `unwrapped` vs `default` vs `empty`, SDPA OFF), and `CELL 17` (prepared `100 test_LR × 10 seeds × 2 prompts` runner, SDPA OFF) are NOT TESTED.** |
| `src/run_dossr.py` | Runner. Reads YAML config, refuses `input_dir` pointing to `test_HR`, supports `--disable-sdpa`, `--disable-vae-sdpa`, and `--prompt-mode {default,empty,unwrapped}`, writes `run_manifest.json`, and **stops with an error if `run_manifest.json` does not say `precision: "fp16"`**. |
| `src/dossr_inference_compat.py` | Launcher around upstream `inference.py`. Enables fp16 autocast, mmap checkpoint load, SDPA patches (unless disabled), `--prompt-mode {default,empty,unwrapped}` (printing exact prompt strings), and `--profile-memory`. |
| `src/attention_compat.py` | Runtime SDPA patches for `CrossAttention` and VAE `AttnBlock`. |
| `scripts/run_seed_study.py` | Multi-seed runner supporting `seeds_file: configs/seeds_10.yaml`, input (`128x128`) and output (`512x512`) size checks, and `run_manifest.json` `precision == "fp16"` verification. |
| `configs/seeds_10.yaml` | Fixed list of 10 seeds (`42..51`), committed before running any 10-seed study. |
| `configs/baseline_realsr_100img_lr128_{default,empty}_prompt_sdpa_off.yaml`, `configs/seed_study_realsr_100img_10seed_{default,empty}_prompt_sdpa_off.yaml` | Prepared 100-image $\times$ 10-seed $\times$ 2-prompt configs (`SDPA OFF`, `precision: fp16`, `steps: 5`, new `output_root`s). **NOT TESTED.** |
| `scripts/compute_metrics.py` | Computes PSNR, SSIM, and LPIPS (`<stem>_0.png` vs `test_HR/<stem>.png`) and across-seed spread (`--device auto|cpu|cuda`). **NOT TESTED.** |
| `scripts/make_text_contact_sheets.py` | CPU-only script (`Pillow` + `pathlib`) that builds numbered contact sheets of `test_HR`, generates the CSV with `readable` left blank (`""` = not yet labeled), and provides `--check` to fail if any blank `readable` rows remain. **NOT TESTED.** |
| `scripts/text_eval.py` | Crops each `readable=yes` box from `<stem>_0.png`, runs OCR (`--ocr-backend {rapidocr,easyocr,tesseract}` kept in one place), applies the `docs/handoff_member2.md` text normalization rules, computes `>=1 wrong char`, `CER`, and seed disagreement with **IMAGE** as the statistical unit, and supports paired `(image, seed)` comparison (`--compare-study-root`). **NOT TESTED.** |
| `scripts/plot_eduroam_figure.py` | CPU-only script (`Pillow` + `pathlib`) that builds the 4-panel `Canon_001_LR4` (`eduroam`) comparison (`LR`, `HR`, `Default Prompt`, `Empty Prompt`) with the same zoomed text region (`--crop x,y,w,h` or `--boxes-csv`). **NOT TESTED.** |
| `data/text_boxes_realsr_template.csv` | Canonical CSV header (`image,box_id,x,y,w,h,transcript,readable`) shared across Members 2, 3, and 4. |
| `docs/lora_feasibility_and_loss_design.md` | Read-only audit of `external/DOSSR` training code, prompts, `pred_x0` equations, caveats, training data options, and LoRA trial estimates (`ESTIMATE — NOT TESTED`). |
| `docs/related_methods_ccsr_pnfs_tadisr.md` | One-page comparison template for Member 4 comparing DoSSR + our text-region consistency loss against **CCSR**, **PNFS**, and **TADiSR** (`TODO: confirm` markers for paper specifics). |
| `docs/handoff_member2.md`, `docs/handoff_member3.md`, `docs/handoff_member4.md` | Simple-English handoff instructions for Members 2, 3, and 4. |
| `docs/after_01_dossr_baseline_discovery.md`, `docs/plans/` | Discovery report (`2026-11-05`) and numbered implementation plans archive. |

## 5. Environment & Verified Findings So Far

1. **Colab GPU (`CELL 03`, Nov 5, 2026):** `NVIDIA A100-SXM4-80GB` (`79.25 GiB`), torch `2.11.0+cu130`, Python `3.13.16`.
2. **RealSR Dataset (`CELL 04`, `251d4a5`):**
   - `RealSRVal_crop128/test_LR`: `100` images, `128x128`.
   - `RealSRVal_crop128/test_HR`: `100` images, `512x512`.
   - Old `2048x2048` folders on Drive are quarantined with `README_DO_NOT_USE_AS_BASELINE.txt`.
3. **1-Image `test_LR` (`128x128 -> 512x512`) FP16 Profile (`CELL 08`–`CELL 09`, `251d4a5`):**
   - **SDPA ON:** `peak_allocated = 7.32 GiB`, `peak_reserved = 8.29 GiB`, `sampler = 1.24 s`, `image = 1.75 s`.
   - **SDPA OFF (`--disable-sdpa`):** `peak_allocated = 7.32 GiB`, `peak_reserved = 8.28 GiB`, `sampler = 1.42 s`, `image = 1.92 s`.
   - **Pixel diff (SDPA ON vs OFF, `0..255`):** `max_abs_diff = 2.0000`, `mean_abs_diff = 0.0334`.
4. **Seed Sensitivity & 5-Image / 5-Seed Study (`CELL 11`–`CELL 13`, `251d4a5`):**
   - `42 vs 42-repeat`: `0.0` diff. `42 vs 43`: `mean_abs_diff = 8.8237`.
   - All 25 outputs (`5 images x 5 seeds 42..46`) verified at `512x512`. On `Canon_001_LR4`, DoSSR restored the `"eduroam"` sign as `"odaroam"`.
5. **Verified Source Findings in `external/DOSSR` (Tasks A.2 & C.9):**
   - **Inference prompt:** `scripts/inference.py:75-78` calls `shift_sample_3_order(..., cfg_scale=1.0)` which uses `positive_prompt="clean, high-resolution, 8k"` (`model/shift_sampler.py:200, 257, 322`).
   - **Training prompt:** `dataset/realesrgan.py:179` uses `"txt": ""` (empty prompt).
   - **Training entry point & loss:** `scripts/train.py:10-30` and `configs/train_cldm.yaml` run `ControlLDM.shared_step` (`model/cldm.py:346-356`), which calls `predict_x0_out_mix(x, preprocess_x, c, t, noise=...)` (`ldm/models/diffusion/ddpm.py:909-917`) and minimizes `(x - pred_x0).abs().mean()`.
6. **Verified `CELL 16` Prompt Wrapper Check (`Canon_001_LR4.png`, `seed 42`, `SDPA OFF`, `FP16`, Nov 7, 2026):**
   - `unwrapped` (`wrapper_installed=False`, `positive_prompt='clean, high-resolution, 8k'`): `Execution time: 2.02 s`.
   - `default` (`wrapper_installed=True`, `positive_prompt='clean, high-resolution, 8k'`): `Execution time: 2.01 s`.
   - `empty` (`wrapper_installed=True`, `positive_prompt=''`): `Execution time: 2.01 s`.
   - **`default` vs `unwrapped` (8-bit `0..255`):** `max_abs_diff = 0.0000`, `mean_abs_diff = 0.0000` (exact match).
   - **`empty` vs `default` (8-bit `0..255`):** `max_abs_diff = 50.0000`, `mean_abs_diff = 0.3704` (verified different picture).
   - All three `run_manifest.json` files verified with `"precision": "fp16"`, `"sdpa_enabled": false`, `"disable_sdpa": true`.

## 6. Rules

- Follow the **Team Notebook & Experimentation Guidelines**: keep Colab runs in `notebooks/`.
- Before coding, create/update a numbered plan (`<NN>_<brief_context>.md`), get approval, and archive in `docs/plans/`.
- **Never edit `external/`.** Use runtime wrappers or our own scripts.
- Use a **new `output_root` for every run**.
- The runner must stop if `run_manifest.json` does not record `precision: "fp16"` for FP16 baseline runs.
- Mark all unexecuted code and engineering estimates as **`NOT TESTED`** (or **`ESTIMATE — NOT TESTED`**) until Colab logs arrive.
- Do not invent facts about **CCSR**, **PNFS**, **TADiSR**, or **Real-CE**. Mark `TODO: confirm` and ask.
- Write docs in **simple English, short sentences**.
- End each task group with: commit hash, changed files, and `git push origin feat/baseline-runner`.

## 7. Open Questions (`TODO: confirm`)

1. **Team Member Names:** Which teammate (`Hoàng Nguyên Anh`, `Nguyễn Phan Minh Nhật`, `Huỳnh Trần Gia Hân`, `Hồ Quang Minh`) is Member 1, Member 2, Member 3, and Member 4?
2. **Next Report Deadline:** What is the exact date/deadline for the next mentor report?
3. **Readable Text Region Count in RealSR (`B.5-Label`):** How many `readable=yes` text boxes does Member 2 find in `RealSRVal_crop128/test_HR`, and what is the exact `(x, y, w, h)` box for `eduroam` in `Canon_001_LR4.png`?
4. **`Real-CE x4` Access:** If RealSR has too few readable text boxes, do we already have `Real-CE x4` on Drive or need to download it?
5. **OCR Choice (`B.6`):** Which OCR option (EasyOCR, RapidOCR/ONNXRuntime, or Tesseract) does the team select for `scripts/text_eval.py`?
6. **LoRA Training Data (`C.9/C.10`):** Which disjoint HR dataset with text boxes (Option 1: `Real-CE`, Option 2: scene-text HR dataset like ICDAR/TextOCR/HierText + Real-ESRGAN degradation, or Option 3: disjoint RealSR train split) should we use for train/val?
7. **LoRA Prompt Mode (`C.9/C.10`):** Confirm the proposal to train LoRA `A`, `B`, and `C` with the empty prompt `""` (matching `realesrgan.py:179`) and evaluate with the same empty-prompt mode.
