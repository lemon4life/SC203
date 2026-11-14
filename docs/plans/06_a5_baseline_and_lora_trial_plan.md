# [SUPERSEDED] Plan 06 — Group Assignment 5 (`Baseline` + Trial LoRA Run) Implementation & Team Split

> **Status:** SUPERSEDED by `docs/plans/06c_a5_baseline_implementation_plan.md` and `docs/plans/07_member1_gpu_pilot_and_realce_split_plan.md`. Retained solely for historical design audit. Do not execute tasks from this file.

**Date:** `2026-11-10`  
**Team:** Team 1 (`LEMMEPASS`, SC203 Scientific Methods — Class 25A02)  
- **Member 1 (`Hoàng Nguyên Anh`, `25125002`):** GPU & Pipeline Lead (Colab Pro GPU: trial LoRA `C` execution & VRAM profiling, final PDF integration & submission).  
- **Member 2 (`Huỳnh Trần Gia Hân`, `25125042`):** Text Evaluation & Training-Split Text Box Preparation Lead (prepare disjoint training/validation text boxes or `Real-CE x4` subset; run `scripts/text_eval.py` on LoRA trial outputs).  
- **Member 3 (`Hồ Quang Minh`, `25125060`):** LoRA & Text-Region Consistency Loss Lead (implement `src/lora_text_loss.py` and `scripts/run_lora_trial.py` supporting variants `A, B, C` without modifying `external/DOSSR`).  
- **Member 4 (`Nguyễn Phan Minh Nhật`, `25125033`):** Report & Visualization Lead (`docs/A5_baseline.tex`, summary plots/tables from `results/metrics_10seed_*.csv` and `results/text_eval_*.csv`, and reproducibility appendix).

---

## 1. Deadlines & Schedule for Assignment 5
- **Tue `10/11 07:00`:** Official submission of Assignment 4 (`docs/main.tex` / `docs/A4_sota_limitations.tex` — **DONE**).
- **Tue `11/11`:** Member 3 resumes LoRA implementation (`src/lora_text_loss.py`); team confirms disjoint training dataset choice (`Real-CE` / ICDAR-TextOCR + Real-ESRGAN degradation / RealSR train split) and empty prompt `""`.
- **Wed `12/11 21:00`:**
  - Member 3 finishes `src/lora_text_loss.py` + `scripts/run_lora_trial.py` (CPU smoke-tested).
  - Member 2 finishes the small disjoint training/validation text-box CSV (`20–50` text-rich HR crops, strictly disjoint from the 100 `RealSRVal` test images).
- **Thu `13/11 21:00`:**
  - Member 1 runs the **short trial of LoRA Version `C`** (`50–100` steps) on Colab GPU and logs `lora_run_manifest.json` + `lora_train_log.csv` (recording `peak_allocated_gib`, `peak_reserved_gib`, `step_time_s`, and loss curves).
  - Member 2 & Member 4 evaluate the trial checkpoint on `seed_42` (or `Canon_001`–`005`) if time permits.
- **Fri `13/11 22:00`:** Member 4 completes full draft of `docs/A5_baseline.tex`.
- **Mon `16/11 21:00`:** Internal team review, proofread, and PDF submission (Official deadline: **Tue `17/11/2026 07:00`**).

---

## 2. What Is Already Done (From A4) vs. What Is New for A5

### Already 100% Done & Verified (Reusable Directly in A5!)
1. **Standardized Baseline Runner & Compatibility Layer (`src/run_dossr.py`, `src/dossr_inference_compat.py`, `src/attention_compat.py`):**
   - Verified equivalence (`SDPA ON vs OFF`: `mean_abs_diff = 0.0334 px`; `default vs unwrapped`: `0.0000 px`; `seed 42 vs 42-repeat`: `0.0000 px`).
   - Verified hardware profile at `128x128 -> 512x512` (`FP16`, `SDPA OFF`): `7.32 GiB` allocated (`8.28 GiB` reserved), `1.92 s/image`.
2. **Full 100-Image $\times$ 10-Seed $\times$ 2-Prompt Baseline Study (`2,000` `512x512` PNGs on Drive):**
   - Whole-image metrics (`results/metrics_10seed_{default,empty}_summary_by_image.csv`):
     - `Default`: `PSNR_RGB = 22.6353 ± 0.5961 dB`, `SSIM_RGB = 0.6406 ± 0.0182`, `LPIPS = 0.3382 ± 0.0191`, `pairwise_px = 10.4508`, `pairwise_lpips = 0.1727`.
     - `Empty`: `PSNR_RGB = 22.6998 ± 0.6079 dB`, `SSIM_RGB = 0.6431 ± 0.0184`, `LPIPS = 0.3352 ± 0.0191`, `pairwise_px = 10.3415`, `pairwise_lpips = 0.1705`.
3. **Full 100-Image Text Restoration & Seed-Instability Baseline (`results/text_eval_*.csv`):**
   - `25` readable images, `193` boxes, `1,930` box-seed crops (`HR ceiling = 70.5%`, `Bicubic floor = 20.2%`, `DoSSR 10-seed = 24.0%`, `Seed disagreement = 32.83%`).
4. **1-Page Related Work (`docs/related_methods_ccsr_pnfs_tadisr.md`):**
   - `CCSR`, `PNFS`, and `TADiSR` comparison completed.

### New Deliverables Needed for Assignment 5 (`Baseline` Report + Mentor Deliverable 3)
1. **Short Trial Run of LoRA Version `C` on Colab GPU with VRAM & Step Time Logged** (Mentor Deliverable 3).
2. **Complete `docs/A5_baseline.tex` Report** following Section 9 of `docs/ke_hoach_nhom_A4_A5.md`:
   - Section 1: Method & Pipeline Overview (DoSSR domain-shift ODE + our runtime runner/wrappers).
   - Section 2: Implementation Details & Equivalence Checks (`SDPA` numerical check, `default vs unwrapped` check, `42 vs 42-repeat` determinism check).
   - Section 3: Experimental Setup (Data, configs, hardware, commits, seeds `42..51`).
   - Section 4: Baseline Results (10-seed whole-image table `PSNR/SSIM/LPIPS`, runtime/VRAM profile, 10-seed text evaluation table, `eduroam` figure, and **LoRA Version `C` trial run VRAM/loss table**).
   - Section 5: Analysis & Limitations.
   - Section 6: Reproducibility Guide (exact CLI commands, configs, commits, and seeds).

---

## 3. Detailed Work Division by Member for Assignment 5

### Member 3 (`Hồ Quang Minh` — LoRA & Loss Lead, starting Tue `11/11`)
- **Branch:** `feat/lora-loss`
- **Files to create (`NOT TESTED` until Colab run):**
  1. `src/lora_text_loss.py`:
     - Injects lightweight LoRA linear adapters (`rank r = 4` or `8`) into `to_q, to_k, to_v, to_out.0` of `CrossAttention` in `model.control_model` (ControlNet) and `model.model.diffusion_model.output_blocks` (UNet decoder, avoiding the `torch.no_grad()` block in `ControlledUnetModel.forward`).
     - Implements the 2-noise wrapper around `model.predict_x0_out_mix(x, preprocess_x, c, t, noise=eps_i)` at the same timestep $t$ for variants `A`, `B`, and `C`:
       - `A`: $\mathcal{L}_A = \mathcal{L}_{\text{orig}}(\epsilon_1, \epsilon_2)$
       - `B`: $\mathcal{L}_B = \mathcal{L}_{\text{orig}}(\epsilon_1, \epsilon_2) + \lambda_{\text{rec}} \mathcal{L}_{\text{text-rec}}(\epsilon_1, \epsilon_2)$
       - `C`: $\mathcal{L}_C = \mathcal{L}_B + \lambda_{\text{cons}} \mathcal{L}_{\text{text-cons}}(\hat{x}_0^{(1)}, \hat{x}_0^{(2)})$
     - Converts HR text boxes `(x, y, w, h)` to $512 \times 512$ binary mask $M_{\text{HR}}$ and downsamples via `F.max_pool2d(M_HR, kernel_size=8, stride=8)` to $64 \times 64$ latent mask $M_{\text{lat}}$.
  2. `scripts/run_lora_trial.py` & `configs/lora_trial_c.yaml`:
     - Loads paired `(LR, HR, text_boxes)` without `RealESRGANBatchTransform._dequeue_and_enqueue` (so masks stay aligned with images).
     - Runs `N` training steps (e.g., `50–100` steps, `batch_size=1`, `accumulate_grad_batches=4`, `fp16` autocast, `prompt_mode="empty"`).
     - Writes `lora_run_manifest.json` (`trainable_param_count`, `total_param_count`, `lora_rank`, `lambda_rec`, `lambda_cons`) and `lora_train_log.csv` (`step`, `loss_total`, `loss_orig_2noise`, `loss_text_rec`, `loss_text_cons`, `step_time_s`, `peak_allocated_gib`, `peak_reserved_gib`).

### Member 2 (`Huỳnh Trần Gia Hân` — Training Data Boxes & Text Eval Lead)
- **Branch:** `feat/text-eval-a5`
- **Tasks:**
  1. Prepare a **small, strictly disjoint training/validation text-box CSV** (`data/text_boxes_lora_train.csv`, e.g., `20–40` text-rich HR images from the RealSR **train** split or `Real-CE` train split — **zero overlap** with the 100 `RealSRVal` test images).
  2. Validate the CSV with `scripts/check_text_boxes.py`.
  3. Once Member 1 runs the short LoRA `C` trial on Colab and generates sample outputs on `seed_42`, run `scripts/text_eval.py` (`--rec-only --rotate-vertical`) to check if the trial LoRA `C` checkpoint runs cleanly through the evaluation pipeline.

### Member 1 (`Hoàng Nguyên Anh` — GPU & Pipeline Lead)
- **Branch:** `feat/baseline-runner`
- **Tasks:**
  1. Merge Member 3's `src/lora_text_loss.py` + `scripts/run_lora_trial.py` and Member 2's `data/text_boxes_lora_train.csv`.
  2. Add `CELL 19` (LoRA Version `C` short trial run & VRAM profiler) to `notebooks/01_dossr_baseline.ipynb` and execute on Colab GPU.
  3. Commit the resulting `results/lora_trial_c_manifest.json` and `results/lora_trial_c_train_log.csv` so Member 4 has verified VRAM, parameter count, and step-time numbers for `docs/A5_baseline.tex`.
  4. Final review, PDF compilation, and submission of Assignment 5 by Mon `16/11`.

### Member 4 (`Nguyễn Phan Minh Nhật` — Report & Visualization Lead)
- **Branch:** `feat/report-a5`
- **Tasks:**
  1. Write `docs/A5_baseline.tex` (can start immediately on Sections 1, 2, 3, 4.1–4.3, 5, and 6 using the already-verified numbers in `docs/fact_sheet_for_report.md` and `docs/main.tex`!).
  2. Create a distribution plot or per-image variance figure from `results/metrics_10seed_default_summary_by_image.csv` and `results/text_eval_summary_by_image.csv` (e.g., showing whole-image PSNR vs. text-box CER / seed disagreement across the 25 text-positive images).
  3. Insert the verified LoRA Version `C` trial VRAM & training speed table into Section 4.4 of `docs/A5_baseline.tex` once Member 1 commits `results/lora_trial_c_train_log.csv` on Thu `13/11`.
