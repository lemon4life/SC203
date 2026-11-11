# Team Plan, Tasks, Deadlines & Dependencies — Assignment 5 (`Baseline`) & Parallel LoRA Trial

**Course:** SC203 Scientific Methods (Class 25A02) — Team 1 (`LEMMEPASS`)  
**Repository & Branch:** `https://github.com/lemon4life/SC203` (`feat/baseline-runner`)  
**Required Submission Format:** One PDF named `<StudentID1>_<StudentID2>_<StudentID3>_<StudentID4>.pdf`  
**Final Submission Filename:** `25125002_25125042_25125060_25125033.pdf`  
(Nguyễn Anh `25125002`, Gia Hân `25125042`, Quang Minh `25125060`, Minh Nhật `25125033`)  
**Date Updated:** Wednesday night, `11/11/2026`  

---

## 1. Two Independent Tracks (Strict Decoupling Rule)

1. **Track 1 — Group Assignment 5 (`A5` = Baseline Report, Official Due: Tue `17/11/2026 07:00`):**
   - **Internal Full Draft Deadline:** **Friday `13/11/2026 22:00`**
   - **Internal Final Submit Deadline:** **Monday `16/11/2026 21:00`**
   - **Scope:** Complete, reproducible DoSSR baseline report (`docs/A5_baseline.tex`) covering:
     1. Method & runtime compatibility pipeline (`SDPA`, prompt wrapper, `FP16` manifest checks).
     2. Implementation & equivalence verification (`SDPA ON vs OFF`, `default vs unwrapped`, `seed 42 vs 42-repeat`).
     3. 10-seed $\times$ 2-prompt whole-image metrics (`PSNR`, `SSIM`, `LPIPS`, cross-seed spread, plus `Bicubic x4` reference row).
     4. 10-seed text evaluation strengthened with the **A4-review upgrades** (unified single-OCR run, standalone `empty-prompt` row, OCR-solvable subset, image-level bootstrap 95% CIs, capped/median CER, and matched-size non-text control boxes for pixel seed spread).
   - **Decoupling Rule:** **`A5` does NOT depend on the LoRA trial.** `A5` is 100% self-contained. Mention LoRA in `A5` only as a short 1-paragraph note if the trial is already completed before the `A5` freeze.

2. **Track 2 — Parallel LoRA Trial & `Real-CE` (`4x`) Feasibility (For Next Mentor Report):**
   - **Scope:** `Real-CE` (`4x`) feasibility check first $\rightarrow$ CPU unit tests `(a)`–`(f)` on `src/lora_text_loss.py` $\rightarrow$ Colab smoke run of `LoRA A` (~20 steps) $\rightarrow$ Colab trial run of `LoRA C` (50–100 steps) logging VRAM & step time $\rightarrow$ 3-image inference sanity check (`no NaN`).

---

## 2. Master Schedule & Milestones (`Thu 12/11` – `Tue 17/11/2026`)

| Day & Date | Time | Milestone / Deliverable | Owner(s) | Track |
| :--- | :---: | :--- | :---: | :---: |
| **Thu `12/11`** | `21:00` | **Task M3-1:** Add OCR manifest logger to `scripts/text_eval.py`.<br>**Task M2-1:** Run unified single-OCR 10-seed `text_eval.py` across all `193` boxes (`default` and `empty` prompts) with manifest.<br>**Task M4-1:** Draft `docs/A5_baseline.tex` Sections 1, 2, 3 (exact `PSNR/SSIM/LPIPS` metric definitions) & generate `Bicubic x4` whole-image baseline row.<br>**Task M1-1:** Inspect `Real-CE` (`4x`) dataset feasibility.<br>**Task M3-2:** Run CPU unit tests `(a)`–`(f)` in `src/lora_text_loss.py`. | **M3** (`Quang Minh`)<br>**M2** (`Gia Hân`)<br>**M4** (`Minh Nhật`)<br>**M1** (`Nguyên Anh`) | Track 1 & 2 |
| **Fri `13/11`** | `18:00` | **Task M2-2:** Run `scripts/analyze_text_eval_a5.py` to produce: (1) OCR-solvable subset (`136` boxes), (2) image-level bootstrap 95% CIs, (3) capped CER $\min(\text{CER}, 1.0)$ & median CER, (4) matched non-text control boxes for pixel seed spread.<br>**Task M3-3 & M1-2:** Run Colab GPU smoke test of `LoRA A` (~20 steps) and trial of `LoRA C` (50–100 steps) with VRAM & step time logged. | **M2** (`Gia Hân`)<br>**M3** (`Quang Minh`) & **M1** (`Nguyên Anh`) | Track 1 & 2 |
| **Fri `13/11`** | **`22:00`** | **INTERNAL DRAFT DEADLINE:** Complete full draft of `docs/A5_baseline.tex` (Sections 1–6) with all `VERIFIED` baseline + A4-review tables and figures. | **M4** (`Minh Nhật`, Lead) + All | **Track 1 (`A5`)** |
| **Sat `14/11`** | `21:00` | **Number-Audit Check:** Every member checks their section's numbers in `docs/A5_baseline.tex` against the Number-Audit Checklist (`docs/fact_sheet_for_report.md`). | All 4 Members | **Track 1 (`A5`)** |
| **Sun `15/11`** | `21:00` | Polish figures, captions, layout, and Section 6 (Reproducibility CLI commands); compile clean PDF on Overleaf. | **M4** (`Minh Nhật`) & **M1** (`Nguyên Anh`) | **Track 1 (`A5`)** |
| **Mon `16/11`** | **`21:00`** | **INTERNAL SUBMIT DEADLINE:** Final proofread, export `25125002_25125042_25125060_25125033.pdf`, submit Assignment 5, and take confirmation screenshot. | **M1** (`Nguyên Anh`) & All | **Track 1 (`A5`)** |
| **Tue `17/11`** | **`07:00`** | **OFFICIAL COURSE DEADLINE: Group Assignment 5 (`Baseline`).** | All | **Track 1 (`A5`)** |

---

## 3. Dependency Graph (Who Waits for Whom)

```text
[Track 1: Assignment 5 Baseline Report — Zero Dependency on LoRA]
  Already on Repo/Drive (2,000 PNGs + data/text_boxes_realsr.csv)
       │
       ├──► M3 (Thu 12/11 12:00): OCR manifest logger in scripts/text_eval.py
       │         │
       │         └──► M2 (Thu 12/11 21:00): Unified 10-seed text_eval (default + empty rows, 1 OCR version)
       │                   │
       │                   └──► M2 (Fri 13/11 18:00): A4-review stats (OCR-solvable subset, bootstrap CIs,
       │                                              capped/median CER, non-text control boxes)
       │                             │
       ├──► M4 (Thu 12/11 21:00): Sections 1–3 + Metric Definitions + Bicubic x4 whole-image row
       │                             │
       └─────────────────────────────┴──► M4 (Fri 13/11 22:00): Full Draft of docs/A5_baseline.tex
                                                   │
                                                   └──► All (Sat 14/11): Number-Audit Check
                                                             │
                                                             └──► M1 (Mon 16/11 21:00): Final PDF Submit
                                                                  (25125002_25125042_25125060_25125033.pdf)

[Track 2: Parallel LoRA Trial for Mentor Report — Does NOT Block A5]
  M1 (Thu 12/11): Real-CE (4x) Feasibility Check ──────┐
  M3 (Thu 12/11): CPU Unit Tests (a)–(f) on LoRA code ──┼──► M1 & M3 (Fri 13/11): Colab Smoke A (20 steps)
                                                        │    & Trial C (50–100 steps, VRAM logged)
                                                        └──► 3-Image Inference Sanity Check (No NaN)
```

---

## 4. Detailed Tasks, Deliverables & Deadlines for Each Member

### Member 1: Hoàng Nguyên Anh (`25125002`) — Team Lead, GPU, Integration & Submission Lead
- **Branch:** `feat/baseline-runner`
- **What You Do (Immediate / Thu `12/11`):**
  1. **Task `M1-0` (Thu `12/11` Morning):** Push the scaffolded `A5` analysis scripts, `docs/A5_baseline.tex` template, and LoRA scaffold (`src/lora_text_loss.py`, `scripts/run_lora_trial.py`) so M2, M3, and M4 can pull immediately.
  2. **Task `M1-1` (By Thu `12/11 21:00`):** Run the **`Real-CE` (`4x`) Feasibility Check** (`scripts/check_realce_feasibility.py`) on Colab/Drive:
     - Check split counts (`train`/`val`/`test`), `LR`/`HR` pairing and image dimensions, annotation format (word vs. line boxes), license, HR OCR accuracy, and run **3 `Real-CE` pairs through DoSSR inference** (`seed 42`, `FP16`, `SDPA OFF`), logging VRAM and time.
     - *Per mentor directive:* If RealSR has too few readable text regions, use `Real-CE x4` to evaluate and train the text portion.
- **What You Do (Fri `13/11` – Mon `16/11`):**
  3. **Task `M1-2` (By Fri `13/11 18:00`, pairs with `M3-2`):** Run the staged Colab GPU LoRA trial:
     - Stage 1: Smoke run `LoRA A` (~20 steps).
     - Stage 2: Trial run `LoRA C` (50–100 steps), saving `lora_run_manifest.json` and `lora_train_log.csv` (GPU model, trainable params, `step_time_s`, `peak_allocated_gib`, `peak_reserved_gib`, loss components) and checkpoints to Drive.
     - Stage 3: Load `LoRA C` checkpoint at inference on 3 validation images (`seed 42`) and verify no `NaN`/black images.
  4. **Task `M1-3` (By Mon `16/11 21:00`, waits for `M4-2`):** Coordinate the team Number-Audit Checklist on `docs/A5_baseline.tex`, compile final PDF `25125002_25125042_25125060_25125033.pdf`, submit to portal, and send the progress report to the mentor.

---

### Member 2: Huỳnh Trần Gia Hân (`25125042`) — Text Evaluation & Statistical Refinement Lead
- **Branch:** `feat/text-eval-a5` (or run on `feat/baseline-runner` and send CSVs)
- **Primary Metric Definition (From Mentor):**
  - **Primary Metric:** Rate of text box recoveries with at least one character error relative to the HR transcript ($\text{Box Error Rate} = 1 - \text{WCR}$). A recovery is marked incorrect if $\ge 1$ character differs from the ground-truth text.
  - **Secondary Metrics:** CER (Character Error Rate), cross-seed disagreement rate ($32.83\%$), and whole-image SSIM / LPIPS.
- **What You Do Right Now (Zero Waiting — All `2,000` PNGs & `data/text_boxes_realsr.csv` Are Ready!):**
  1. **Task `M2-1` (By Thu `12/11 21:00` — Unified Single-OCR 10-Seed Run + Standalone Empty-Prompt Row):**
     - Pull `feat/baseline-runner` (`git pull origin feat/baseline-runner`).
     - Run `scripts/text_eval.py` **once** on the full 100-image `data/text_boxes_realsr.csv` (`193` boxes) so both Canon (`1–50`) and Nikon (`51–100`) are scored with the **exact same RapidOCR version and flags (`--rec-only --rotate-vertical --upscale 1`, and a second run with `--study-root <EMPTY_10SEED_ROOT>` so we have the full standalone `Empty-Prompt` row)**:
       ```bash
       # 1) Full 100-image (193-box) unified run: Default prompt as primary + Empty prompt paired + HR ceiling + Bicubic floor
       python scripts/text_eval.py \
         --study-root "/content/drive/MyDrive/lemmepass/outputs/seed-study-realsr-100img-10seed-default-prompt-sdpa-off-fp16" \
         --compare-study-root "/content/drive/MyDrive/lemmepass/outputs/seed-study-realsr-100img-10seed-empty-prompt-sdpa-off-fp16" \
         --hr-dir "/content/drive/MyDrive/lemmepass/data/StableSR_testsets/StableSR_testsets/RealSRVal_crop128/test_HR" \
         --lr-dir "/content/drive/MyDrive/lemmepass/data/StableSR_testsets/StableSR_testsets/RealSRVal_crop128/test_LR" \
         --boxes-csv "data/text_boxes_realsr.csv" \
         --ocr-backend rapidocr \
         --rec-only --rotate-vertical --upscale 1 \
         --output-dir "results/text_eval_unified_default"

       # 2) Standalone 10-seed Empty-Prompt summary (for the Empty-Prompt row in A5 Table 2)
       python scripts/text_eval.py \
         --study-root "/content/drive/MyDrive/lemmepass/outputs/seed-study-realsr-100img-10seed-empty-prompt-sdpa-off-fp16" \
         --boxes-csv "data/text_boxes_realsr.csv" \
         --ocr-backend rapidocr \
         --rec-only --rotate-vertical --upscale 1 \
         --output-dir "results/text_eval_unified_empty"
       ```
     - Also print your exact RapidOCR version: `python -c "import rapidocr_onnxruntime as r; print(r.__version__)"`.
  2. **Task `M2-2` (By Fri `13/11 18:00` — Statistical Tables & `eduroam` Figure):**
     - Run `scripts/analyze_text_eval_a5.py` to produce:
       - **Table A:** Full `193`-box vs. **OCR-Solvable Subset** (`136` boxes where `HR_CEILING` had zero errors).
       - **Table B:** **Image-level 95% bootstrap confidence intervals** ($N = 25$ images, $10{,}000$ resamples) for Box Error Rate ($1-\text{WCR}$), CER, **capped CER $\min(\text{CER}, 1.0)$**, **median (IQR) CER**, and seed disagreement.
       - **Table C:** **Matched-size non-text control boxes** (comparing pixel seed spread $L_1$ inside text boxes vs. non-text boxes in the same 25 images).
     - **Export `eduroam` Comparison Figure:** Run `scripts/plot_eduroam_figure.py` to create the 4-panel visual comparison (LR, HR, Default prompt, Empty prompt) with zoomed-in text crop showing character-level degradation across seeds.
     - **Note on Training Data:** Do **NOT** hand-label RealSR training images! Wait for the `Real-CE` (`4x`) feasibility check first.
- **What to Send Back to Member 1 & Member 4:**
  - Commit/send `results/text_eval_unified_default/*.csv`, `results/text_eval_unified_empty/*.csv`, the statistical tables, and the `eduroam` PNG by **Fri `13/11 18:00`**.

---

### Member 3: Hồ Quang Minh (`25125060`) — LoRA & Text-Region Consistency Loss Lead
- **Branch:** `feat/lora-loss`
- **Research Question (From Mentor):**
  > *"Does adding a loss during LoRA training help DoSSR recover correct text more consistently across seeds?"*
- **Experimental Protocol to Isolate New Loss (3 LoRA Variants):**
  - **Variant 1 (`LoRA A`):** Original DoSSR loss (drawing 2 noise samples per step for fair compute comparison).
  - **Variant 2 (`LoRA B`):** Increased text-region loss weight (base DoSSR loss with extra weight on HR text masks).
  - **Variant 3 (`LoRA C`):** Same text-region weight + consistency loss penalizing discrepancies between two predicted clean latents $\hat{x}_0^{(1)}, \hat{x}_0^{(2)}$ inside HR text masks.
  - *Gating Condition:* Proceed to the full test run only if `LoRA C` reduces text errors on the validation set compared to `LoRA B`.
  - *Instruction Tuning:* Postponed until after the loss formulation is validated.
- **What You Can Do Right Now (Zero Waiting — Runs 100% on CPU Laptop!):**
  1. **Task `M3-1` (By Thu `12/11 21:00` — Verify & Tune `src/lora_text_loss.py` + Run CPU Unit Tests `(a)`–`(f)`):**
     - Read `docs/lora_code_walkthrough.md` and inspect `src/lora_text_loss.py` and `scripts/run_lora_trial.py`.
     - Run the built-in 6 CPU unit tests (`python src/lora_text_loss.py --run-cpu-tests`):
       - `(a)` Mask downsampling (`512x512 -> 64x64` via `F.max_pool2d(..., 8, 8)`) and mask/image alignment after crop and horizontal flip.
       - `(b)` LoRA output equals base output at initialization (`lora_B = 0`, `max_abs_diff == 0.0`).
       - `(c)` Gradient flow check: trainable LoRA params in `control_model` and `diffusion_model.output_blocks` (both self-attention `attn1` and cross-attention `attn2`) receive non-zero gradients, while all frozen base params have `grad is None`.
       - `(d)` Both noise samples $(\epsilon_1, \epsilon_2)$ use the exact same batch $x_0$, degraded LR $\hat{x}_{\text{LR}}$, conditioning $c$, and timestep $t$.
       - `(e)` Fairness check: Variants `A`, `B`, and `C` all draw 2 noise samples per step, and $\mathcal{L}_C(\lambda_{\text{cons}}=0) == \mathcal{L}_B$, $\mathcal{L}_B(\lambda_{\text{rec}}=0) == \mathcal{L}_A$.
       - `(f)` CPU dry run: 2 optimizer steps on a tiny dummy module, save/reload LoRA checkpoint, verify bit-exact reload.
  2. **Task `M3-2` (By Fri `13/11 18:00` — Pair with Member 1 on Colab Smoke `A` & Trial `C`):**
     - Help Member 1 run the 20-step Colab smoke test (`LoRA A`) and 50–100 step trial (`LoRA C`), logging VRAM peak and step time, plus 3-image inference sanity check (`no NaN`).
- **What to Send Back to Member 1:**
  - Terminal output showing all 6 CPU tests `(a)`–`(f)` passing (`Thu 12/11 21:00`), and any adjustments to `src/lora_text_loss.py` / `scripts/run_lora_trial.py` via PR from `feat/lora-loss`.

---

### Member 4: Nguyễn Phan Minh Nhật (`25125033`) — Baseline Report (`A5`) & Figures/Metrics Lead
- **Branch:** `feat/report-a5` (or `feat/tv4-report-text-eval`)
- **Key Resources Ready for Use:**
  - **1-Page Concept Comparison (CCSR, PNFS, TADiSR):** Comprehensive reference already written in [`docs/related_methods_ccsr_pnfs_tadisr.md`](docs/related_methods_ccsr_pnfs_tadisr.md). Integrate directly into the report.
  - **DoSSR Inference Steps:** Fixed strictly to **5 steps** (`steps: 5`, 3rd-order solver, `trunc_12` schedule).
- **What You Can Do Right Now (Zero Waiting — All Whole-Image & Equivalence Numbers Are `VERIFIED`!):**
  1. **Task `M4-1` (By Thu `12/11 21:00` — Write Sections 1, 2, 3, 4.1, and 6 of `docs/A5_baseline.tex` + `Bicubic x4` Whole-Image Row):**
     - Start from the `docs/A5_baseline.tex` template (which already has all `VERIFIED` numbers from `docs/fact_sheet_for_report.md` pre-filled and `PENDING` placeholders marked).
     - Complete **Section 1 (Method & Pipeline)**, **Section 2 (Implementation Details & Equivalence Checks: `SDPA`, `prompt wrapper`, `seed 42 repeat`)**, **Section 3 (Experimental Setup + Explicit `PSNR/SSIM/LPIPS` Metric Definitions: library, channels RGB vs BT.601 Y, `crop_border=0`, AlexNet `normalize=True`)**, **Section 4.1 (10-Seed Whole-Image Metrics Table)**, and **Section 6 (Reproducibility Guide)**.
     - Compute the **Bicubic $\times 4$ (`test_LR -> 512x512`) whole-image `PSNR`, `SSIM`, `LPIPS` reference row** so Section 4.1 has a lower-bound anchor alongside `DoSSR default` and `DoSSR empty`.
- **What You Do After M2 Sends the Unified Text Tables (Fri `13/11`):**
  2. **Task `M4-2` (By Fri `13/11 22:00` — Complete Full Draft of `docs/A5_baseline.tex`, Waits for `M2-2`):**
     - Fill the remaining `PENDING` placeholders in **Section 4.2 & 4.3** using Member 2's outputs (Box Error Rate $1-\text{WCR}$, `136`-box solvable subset, bootstrap CIs, capped/median CER, non-text control boxes, and the `eduroam` 4-panel comparison figure).
     - Integrate the 1-page CCSR, PNFS, TADiSR comparison into Section 5 / Discussion.
     - *Remember the Decoupling Rule:* Do **not** wait for LoRA to finish `docs/A5_baseline.tex`! A5 is strictly the Baseline Report.
  3. **Task `M4-3` (Sat `14/11` – Sun `15/11`):** Run through the **Number-Audit Checklist** with the team and finalize formatting for Monday's submission (`25125002_25125042_25125060_25125033.pdf`).
