# Team Plan, Task Division, Deadlines & Dependencies — Assignment 5 (`Baseline`) & LoRA Pilot

**Course:** SC203 Scientific Methods (Class 25A02) — Team 1 (`LEMMEPASS`)  
**Repository & Branch:** `https://github.com/lemon4life/SC203` (`feat/baseline-runner`)  
**Required Submission Format:** One PDF named `<StudentID1>_<StudentID2>_<StudentID3>_<StudentID4>.pdf`  
**Final Submission Filename:** `25125002_25125042_25125060_25125033.pdf`  
(Nguyễn Anh `25125002`, Gia Hân `25125042`, Quang Minh `25125060`, Minh Nhật `25125033`)  
**Last Updated:** Monday, `16/11/2026` (integrating Member 3's `--rec-only --upscale 2` OCR re-score and 5-crop diagnostic check)

---

## 1. Dual Independent Tracks (Strict Decoupling Rule)

1. **Track 1 — Group Assignment 5 (`A5` = Baseline Report):**
   - **Official Course Deadline:** **Tuesday `17/11/2026 07:00`**
   - **Internal Submission Deadline:** **Monday `16/11/2026 21:00`** (10-hour safety buffer)
   - **Scope:** Complete, reproducible DoSSR baseline report (`docs/A5_baseline.tex`) covering:
     1. Method & runtime compatibility pipeline (`SDPA`, prompt wrapper, `FP16` manifest checks).
     2. Implementation & equivalence verification (`SDPA ON vs OFF`, `default vs unwrapped`, `seed 42 vs 42-repeat`).
     3. 10-seed $\times$ 2-prompt whole-image metrics (`PSNR`, `SSIM`, `LPIPS`, cross-seed spread, plus `Bicubic x4` reference row).
     4. 10-seed text evaluation strengthened with mentor-mandated baseline audits:
        - **193 labeled boxes vs. 136 OCR-solvable subset:** Report performance separately on all 193 boxes and the 136 boxes where HR ground truth OCR matches the human transcript.
        - **Formal mathematical formula for 32.83% cross-seed disagreement rate.**
        - **Traceability audit for `eduroam` seed 42 crops:** Explain the framing difference between the word box ($192\times 46$) and the figure sign crop ($202\times 85$ with Wi-Fi icon).
   - **Decoupling Rule:** **`A5` does NOT depend on LoRA.** `A5` is 100% self-contained. Mention LoRA in `A5` only as a short 1-paragraph note in Future Work.

2. **Track 2 — Parallel LoRA Pilot & Real-CE $\times 4$ Pipeline (For Mentor Report):**
   - **Mentor Directives (Received Thursday 11:41pm):**
     1. Use the **train split of Real-CE $\times 4$** for training LoRA.
     2. **Split validation by original image or scene before cropping patches** (guarantee zero patch leakage between train and val).
     3. **Prioritize pilot comparison of LoRA 2 vs. LoRA 3:**
        - **LoRA 2:** Increased text-region loss weight ($\lambda_{\text{text}}$).
        - **LoRA 3:** Same text-region weight + consistency loss penalizing discrepancies between two clean latent predictions ($\hat{x}_0^{(1)}, \hat{x}_0^{(2)}$) inside text masks across two independent noise samples $(\epsilon_1, \epsilon_2)$ at the same timestep $t$.
        - Same data, initialization checkpoint, empty prompt `""`, 2 noise samples, and step budget.
        - Evaluate both on the fixed validation set across the **same 10 seeds** and **5 inference steps**.
        - Report: $\Delta$ Text Error Rate (share of boxes with $\ge 1$ wrong char vs HR transcript), $\Delta$ CER, $\Delta$ seed disagreement rate, $\Delta$ whole-image LPIPS, peak VRAM, and per-step training time.
     4. **Gating Rule:**
        - If LoRA 3 reduces text errors and LPIPS meets threshold $\rightarrow$ complete LoRA 1 (original loss baseline) and benchmark all 3 on Real-CE test and RealSRVal.
        - If text errors are comparable or increase $\rightarrow$ inspect failure cases on validation to diagnose mask alignment and loss weights.

---

## 2. Master Schedule & Deadlines (`14/11` – `17/11/2026`)

| Day & Date | Time | Milestone / Deliverable | Owner(s) | Track |
| :--- | :---: | :--- | :---: | :---: |
| **Sat `14/11`** | `21:00` | **`VERIFIED` Task M1-1:** Scene-disjoint `scripts/split_realce_by_scene.py` (`0` overlap).<br>**`VERIFIED` Task M3-1:** `src/lora_text_loss.py` CPU tests `(a)`–`(f)` + dry run.<br>**Pending Task M4-1:** Update `docs/A5_baseline.tex` with 193 vs 136 table, 32.83% formula, and eduroam crop note. | **M1** (`Nguyên Anh`)<br>**M3** (`Quang Minh`)<br>**M4** (`Minh Nhật`) | Track 1 & 2 |
| **Sun `15/11`** | `14:00` | **`VERIFIED` Task M1-2 & M3-2:** Colab GPU Pilot (`400` steps each) + 10-seed 5-step Val inference completed (`8.04 GiB` peak alloc for both, `4.67s` vs `5.38s/step`, `-57.5%` pixel / `-64.1%` LPIPS seed diff, `-28.5%` OCR disagreement). | **M1** (`Nguyên Anh`) & **M3** (`Quang Minh`) | Track 2 |
| **Sun `15/11` – Mon `16/11`** | `18:00` | **`VERIFIED` Task M2-2 & M3-3:** Re-scored saved Drive PNGs with `--rec-only --rotate-vertical --upscale 2` (`154` boxes: $\Delta\text{wrong\_char} = -0.0262 \pm 0.0084$, $\Delta\text{CER} = -0.0264 \pm 0.0122$, `-0.0619` on 63 HR-solvable boxes, `-13.82%` OCR disagreement) + 5 crop grids & mask alignment verified (`docs/M3_3_pilot_reconly_and_crops_notes.md`). | **M3** (`Quang Minh`) *(covering M2-2 + M3-3)* | Track 2 |
| **Sun `15/11`** | `21:00` | **A5 Internal Freeze:** Complete Section 1–6 in `docs/A5_baseline.tex`, compile clean PDF on Overleaf, execute Number-Audit Check against `docs/fact_sheet_for_report.md`. | **M4** (`Minh Nhật`) & **M1** (`Nguyên Anh`) | **Track 1 (`A5`)** |
| **Mon `16/11`** | **`21:00`** | **INTERNAL SUBMIT DEADLINE:** Final proofread, export `25125002_25125042_25125060_25125033.pdf`, submit Assignment 5, and archive confirmation screenshot. | **M1** (`Nguyên Anh`) & All | **Track 1 (`A5`)** |
| **Tue `17/11`** | **`07:00`** | **OFFICIAL COURSE DEADLINE: Group Assignment 5 (`Baseline`).** | All | **Track 1 (`A5`)** |

---

## 3. Dependency Graph

```text
[Track 1: Assignment 5 Baseline Report — Zero Dependency on LoRA]
  All 2,000 RealSR PNGs & data/text_boxes_realsr.csv ready on Drive
       │
       ├──► M4 (Sat 14/11 21:00): Update docs/A5_baseline.tex with:
       │    ├── Table: 193 labeled boxes vs 136 solvable subset
       │    ├── Equation: Exact 32.83% seed disagreement formula
       │    └── Note: Eduroam seed 42 crop distinction (word vs sign)
       │         │
       │         └──► All (Sun 15/11 21:00): Number-Audit against fact_sheet_for_report.md
       │                   │
       │                   └──► M1 (Mon 16/11 21:00): Final PDF Export & Submit
       │                        (25125002_25125042_25125060_25125033.pdf)

[Track 2: Parallel LoRA Pilot — Mentor Update Package]
  M1 (Sat 14/11): Real-CE Scene-Level Split (zero leakage) ──┐
  M3 (Sat 14/11): LoRA 2 & 3 Configs + CPU Tests (a)-(f) ────┼──► M1 & M3 (Sun 15/11): Colab GPU Pilot Run
                                                             │    (LoRA 2 vs 3, VRAM & step time logged)
                                                             │         │
                                                             │         └──► M1: 10-seed 5-step Val Inference
                                                             │                   │
                                                             └───────────────────┴──► M2 (Sun 15/11): Pilot Eval Table
                                                                                      (Δ Error, Δ CER, Δ Disagree, Δ LPIPS)
                                                                                           │
                                                                                           ├── [Pass] ──► Train LoRA 1 & Full Benchmark
                                                                                           └── [Fail] ──► Debug Mask / Loss Weights
```

---

## 4. Member Task Assignments, Guides & Verification

### Member 1: Hoàng Nguyên Anh (`25125002`) — Team Lead, GPU & Integration Lead
* **Primary Branch:** `feat/baseline-runner`
* **Core Responsibilities:**
  1. **Task M1-1: Real-CE $\times 4$ Scene-Level Split Script (`scripts/split_realce_by_scene.py`) [`VERIFIED` 2026-11-15]:**
     - Group images by base scene / original image prefix before cropping patches.
     - Partition into `train` (~80%) and `val` (~20%) with zero scene overlap (`0` shared scenes verified in Colab `CELL 04`).
     - Export split manifest `data/realce_splits_manifest.json` and extract aligned patches + `data/realce_train_manifest.csv` / `data/realce_val_boxes.csv`.
  2. **Task M1-2: Execute Colab GPU Pilot (LoRA 2 vs. LoRA 3) [`VERIFIED` 2026-11-15]:**
     - Trained LoRA 2 (`configs/lora2_pilot.yaml`, `lambda_cons=0.0`) and LoRA 3 (`configs/lora3_pilot.yaml`, `lambda_cons=0.5`) for `400` steps each under identical initialization, seed `231`, empty prompt `""`, $K=2$ noise draws, `alpha_text=5.0`, and `1,659,904` trainable LoRA params.
     - **Verified Compute & VRAM (`results/lora_pilot_summary.json`):** LoRA 2 `4.67 s/step` (`p95=4.70s`), `8.04 GiB` allocated (`8.52 GiB` reserved); LoRA 3 `5.38 s/step` (`p95=5.41s`, `+15.20%`), **`8.04 GiB` allocated (`8.52 GiB` reserved, `0.00 GiB` extra VRAM)**. Clean checkpoints saved to Drive; zero NaN/Inf.
  3. **Task M1-3: Run 10-Seed 5-Step Validation Inference [`VERIFIED` 2026-11-15]:**
     - Evaluated both checkpoints (`configs/lora{2,3}_pilot.yaml`) across seeds `42`–`51`, `5` steps (`t1_fraction=0.5`), on `30` held-out Real-CE val patches (`300` PNGs per variant) saved under `/content/drive/MyDrive/lemmepass/outputs/pilot-lora{2,3}-realce-val-10seed/`.
     - **Verified Whole-Patch & Stability Results (`LoRA 3 - LoRA 2`):** `PSNR_Y` `17.6980 -> 18.2828` (`+0.5848 dB`), `SSIM_Y` `0.6442 -> 0.6742` (`+0.0300`), `LPIPS` `0.4231 -> 0.4711` (`+0.0480`), `pair_px` `9.2715 -> 3.9405` (**`-57.50%`**), `pair_lpips` `0.2020 -> 0.0726` (**`-64.06%`**).
  4. **Task M1-4: Final A5 Assembly & Submission:**
     - Run `scripts/compute_metrics.py --bicubic-lr-dir ...` on Colab to fill the `Bicubic x4` whole-image row in Table 1.
     - Coordinate final number-audit review with Member 4 (`results/text_eval_a5_summary.json` and `docs/fact_sheet_for_report.md`).
     - Compile and verify final PDF: `25125002_25125042_25125060_25125033.pdf`.
     - Submit before internal deadline **Mon 16/11 21:00**.

---

### Member 2: Huỳnh Trần Gia Hân (`25125042`) — Text Evaluation & Statistical Refinement Lead
* **Primary Branch:** `feat/text-eval-a5`
* **Core Responsibilities:**
  1. **Task M2-1: Lock Baseline Text Evaluation Tables for A5:**
     - Ensure `results/text_eval_*.csv` and `results/text_eval_a5_summary.json` (`scripts/analyze_text_eval_a5.py`) match the verified 193-box / 136-solvable / 57-unsolvable breakdown.
     - Verify the 136-solvable subset stats ($0.0\%$ HR ceiling, $66.32\%$ DoSSR error rate, $71.32\%$ Bicubic floor, $34.05\%$ cross-seed disagreement).
     - Verify the 4-panel `eduroam` comparison figure (`figures/eduroam_10seeds.png` or `figures/eduroam_comparison.png`).
  2. **Task M2-2: Pilot Evaluation Pipeline (LoRA 2 vs. LoRA 3) [`VERIFIED` 2026-11-16 — Completed by M3]:**
     - Initial default RapidOCR (`det+rec`, `--upscale 1`, `n=30` patches) recorded in `CELL 07` (`results/lora_pilot_summary.json`) was misled by `DBNet` missing tight crops (`HR Ceiling` CER `0.7334`).
     - Re-scored on Nov 16 by `M3` (`Quang Minh`) using `--ocr-backend rapidocr --rec-only --rotate-vertical --upscale 2` (`results/pilot_lora2_vs_lora3_reconly/`, `results/pilot_lora3_reconly/`, `154` boxes across `30` patches):
       * **HR Ceiling (`n=30`):** `wrong_char=0.5791 (±0.0491)` | `CER=0.2722 (±0.0538)` (`63/154 = 40.9%` exact HR-solvable boxes).
       * **Bicubic $\times 4$ (`n=30`):** `wrong_char=0.7979 (±0.0445)` | `CER=0.5422 (±0.0694)` (`0.5556` / `0.4184` on 63 solvable boxes).
       * **LoRA 2 VAL (`42..51`):** `wrong_char=0.9141 (±0.0269)` | `CER=0.6914 (±0.0518)` | `seed_disagree=0.5197 (±0.0710)` (`0.8222` / `0.6144` on 63 solvable).
       * **LoRA 3 VAL (`42..51`):** **`wrong_char=0.8878`** `(±0.0291)` | **`CER=0.6650`** `(±0.0549)` | **`seed_disagree=0.4479`** `(±0.0660)` (`0.7603` / `0.5803` on 63 solvable).
       * **Paired Deltas (`LoRA 3 - LoRA 2`):** $\Delta \text{wrong\_char} = -0.0262 \pm 0.0084$ (`-0.0619` on 63 solvable), $\Delta \text{CER} = -0.0264 \pm 0.0122$ (`-0.0340` on 63 solvable), $\Delta \text{seed\_disagree} = -0.0718$ (**`-13.82%`**), $\Delta \text{crop\_px\_L1} = -8.75$ (**`-55.08%`**), $\Delta \text{LPIPS} = +0.0480$ (`0.4231 -> 0.4711`).

---

### Member 3: Hồ Quang Minh (`25125060`) — LoRA Architecture & Consistency Loss Lead
* **Primary Branch:** `feat/lora-loss`
* **Core Responsibilities:**
  1. **Task M3-1: LoRA 2 & LoRA 3 Configurations & CPU Unit Tests [`VERIFIED` 2026-11-15]:**
     - Verified `python src/lora_text_loss.py --run-cpu-tests` (6/6 passing) and `python scripts/run_lora_trial.py --dry-run-cpu` (`use_checkpoint` & `autocast` patches verified):
       * `(a)` `LoRALinear` zero-init equivalence & base-weight freeze.
       * `(b)` Selective LoRA injection on `control_model` + `output_blocks` skipping `no_grad` encoder/middle blocks.
       * `(c)` HR box mask rasterization (`512x512`) and latent max-pool downsampling (`64x64`).
       * `(d)` Text-weighted L1 equivalence ($\alpha_{\text{text}}=1$) and text-box weighting ($\alpha_{\text{text}}>1$).
       * `(e)` Two-pass chain-rule backward exact gradient equivalence vs. single-graph joint backward.
       * `(f)` LoRA checkpoint save/load roundtrip fidelity.
  2. **Task M3-2: Support Colab Pilot Execution [`VERIFIED` 2026-11-15]:**
     - Verified lockstep convergence (`step 1` identical `rec=0.9776`, `text_l1=0.3544`; `step 400` LoRA 2 `rec=0.4878` vs. LoRA 3 `rec=0.4851`, `cons` `0.1571 -> 0.1299`) at zero extra peak VRAM (`8.04 GiB` allocated).
  3. **Task M3-3: Post-Pilot Decision Action [`VERIFIED` 2026-11-16 — Pending Mentor Sign-Off]:**
     - Executed `--rec-only --rotate-vertical --upscale 2` OCR re-score (`M2-2`) and inspected 5 multi-seed validation crop grids (`results/pilot_lora2_vs_lora3_crops/`, documented in `docs/M3_3_pilot_reconly_and_crops_notes.md`).
     - Verified mask alignment is clean; diagnosed `LPIPS +0.0480` and thin-stroke erasure (`2024/06/17`) as `lambda_cons=0.5` over-smoothing across all $t \in [0, 999)$.
     - Proposed next tuning step for Dr. Thuận: reduce `lambda_cons` (`0.5 -> 0.1`–`0.2`) and/or apply consistency only for $t \le 500$ (matching inference `t1_fraction=0.5`).

---

### Member 4: Nguyễn Phan Minh Nhật (`25125033`) — Baseline Report (`A5`) Lead
* **Primary Branch:** `feat/report-a5`
* **Core Responsibilities:**
  1. **Task M4-1: Draft & Complete `docs/A5_baseline.tex`:**
     - Write Sections 1–6 adhering strictly to the A5 brief.
     - Integrate the **1-page related methods comparison** (`docs/related_methods_ccsr_pnfs_tadisr.md`).
     - Incorporate the **193 vs 136 Solvable Subset Table**:
       * Full 193: HR Ceiling $29.5\%$, Bicubic $79.8\%$, DoSSR $76.0\%$ (Disagreement $32.83\%$).
       * Solvable 136: HR Ceiling $0.0\%$, Bicubic $71.3\%$, DoSSR $66.3\%$ (Disagreement $32.83\%$).
     - Embed the **Formal 32.83% Disagreement Formula** in Section 3/4.
     - Add the **Eduroam Crop Traceability Note** in Section 4.3.
  2. **Task M4-2: Number-Audit Verification:**
     - Cross-check every LaTeX table entry against `docs/fact_sheet_for_report.md`.
     - Ensure no unverified numbers or pending placeholders remain.
  3. **Task M4-3: Overleaf Build & PDF Export:**
     - Ensure zero LaTeX compilation errors/warnings.
     - Verify output meets the course naming rule: `25125002_25125042_25125060_25125033.pdf`.

---

## 5. End-to-End Master Project Checklist (Estimated Timeline)

```text
[PHASE 1: Problem Discovery & SOTA Limitations — Assignment 4] (COMPLETED)
  [x] Set up DoSSR inference on Colab GPU with FP16 and SDPA compatibility.
  [x] Discover severe seed sensitivity on real-world text (RealSRVal).
  [x] Label ground-truth text boxes (193 boxes across 25 images).
  [x] Submit Group Assignment 4 PDF (10/11/2026).

[PHASE 2: DoSSR Baseline & Equivalence Verification — Assignment 5] (IN PROGRESS — Freeze Mon 16/11)
  [x] Complete 10-seed whole-image evaluations (PSNR, SSIM, LPIPS across 2,000 images).
  [x] Verify bit-exact same-seed determinism and prompt-wrapper equivalence.
  [x] Audit 193 total boxes vs 136 solvable subset.
  [x] Derive formal mathematical formula for 32.83% cross-seed disagreement.
  [x] Resolve eduroam crop framing distinction (word box vs full sign).
  [ ] Draft Section 1–6 in docs/A5_baseline.tex (M4).
  [ ] Team Number-Audit against docs/fact_sheet_for_report.md (All).
  [ ] Compile and submit 25125002_25125042_25125060_25125033.pdf (Mon 16/11 21:00).

[PHASE 3: Real-CE x4 Dataset Preparation & Leak-Free Split] (COMPLETED / VERIFIED 2026-11-15)
  [x] Real-CE x4 feasibility check (pairing, 4x scale, annotations, HR OCR).
  [x] Implement & CPU-test scripts/split_realce_by_scene.py (scene-level partitioning, LR/HR pairing, --extract-patches).
  [x] Executed split script on Colab Real-CE data with 0 scene overlap (data/realce_val_boxes.csv / data/realce_train_manifest.csv).

[PHASE 4: LoRA Loss Architecture & Unit Testing] (COMPLETED / VERIFIED)
  [x] Mathematical formulation of cross-noise latent consistency loss & two-pass chain rule.
  [x] Pass 6 CPU unit tests (a)-(f) in src/lora_text_loss.py & scripts/run_lora_trial.py --dry-run-cpu.
  [x] Configure LoRA 2 (configs/lora2_pilot.yaml) and LoRA 3 (configs/lora3_pilot.yaml) + notebooks/03_lora_pilot_colab.ipynb.

[PHASE 5: Colab GPU Pilot Run & Gating Evaluation] (COMPLETED / VERIFIED 2026-11-16)
  [x] Run LoRA 2 pilot on Colab GPU (4.67s/step mean, 8.04 GiB alloc / 8.52 GiB reserved).
  [x] Run LoRA 3 pilot on Colab GPU (5.38s/step mean, 8.04 GiB alloc / 8.52 GiB reserved — 0.00 GiB extra VRAM).
  [x] 10-seed 5-step inference on fixed validation set (30 patches x 10 seeds: PSNR_Y +0.5848 dB, SSIM_Y +0.0300, LPIPS +0.0480, pair_px -57.50%, pair_lpips -64.06%).
  [x] Re-run text_eval.py --rec-only --rotate-vertical --upscale 2 on saved Drive PNGs + inspect 5 visual crops (M3 on 2026-11-16: delta_wrong_char -0.0262 ± 0.0084, delta_CER -0.0264 ± 0.0122, -6.19% on 63 solvable boxes, -13.82% OCR disagreement, -55.08% crop pixel spread; masks aligned; lambda_cons=0.5 over-smoothing diagnosed).

[PHASE 6: Gated Full Training & Gating Decision] (TARGET: Wed 18/11)
  [ ] If Gate Passes: Train LoRA 1 (baseline loss) for the full 3-way ablation.
  [ ] If Gate Fails: Analyze failure cases, refine text mask dilation & loss weights.
  [ ] Lock best checkpoint and loss weights on validation.

[PHASE 7: Comprehensive Dual-Dataset Evaluation & Transferability] (TARGET: Sat 21/11)
  [ ] Benchmark locked LoRA model on Real-CE test set (in-domain text SR).
  [ ] Benchmark locked LoRA model on RealSRVal (cross-dataset generalization).
  [ ] Compare against DoSSR original and Bicubic floor across both datasets.

[PHASE 8: Final Report & Presentation — Assignment 6 / Final] (TARGET: End of Term)
  [ ] Draft Final Research Paper with complete methodology, ablation tables, and figures.
  [ ] Open-source verified code, checkpoints, and evaluation manifests.
  [ ] Final slide deck & oral presentation.
```
