# Team Plan, Task Division, Deadlines & Dependencies — Assignment 5 (`Baseline`) & LoRA Pilot

**Course:** SC203 Scientific Methods (Class 25A02) — Team 1 (`LEMMEPASS`)  
**Repository & Branch:** `https://github.com/lemon4life/SC203` (`feat/baseline-runner`)  
**Required Submission Format:** One PDF named `<StudentID1>_<StudentID2>_<StudentID3>_<StudentID4>.pdf`  
**Final Submission Filename:** `25125002_25125042_25125060_25125033.pdf`  
(Nguyễn Anh `25125002`, Gia Hân `25125042`, Quang Minh `25125060`, Minh Nhật `25125033`)  
**Last Updated:** Saturday, `14/11/2026` (incorporating Dr. Thuận's latest directives)

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
| **Sat `14/11`** | `21:00` | **Task M1-1:** Author `scripts/split_realce_by_scene.py` to create leak-free train/val splits from Real-CE $\times 4$.<br>**Task M3-1:** Verify `src/lora_text_loss.py` CPU unit tests `(a)`–`(f)` and prepare LoRA 2 & 3 pilot configs.<br>**Task M4-1:** Update `docs/A5_baseline.tex` with the 193 vs 136 table, 32.83% formula, and eduroam crop note. | **M1** (`Nguyên Anh`)<br>**M3** (`Quang Minh`)<br>**M4** (`Minh Nhật`) | Track 1 & 2 |
| **Sun `15/11`** | `14:00` | **Task M1-2 & M3-2:** Launch Colab GPU Pilot of LoRA 2 vs. LoRA 3; log peak VRAM (allocated/reserved) and step time; run 10-seed 5-step inference on fixed validation set. | **M1** (`Nguyên Anh`) & **M3** (`Quang Minh`) | Track 2 |
| **Sun `15/11`** | `18:00` | **Task M2-2:** Run text evaluation on LoRA 2 vs LoRA 3 pilot validation outputs (Error Rate, CER, Disagreement, LPIPS); evaluate gating condition. | **M2** (`Gia Hân`) | Track 2 |
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
  1. **Task M1-1: Real-CE $\times 4$ Scene-Level Split Script (`scripts/split_realce_by_scene.py`):**
     - Group images by base scene / original image prefix before cropping patches.
     - Partition into `train` (~80%) and `val` (~20%) with zero scene overlap.
     - Export split manifest `data/realce_splits_manifest.json`.
     - *Verification:* Assert $\text{scenes}(\text{train}) \cap \text{scenes}(\text{val}) = \emptyset$.
  2. **Task M1-2: Execute Colab GPU Pilot (LoRA 2 vs. LoRA 3):**
     - Launch `notebooks/03_lora_pilot_colab.ipynb` or runner script.
     - Train LoRA 2 (text-weighted) for $N$ steps (e.g. 100 steps) with empty prompt `""`.
     - Train LoRA 3 (text-weighted + consistency loss) under identical initialization, seed, batch size, and steps.
     - Record in `results/pilot_lora_manifest.json`: GPU model, peak allocated VRAM, peak reserved VRAM, and average step time (sec/step).
     - *Verification:* Checkpoints save cleanly to Google Drive; no NaN/Inf in training loss.
  3. **Task M1-3: Run 10-Seed 5-Step Validation Inference:**
     - Run both checkpoints (`configs/lora2_pilot.yaml` and `configs/lora3_pilot.yaml`) via `python scripts/run_seed_study.py --config configs/lora{2,3}_pilot.yaml` across seeds `42`–`51` with `--per-image-seed` and `5` inference steps.
     - Save outputs to Drive: `outputs/pilot-lora2-realce-val-10seed/` and `outputs/pilot-lora3-realce-val-10seed/`.
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
  2. **Task M2-2: Pilot Evaluation Pipeline (LoRA 2 vs. LoRA 3):**
     - Run `scripts/text_eval.py` on the validation outputs from Member 1:
       ```bash
       python scripts/text_eval.py \
         --study-root "/content/drive/MyDrive/lemmepass/outputs/pilot-lora3-realce-val-10seed" \
         --compare-study-root "/content/drive/MyDrive/lemmepass/outputs/pilot-lora2-realce-val-10seed" \
         --boxes-csv "data/realce_val_boxes.csv" \
         --output-dir "results/pilot_lora2_vs_lora3"
       ```
     - Output the delta comparison table:
       * $\Delta \text{Box Error Rate} = \text{Error}(\text{LoRA 3}) - \text{Error}(\text{LoRA 2})$
       * $\Delta \text{CER} = \text{CER}(\text{LoRA 3}) - \text{CER}(\text{LoRA 2})$
       * $\Delta \text{Seed Disagreement} = \text{Dis}(\text{LoRA 3}) - \text{Dis}(\text{LoRA 2})$
       * $\Delta \text{LPIPS} = \text{LPIPS}(\text{LoRA 3}) - \text{LPIPS}(\text{LoRA 2})$
     - *Verification:* Check if $\Delta \text{Box Error Rate} < 0$ and LPIPS does not degrade.

---

### Member 3: Hồ Quang Minh (`25125060`) — LoRA Architecture & Consistency Loss Lead
* **Primary Branch:** `feat/lora-loss`
* **Core Responsibilities:**
  1. **Task M3-1: LoRA 2 & LoRA 3 Configurations & CPU Unit Tests:**
     - Maintain and verify `src/lora_text_loss.py` (`python src/lora_text_loss.py --run-cpu-tests`) and `scripts/run_lora_trial.py --dry-run-cpu`.
     - Confirm all 6 CPU unit tests pass:
       * `(a)` `LoRALinear` zero-init equivalence & base-weight freeze.
       * `(b)` Selective LoRA injection on `control_model` + `output_blocks` skipping `no_grad` encoder/middle blocks.
       * `(c)` HR box mask rasterization (`512x512`) and latent max-pool downsampling (`64x64`).
       * `(d)` Text-weighted L1 equivalence ($\alpha_{\text{text}}=1$) and text-box weighting ($\alpha_{\text{text}}>1$).
       * `(e)` Two-pass chain-rule backward exact gradient equivalence vs. single-graph joint backward.
       * `(f)` LoRA checkpoint save/load roundtrip fidelity.
  2. **Task M3-2: Support Colab Pilot Execution:**
     - Assist Member 1 with LoRA hyperparameters ($\lambda_{\text{text}}$, $\lambda_{\text{cons}}$, rank $r=8$ or $16$, learning rate $1\times 10^{-4}$).
     - Monitor loss curves (diffusion loss vs consistency loss) to ensure stable convergence.
  3. **Task M3-3: Post-Pilot Decision Action:**
     - If LoRA 3 passes gating: configure LoRA 1 (baseline loss) for the full 3-variant comparison.
     - If LoRA 3 fails gating: inspect text masks and gradient magnitudes on failure samples.

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

[PHASE 3: Real-CE x4 Dataset Preparation & Leak-Free Split] (SCRIPT READY / COLAB EXECUTION PENDING)
  [x] Real-CE x4 feasibility check (pairing, 4x scale, annotations, HR OCR).
  [x] Implement & CPU-test scripts/split_realce_by_scene.py (scene-level partitioning, LR/HR pairing, --extract-patches).
  [ ] Run split script on Real-CE Drive data and populate data/realce_val_boxes.csv / data/realce_train_manifest.csv.

[PHASE 4: LoRA Loss Architecture & Unit Testing] (COMPLETED / VERIFIED)
  [x] Mathematical formulation of cross-noise latent consistency loss & two-pass chain rule.
  [x] Pass 6 CPU unit tests (a)-(f) in src/lora_text_loss.py & scripts/run_lora_trial.py --dry-run-cpu.
  [x] Configure LoRA 2 (configs/lora2_pilot.yaml) and LoRA 3 (configs/lora3_pilot.yaml) + notebooks/03_lora_pilot_colab.ipynb.

[PHASE 5: Colab GPU Pilot Run & Gating Evaluation] (TARGET: Sun 15/11)
  [ ] Run LoRA 2 pilot on Colab GPU (log peak VRAM & sec/step).
  [ ] Run LoRA 3 pilot on Colab GPU (log peak VRAM & sec/step).
  [ ] 10-seed 5-step inference on fixed validation set.
  [ ] Evaluate Δ Error Rate, Δ CER, Δ Disagreement, Δ LPIPS (M2).
  [ ] Evaluate Gating Condition (LoRA 3 vs LoRA 2).

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
