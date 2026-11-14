# context.md: read this first

You are joining Team 1 (`LEMMEPASS`) in SC203 Scientific Methods (Class 25A02). This file tells you what the project is, what is done, what is not, and what to do next. Facts below come from our real logs (Oct 29 – Nov 14, 2026), direct source inspection of `external/DOSSR`, and verified team evaluation CSVs in `data/` and `results/`. Anything not confirmed is marked **NOT TESTED** or **PENDING**. Do not guess it. Ask the user.

## 1. Project, Deadlines & Mentor Direction (Nov 2026)

- **Topic 3:** Seed-robust real-world super-resolution with text-region consistency.
- **Base model:** DoSSR (public code and pretrained checkpoint).
- **Updated Research Question (Nov 2026 Mentor Direction):**
  - *"Does adding a loss during LoRA training help DoSSR restore text correctly more consistently when the seed changes?"*
  - The baseline is tentatively accepted. Prompt/instruction injection (InstructIR) remains **deferred** until the text-region consistency loss is validated.
- **Dr. Thuận's Nov 12 Protocol & Baseline Directives:**
  - **LoRA Training Data & Leak-Free Split:** Train LoRA on the **`train` split of `Real-CE x4`**; partition `val` by original image or scene **before** cutting patches (holding `test` out untouched); select checkpoint and loss weights on `val`, then evaluate on `Real-CE test` and check cross-dataset transfer on `RealSRVal`.
  - **Priority Controlled Pilot (`LoRA 2` vs. `LoRA 3`):** Compare **LoRA 2** (increased text-region loss weight) against **LoRA 3** (same text-region weight + cross-noise consistency loss over $K=2$ noise draws) using identical data, initial checkpoint, empty prompt `""`, step budget, and fixed validation across the **same 10 seeds (`42..51`) and 5 inference steps**. Report $\Delta$ text error rate, $\Delta$ CER, $\Delta$ cross-seed disagreement, whole-image LPIPS, peak VRAM, and `s/step`.
  - **Baseline Audit for Assignment 5:** Report separately on all `193` labeled boxes and the `136` boxes where `test_HR` OCR matches the human transcript; state the exact formula for `32.83%` cross-seed disagreement; trace why earlier `seed 42` `eduroam` (`Canon_001`) figures differed.
- **Deadlines:**
  - **Group Assignment 4 (`SOTA's Limitations`):** Submitted (`docs/A4_sota_limitations.tex`).
  - **Group Assignment 5 (`Baseline`):** Official due **Tue `17/11/2026 07:00`** (Internal submit **Mon `16/11/2026 21:00`**, output filename `25125002_25125042_25125060_25125033.pdf`).

## 2. Confirmed Team Roster (4 Members), Branches & PR Rule

- **Merge & Status Rule:** **Members 2, 3, and 4 must NOT edit `context.md` directly.** Members report progress in the team chat, and Member 1 (via the agent) updates `context.md` and merges teammate branches into `feat/baseline-runner` / `main`.

| Member | Full Name (`Student ID`) | Role | Branch | Owned Files & Deliverables | Hardware |
|---|---|---|---|---|---|
| **Member 1 (User)** | **Hoàng Nguyên Anh (`25125002`)** | Team, GPU & Pipeline Lead | `feat/baseline-runner` / `main` | `src/run_dossr.py`, `src/dossr_inference_compat.py`, `src/attention_compat.py`, `src/lora_text_loss.py`, `scripts/run_seed_study.py`, `scripts/check_study_outputs.py`, `scripts/merge_text_boxes.py`, `scripts/check_realce_feasibility.py`, `scripts/split_realce_by_scene.py`, `scripts/run_lora_trial.py`, `configs/*`, `notebooks/01_dossr_baseline.ipynb`, `notebooks/02_realce_feasibility_check.ipynb`, `notebooks/03_lora_pilot_colab.ipynb`, `context.md` | Colab Pro GPU (`NVIDIA L4`) + local Mac |
| **Member 2** | **Huỳnh Trần Gia Hân (`25125042`)** | Text Labeling (`1–50`) & Text Evaluation Lead | `feat/text-labels-canon` (merged) | `data/text_boxes_realsr_canon.csv`, `results/text_eval_canon_*.csv`, `scripts/analyze_text_eval_a5.py`, `results/text_eval_a5_cohort_breakdown.csv`, `results/text_eval_a5_summary.json`, `scripts/make_text_contact_sheets.py`, `docs/handoff_member2.md` | CPU laptop / Free Colab |
| **Member 3** | **Hồ Quang Minh (`25125060`)** | Text Labeling (`51–100`), OCR Calibration & LoRA Lead | `feat/text-labels-nikon`, `fix/ocr-rec-only`, `feat/lora-loss` (merged) | `data/text_boxes_realsr_nikon.csv`, `scripts/ocr_check_hr.py`, `scripts/text_eval.py` (`--rec-only --rotate-vertical`), `results/text_eval_nikon_*.csv`, `docs/lora_feasibility_and_loss_design.md`, `configs/lora2_pilot.yaml`, `configs/lora3_pilot.yaml`, `docs/handoff_member3.md` | CPU laptop / Free Colab |
| **Member 4** | **Nguyễn Phan Minh Nhật (`25125033`)** | Report (`A4`/`A5`), Whole-Image Metrics & Visualization Lead | `feat/report-figures` / `feat/tv4-report-text-eval` (merged) | `docs/A4_sota_limitations.tex`, `docs/A5_baseline.tex`, `docs/related_methods_ccsr_pnfs_tadisr.md`, `scripts/compute_metrics.py`, `scripts/plot_eduroam_figure.py`, `scripts/plot_eduroam_10seeds.py`, `docs/handoff_member4.md` | CPU laptop / Free Colab GPU (`--device cuda`) |

## 3. Task Status & Dependency Table

> **Rule:** Only Member 1 / the agent updates this table. GPU notebooks stay `NOT TESTED` until Colab execution logs are returned.

| Task | Description | Owner | Status | Blocked By |
|---|---|---|---|---|
| **B.5 & B.5-Label** | Contact sheet generator, validator, merger (`data/text_boxes_realsr.csv`: `100` images, `25` text-positive [`10` Canon + `15` Nikon], `193` `readable=yes` boxes [`50` Canon + `143` Nikon], `0` errors) | Member 1, 2, 3 | **Done (`VERIFIED`)** | None |
| **B.6-OCR & Eval** | `scripts/ocr_check_hr.py` + unified 10-seed `scripts/text_eval.py` (`--rec-only --rotate-vertical`) across `1,930` box-seed crops, `HR_CEILING`, `BICUBIC_X4_FLOOR`, and `default` vs `empty` prompts | Member 2 & Member 3 | **Done (`VERIFIED` in `results/text_eval_*.csv`)** | None |
| **A.1–A.4-Run & B.8** | 100-image $\times$ 10-seed $\times$ 2-prompt study (`2,000` `512x512` PNGs on Drive, all 20 manifests `fp16` + `SDPA OFF`) & `eduroam` figures (`plot_eduroam_figure.py`, `plot_eduroam_10seeds.py`) | Member 1 & Member 4 | **Done (`VERIFIED` on `131e636` / `334808a`)** | None |
| **B.7** | Whole-image DoSSR 10-seed metrics (`results/metrics_10seed_*.csv`) | Member 4 | **Done (`VERIFIED` in `7564c85`)** | None |
| **Phase A — A5 Cohort & Metrics** | `scripts/analyze_text_eval_a5.py` (`193` vs `136` solvable vs `57` unsolvable cohorts + `seed 42` audit), `scripts/compute_metrics.py --bicubic-lr-dir` (synthetic PNG check), and `docs/A5_baseline.tex` (`pdflatex` 4-page build) | Member 1, 2, 4 | **Done (`VERIFIED` locally); `RealSRVal` Bicubic whole-image row `PENDING` GPU/Drive run** | Colab cell `03` of `03_lora_pilot_colab.ipynb` |
| **Phase A — Real-CE Split & LoRA 2/3 Harness** | `scripts/split_realce_by_scene.py` (leak-free `train`/`val` split + paired patches; synthetic test `VERIFIED`), `src/lora_text_loss.py` (`6/6` CPU unit tests `VERIFIED`), `scripts/run_lora_trial.py` (`--dry-run-cpu` `VERIFIED`), `configs/lora{2,3}_pilot.yaml`, `notebooks/03_lora_pilot_colab.ipynb` | Member 1 & Member 3 | **Done (`VERIFIED` on CPU); GPU training `NOT TESTED`** | Phase B Colab GPU execution (`02` + `03` notebooks) |
| **D-Report (`A4` & `A5`)** | `docs/A4_sota_limitations.tex` (submitted), `docs/fact_sheet_for_report.md` (`VERIFIED`), `docs/related_methods_ccsr_pnfs_tadisr.md`, and `docs/A5_baseline.tex` (due Tue `17/11 07:00`) | Member 4 & All | **A4 Done; A5 Compiles Cleanly (`PENDING` Bicubic whole-image floor & final review)** | Bicubic floor from Colab + final team cross-check |

## 4. Repo & Key Files

```bash
git clone https://github.com/lemon4life/SC203
cd SC203
git checkout main   # synced with feat/baseline-runner
```

| Path | What it is |
|---|---|
| `docs/fact_sheet_for_report.md` | **Master verified fact sheet for A4 & A5 reports:** dataset stats, hardware/VRAM profiles, 10-seed text evaluation tables, dedicated `193` vs `136` vs `57` cohort breakdown, `32.83%` disagreement formula, and `eduroam` `seed 42` provenance audit. |
| `docs/A5_baseline.tex` | Draft Assignment 5 (`Baseline`) LaTeX report (`pdflatex` tested cleanly to 4 pages; only Bicubic $\times 4$ whole-image row remains `PENDING`). |
| `docs/team_plan_A5_tasks_deadlines_dependencies.md` & `docs/plans/07_member1_gpu_pilot_and_realce_split_plan.md` | Active A5 & LoRA pilot task breakdown, timeline, dependencies, and Colab execution runbook. |
| `data/text_boxes_realsr.csv` | Merged 100-image ground-truth text box dataset (`276` rows, `100` images, `25` images with `readable=yes` text, `193` `readable=yes` boxes). Individual camera splits: `data/text_boxes_realsr_canon.csv` (`50` boxes) and `data/text_boxes_realsr_nikon.csv` (`143` boxes). |
| `data/realce_splits_manifest.json` | Schema/template manifest for scene-level `Real-CE x4` `train`/`val` splits (`scripts/split_realce_by_scene.py`). |
| `results/text_eval_*.csv` & `results/text_eval_a5_*` | Complete 10-seed text evaluation CSVs (`Combined 100`, `Canon`, `Nikon`), plus `results/text_eval_a5_cohort_breakdown.csv` and `results/text_eval_a5_summary.json`. |
| `results/metrics_10seed_*.csv` | Whole-image 10-seed `PSNR`, `SSIM`, `LPIPS`, and cross-seed spread for `default` and `empty` prompts. |
| `scripts/analyze_text_eval_a5.py`, `scripts/text_eval.py`, `scripts/ocr_check_hr.py` | Standardized RapidOCR text evaluators (`--rec-only --rotate-vertical`) and A5 cohort + `seed 42` provenance analyzer. |
| `scripts/compute_metrics.py` | Full-reference PSNR, SSIM, LPIPS, pairwise seed spread, and `--bicubic-lr-dir` Bicubic $\times 4$ floor evaluator. |
| `scripts/split_realce_by_scene.py` & `scripts/check_realce_feasibility.py` | `Real-CE x4` feasibility auditor and leak-free scene-level `train`/`val` splitter + aligned patch extractor (`128x128` LR, `512x512` HR). |
| `src/lora_text_loss.py`, `scripts/run_lora_trial.py`, `configs/lora2_pilot.yaml`, `configs/lora3_pilot.yaml` | LoRA 2 (text-weighted) & LoRA 3 (text-weighted + $K=2$ cross-noise consistency) training loss, trial runner (supports `per_image_seed: true` 10-seed val inference), and pilot YAML configs. |
| `notebooks/01_dossr_baseline.ipynb`, `02_realce_feasibility_check.ipynb`, `03_lora_pilot_colab.ipynb` | Colab notebooks: `01` (`VERIFIED` RealSRVal baseline), `02` (`NOT TESTED` Real-CE Drive staging & feasibility), `03` (`NOT TESTED` Bicubic floor + Real-CE scene split + LoRA 2 vs. LoRA 3 L4 pilot). |
| `docs/handoff_member2.md`, `docs/handoff_member3.md`, `docs/handoff_member4.md` | Up-to-date handoff guides for Members 2, 3, and 4. |

## 5. Summary of Verified Findings (`CELL 01`–`18`, 100-Image `text_eval.py`, & A5 Audit)

1. **Colab GPU (`CELL 17` / `CELL 03`):** `NVIDIA L4` (`22.5 GiB` used for the 10-seed studies), PyTorch `2.11.0+cu130`, Python `3.13.16`.
2. **RealSR Dataset & Text Box Ground Truth (`CELL 04` + `data/text_boxes_realsr.csv`):**
   - `100` paired `test_LR` (`128x128`) and `test_HR` (`512x512`) images.
   - **`25 / 100` images** contain human-readable text (`10` Canon + `15` Nikon), totaling **`193` `readable=yes` word boxes** (`50` Canon + `143` Nikon).
3. **1-Image `test_LR` (`128x128 -> 512x512`) FP16 Profile (`CELL 08`–`09`):**
   - `SDPA ON`: `7.32 GiB` allocated (`8.29 GiB` reserved), `1.75 s/img`.
   - `SDPA OFF` (`--disable-sdpa`): `7.32 GiB` allocated (`8.28 GiB` reserved), `1.92 s/img`.
4. **Full 10-Seed $\times$ 2-Prompt Study (`CELL 16`–`18`, `2,000` PNGs `VERIFIED`):**
   - `default` (`"clean, high-resolution, 8k"`) and `empty` (`""`) prompt studies (`1,000` `512x512` PNGs each across `seed_42`..`seed_51`, `SDPA OFF`, `fp16`) are verified on Drive.
   - Whole-image metrics (`results/metrics_10seed_*.csv`): `default` `PSNR_RGB = 22.6353±0.5961 dB`, `SSIM_RGB = 0.6406±0.0182`, `LPIPS = 0.3382±0.0191`; `empty` `PSNR_RGB = 22.6998±0.6079 dB`, `SSIM_RGB = 0.6431±0.0184`, `LPIPS = 0.3352±0.0191`.
5. **10-Seed Text Restoration & Seed-Instability Cohort Breakdown (`VERIFIED` in `results/text_eval_a5_cohort_breakdown.csv`):**
   - **All `193` Labeled Boxes (`25` images):** `HR_CEILING = 70.5%` (`136/193`, `CER = 0.1648`); `BICUBIC_X4_FLOOR = 20.2%` (`39/193`, `WCR = 79.79%`, `CER = 0.7423`); DoSSR 10-seed (`default`, `1,930` box-seeds) = `24.0%` (`463/1930`), **`WCR = 76.01%`**, **`CER = 0.7127`**, **`Disagreement = 32.83%`** (`3.15` distinct readings / 10 seeds).
   - **`136` OCR-Solvable Subset (`20` images, where `HR CER = 0`):** `HR_CEILING = 100.0%`; `BICUBIC_X4_FLOOR = 28.7%` (`WCR = 71.32%`, `CER = 0.6626`); DoSSR 10-seed = `33.7%` (`458/1360`), **`WCR = 66.32%`**, **`CER = 0.6186`**, **`Disagreement = 34.05%`** (`3.22` distinct readings).
   - **`57` HR-Unsolvable Subset (`20` images):** `HR_CEILING = 0.0%` (`CER = 0.5579`); DoSSR 10-seed = `0.9%` (`5/570`), **`WCR = 99.12%`**, **`CER = 0.9370`**, **`Disagreement = 29.90%`** (`2.98` distinct readings).
   - **Paired Prompt Comparison (`empty - default`, `1,930` pairs):** $\Delta\text{WCR} = -0.16\%$, $\Delta\text{CER} = +0.0001$.
6. **Eduroam (`Canon_001`) `seed 42` Provenance (`VERIFIED`):** Earlier visual figures differed due to (a) upstream `inference.py` calling `pl.seed_everything(args.seed)` once per folder while `os.walk()` traverses files unsorted (1-image study drew 1st RNG vs. 100-image study drew a later PNG's RNG), and (b) OCR using tight word box `(320,38,192,46)` while figure scripts use full sign crop `(310,5,202,85)`. All canonical figures read from `seed-study-100img`.

## 6. Phase B Next Steps (Pending Colab L4 GPU Execution)

1. Execute **`notebooks/02_realce_feasibility_check.ipynb`** on Colab to download/unpack `Real-CE` onto Google Drive (`/content/drive/MyDrive/lemmepass/data/Real-CE`), verify pairing/annotations/HR OCR, and export `results/realce_feasibility_summary.json`.
2. Execute **`notebooks/03_lora_pilot_colab.ipynb`** on Colab L4 GPU to compute the `RealSRVal` Bicubic $\times 4$ whole-image row (`scripts/compute_metrics.py --bicubic-lr-dir ...`), generate leak-free `Real-CE x4` scene splits + patches (`scripts/split_realce_by_scene.py`), run **LoRA 2 vs. LoRA 3** pilot training, and run 10-seed 5-step fixed validation inference + gating comparison.
