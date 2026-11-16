# context.md: read this first

You are joining Team 1 (`LEMMEPASS`) in SC203 Scientific Methods (Class 25A02). This file tells you what the project is, what is done, what is not, and what to do next. Facts below come from our real logs (Oct 29 – Nov 16, 2026), direct source inspection of `external/DOSSR`, verified team evaluation CSVs in `data/` and `results/`, and verified Colab GPU LoRA pilot logs (`results/lora_pilot_summary.json`). Anything not confirmed is marked **NOT TESTED** or **PENDING**. Do not guess it. Ask the user.

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
| **Phase A — A5 Cohort & Metrics** | `scripts/analyze_text_eval_a5.py` (`193` vs `136` solvable vs `57` unsolvable cohorts + `seed 42` audit), `scripts/compute_metrics.py --bicubic-lr-dir` (`100`-image `RealSRVal` Bicubic $\times 4$ floor in `results/metrics_realsr_bicubic_x4_summary.csv`), and `docs/A5_baseline.tex` (`4`-page Overleaf build) | Member 1, 2, 4 | **Done (`VERIFIED` in `results/text_eval_a5_*` & `results/metrics_realsr_bicubic_x4_summary.csv`)** | None |
| **Phase A & B — Real-CE Split & LoRA 2/3 Pilot** | `scripts/split_realce_by_scene.py` (`0` scene overlap verified in Colab `CELL 04`), `src/lora_text_loss.py` (`6/6` CPU tests), `scripts/run_lora_trial.py`, `configs/lora{2,3}_pilot.yaml`, `notebooks/03_lora_pilot_colab.ipynb` (`400`-step GPU pilot + 10-seed val + M3 `--rec-only` OCR re-score & 5 crop grids) | Member 1 & Member 3 | **Done (`VERIFIED` on CPU & Colab L4 GPU in `results/lora_pilot_summary.json`, `results/pilot_lora2_vs_lora3_reconly/`)** | Mentor sign-off on next tuning sweep |
| **D-Report (`A4` & `A5`)** | `docs/A4_sota_limitations.tex` (submitted), `docs/fact_sheet_for_report.md` (`VERIFIED`), `docs/related_methods_ccsr_pnfs_tadisr.md`, and `docs/A5_baseline.tex` (due Tue `17/11 07:00`) | Member 4 & All | **A4 Done; A5 Complete & Verified (`docs/A5_baseline.tex`, zero pending items)** | Final Overleaf PDF export & portal upload |

## 4. Repo & Key Files

```bash
git clone https://github.com/lemon4life/SC203
cd SC203
git checkout main   # synced with feat/baseline-runner
```

| Path | What it is |
|---|---|
| `docs/fact_sheet_for_report.md` | **Master verified fact sheet for A4 & A5 reports:** dataset stats, hardware/VRAM profiles, 10-seed text evaluation tables, dedicated `193` vs `136` vs `57` cohort breakdown, `32.83%` disagreement formula, and `eduroam` `seed 42` provenance audit. |
| `docs/A5_baseline.tex` | Assignment 5 (`Baseline`) LaTeX report (`4` pages; all whole-image, Bicubic $\times 4$, `193` vs `136` vs `57` text tables, `32.83%` formula, and `eduroam` audit `VERIFIED`, zero pending items). |
| `docs/team_plan_A5_tasks_deadlines_dependencies.md` & `docs/plans/07_member1_gpu_pilot_and_realce_split_plan.md` | Active A5 & LoRA pilot task breakdown, timeline, dependencies, and Colab execution runbook. |
| `data/text_boxes_realsr.csv` | Merged 100-image ground-truth text box dataset (`276` rows, `100` images, `25` images with `readable=yes` text, `193` `readable=yes` boxes). Individual camera splits: `data/text_boxes_realsr_canon.csv` (`50` boxes) and `data/text_boxes_realsr_nikon.csv` (`143` boxes). |
| `data/realce_splits_manifest.json` | Scene-level `Real-CE x4` `train`/`val` split manifest (`0` scene overlap; validation boxes saved in `data/realce_val_boxes.csv`). |
| `results/text_eval_*.csv` & `results/text_eval_a5_*` | Complete 10-seed text evaluation CSVs (`Combined 100`, `Canon`, `Nikon`), plus `results/text_eval_a5_cohort_breakdown.csv` and `results/text_eval_a5_summary.json`. |
| `results/metrics_10seed_*.csv` & `results/metrics_realsr_bicubic_x4_summary.csv` | Whole-image 10-seed `PSNR`, `SSIM`, `LPIPS`, pairwise seed spread for `default`/`empty` prompts, and deterministic `Bicubic x4` floor (`100` images). |
| `scripts/analyze_text_eval_a5.py`, `scripts/text_eval.py`, `scripts/ocr_check_hr.py` | Standardized RapidOCR text evaluators (`--rec-only --rotate-vertical`) and A5 cohort + `seed 42` provenance analyzer. |
| `scripts/compute_metrics.py` | Full-reference PSNR, SSIM, LPIPS, pairwise seed spread, and `--bicubic-lr-dir` Bicubic $\times 4$ floor evaluator. |
| `scripts/split_realce_by_scene.py` & `scripts/check_realce_feasibility.py` | `Real-CE x4` feasibility auditor and leak-free scene-level `train`/`val` splitter + aligned patch extractor (`128x128` LR, `512x512` HR). |
| `src/lora_text_loss.py`, `scripts/run_lora_trial.py`, `configs/lora2_pilot.yaml`, `configs/lora3_pilot.yaml` | LoRA 2 (text-weighted) & LoRA 3 (text-weighted + $K=2$ cross-noise consistency) training loss, trial runner (supports `per_image_seed: true` 10-seed val inference), and pilot YAML configs. |
| `notebooks/01_dossr_baseline.ipynb`, `02_realce_feasibility_check.ipynb`, `03_lora_pilot_colab.ipynb` | Colab notebooks: `01` (`VERIFIED` RealSRVal baseline), `02` (`VERIFIED` Real-CE Drive staging & feasibility), `03` (`VERIFIED` Bicubic floor + Real-CE scene split + LoRA 2 vs. LoRA 3 L4 pilot in `results/lora_pilot_summary.json`). |
| `docs/handoff_member2.md`, `docs/handoff_member3.md`, `docs/handoff_member4.md` | Up-to-date handoff guides for Members 2, 3, and 4. |

## 5. Summary of Verified Findings (`CELL 01`–`18`, 100-Image `text_eval.py`, & A5 Audit)

1. **Colab GPU (`CELL 17` / `CELL 03`):** `NVIDIA L4` (`22.5 GiB` used for the 10-seed studies), PyTorch `2.11.0+cu130`, Python `3.13.16`.
2. **RealSR Dataset & Text Box Ground Truth (`CELL 04` + `data/text_boxes_realsr.csv`):**
   - `100` paired `test_LR` (`128x128`) and `test_HR` (`512x512`) images.
   - **`25 / 100` images** contain human-readable text (`10` Canon + `15` Nikon), totaling **`193` `readable=yes` word boxes** (`50` Canon + `143` Nikon).
3. **1-Image `test_LR` (`128x128 -> 512x512`) FP16 Profile (`CELL 08`–`09`):**
   - `SDPA ON`: `7.32 GiB` allocated (`8.29 GiB` reserved), `1.75 s/img`.
   - `SDPA OFF` (`--disable-sdpa`): `7.32 GiB` allocated (`8.28 GiB` reserved), `1.92 s/img`.
4. **Full 10-Seed $\times$ 2-Prompt Study & `Bicubic x4` Whole-Image Floor (`CELL 16`–`18`, `2,000` PNGs `VERIFIED`):**
   - `default` (`"clean, high-resolution, 8k"`) and `empty` (`""`) prompt studies (`1,000` `512x512` PNGs each across `seed_42`..`seed_51`, `SDPA OFF`, `fp16`) are verified on Drive.
   - Whole-image metrics (`results/metrics_10seed_*.csv` & `results/metrics_realsr_bicubic_x4_summary.csv`): `Bicubic x4` `PSNR_RGB = 25.4776 dB`, `SSIM_RGB = 0.7189`, `PSNR_Y = 27.1128 dB`, `SSIM_Y = 0.7561`, `LPIPS = 0.4563`; `default` `PSNR_RGB = 22.6353±0.5961 dB`, `SSIM_RGB = 0.6406±0.0182`, `PSNR_Y = 24.1610±0.6170 dB`, `SSIM_Y = 0.6820±0.0179`, `LPIPS = 0.3382±0.0191`; `empty` `PSNR_RGB = 22.6998±0.6079 dB`, `SSIM_RGB = 0.6431±0.0184`, `PSNR_Y = 24.2262±0.6295 dB`, `SSIM_Y = 0.6845±0.0181`, `LPIPS = 0.3352±0.0191`.
5. **10-Seed Text Restoration & Seed-Instability Cohort Breakdown (`VERIFIED` in `results/text_eval_a5_cohort_breakdown.csv`):**
   - **All `193` Labeled Boxes (`25` images):** `HR_CEILING = 70.5%` (`136/193`, `CER = 0.1648`); `BICUBIC_X4_FLOOR = 20.2%` (`39/193`, `Wrong-Char = 79.79%`, `CER = 0.7423`); DoSSR 10-seed (`default`, `1,930` box-seeds) = `24.0%` (`463/1930`), **`Wrong-Char = 76.01%`**, **`CER = 0.7127`**, **`Disagreement = 32.83%`** (`3.15` distinct readings / 10 seeds).
   - **`136` OCR-Solvable Subset (`20` images, where `HR CER = 0`):** `HR_CEILING = 100.0%`; `BICUBIC_X4_FLOOR = 28.7%` (`Wrong-Char = 71.32%`, `CER = 0.6626`); DoSSR 10-seed = `33.7%` (`458/1360`), **`Wrong-Char = 66.32%`**, **`CER = 0.6186`**, **`Disagreement = 34.05%`** (`3.22` distinct readings).
   - **`57` HR-Unsolvable Subset (`20` images):** `HR_CEILING = 0.0%` (`CER = 0.5579`); DoSSR 10-seed = `0.9%` (`5/570`), **`Wrong-Char = 99.12%`**, **`CER = 0.9370`**, **`Disagreement = 29.90%`** (`2.98` distinct readings).
   - **Paired Prompt Comparison (`empty - default`, `1,930` pairs):** $\Delta\text{Wrong-Char} = -0.16\%$, $\Delta\text{CER} = +0.0001$.
6. **Eduroam (`Canon_001`) `seed 42` Provenance (`VERIFIED`):** Earlier visual figures differed due to (a) upstream `inference.py` calling `pl.seed_everything(args.seed)` once per folder while `os.walk()` traverses files unsorted (1-image study drew 1st RNG vs. 100-image study drew a later PNG's RNG), and (b) OCR using tight word box `(320,38,192,46)` while figure scripts use full sign crop `(310,5,202,85)`. All canonical figures read from `seed-study-100img`.
7. **Controlled LoRA 2 vs. LoRA 3 Pilot on Scene-Disjoint `Real-CE x4` (`VERIFIED` Nov 15–16, `results/lora_pilot_summary.json`):**
   - `400` optimizer steps each (`alpha_text=5.0`, `lambda_cons=0.0` vs `0.5`, empty prompt `""`, $K=2$ noise draws, `1,659,904` trainable params, `0` scene overlap), evaluated on `30` val patches (`154` boxes) across seeds `42..51` (`5` steps):
   - **Compute & VRAM:** LoRA 2 `4.67 s/step` (`8.04 GiB` alloc); LoRA 3 `5.38 s/step` (`+15.20%`, **`8.04 GiB` alloc / `0.00 GiB` extra VRAM**).
   - **Whole-Patch & Stability (`LoRA 3 - LoRA 2`):** `PSNR_Y` `+0.5848 dB` (`17.6980 -> 18.2828`), `SSIM_Y` `+0.0300`, `LPIPS` `+0.0480` (`0.4231 -> 0.4711`), `pair_px` **`-57.50%`** (`9.2715 -> 3.9405`), `pair_lpips` **`-64.06%`** (`0.2020 -> 0.0726`).
   - **Corrected `--rec-only --rotate-vertical --upscale 2` OCR (`M3`, Nov 16):** Image-mean `wrong_char` `0.9141 -> 0.8878` (**`-0.0262 ± 0.0084`**), `CER` `0.6914 -> 0.6650` (**`-0.0264 ± 0.0122`**), `-0.0619` error rate (`-0.0340` CER) on `63` HR-solvable boxes, `seed_disagree` `0.5197 -> 0.4479` (**`-13.82%`**), `crop_px_l1` **`-55.08%`**. Mask alignment verified; `lambda_cons=0.5` across all $t \in [0, 999)$ caused stroke softening/erasure (`docs/M3_3_pilot_reconly_and_crops_notes.md`).

## 6. Immediate Next Steps

1. **Track 1 (`A5` Submission Before Tue `17/11 07:00`):** Finalize Section 4 analytical write-up in Overleaf ([docs/A5_baseline.tex](docs/A5_baseline.tex)), compile and verify `25125002_25125042_25125060_25125033.pdf`, and submit to the course portal.
2. **Track 2 (`Mentor Update & Pilot Tuning Gate`):** Send Dr. Thuận the update package covering (a) Real-CE $\times 4$ scene-disjoint split, (b) updated A5 baseline tables, and (c) LoRA 2 vs. LoRA 3 pilot tables (peak VRAM, `s/step`, `--rec-only` OCR deltas, and 5 visual crop grids), and request sign-off to sweep lower `lambda_cons` (`0.5 -> 0.1–0.2`) and/or restrict consistency to timesteps $t \le 500$ (`t1_fraction = 0.5`).
