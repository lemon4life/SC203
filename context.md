# context.md: read this first

You are joining Team 1 (`LEMMEPASS`) in SC203 Scientific Methods (Class 25A02). This file tells you what the project is, what is done, what is not, and what to do next. Facts below come from our real logs (Oct 29 – Nov 9, 2026), direct source inspection of `external/DOSSR`, and verified team evaluation CSVs in `data/` and `results/`. Anything not confirmed is marked **TODO: confirm**. Do not guess it. Ask the user.

## 1. Project, Deadlines & Mentor Direction (Nov 2026)

- **Topic 3:** Seed-robust real-world super-resolution with text-region consistency.
- **Base model:** DoSSR (public code and pretrained checkpoint).
- **Updated Research Question (Nov 2026 Mentor Direction):**
  - *"Does adding a loss during LoRA training help DoSSR restore text correctly more consistently when the seed changes?"*
  - The baseline is tentatively accepted. The instruction-conditioning part (InstructIR / prompt injection) is **deferred** until the text-region consistency loss is validated. **Do not work on prompts/InstructIR/text injection now.**
- **Deadlines:**
  - **Group Assignment 4 (`SOTA's Limitations`):** Official due **Tue `10/11/2026 07:00`** (Internal submit Mon `09/11/2026` night). Shows DoSSR's limitation on text restoration accuracy and seed instability across 10 seeds (`42..51`).
  - **Group Assignment 5 (`Baseline`):** Official due **Tue `17/11/2026 07:00`** (Draft Fri `13/11`, internal submit Mon `16/11` night).

## 2. Confirmed Team Roster (4 Members), Branches & PR Rule

- **Merge & Status Rule:** **Members 2, 3, and 4 must NOT edit `context.md` directly.** Members report progress in the team chat, and Member 1 (via the agent) updates `context.md` and merges teammate branches into `feat/baseline-runner`.

| Member | Full Name (`Student ID`) | Role | Branch | Owned Files & Deliverables | Hardware |
|---|---|---|---|---|---|
| **Member 1 (User)** | **Hoàng Nguyên Anh (`25125002`)** | GPU & Pipeline Lead | `feat/baseline-runner` | `src/run_dossr.py`, `src/dossr_inference_compat.py`, `src/attention_compat.py`, `scripts/run_seed_study.py`, `scripts/check_study_outputs.py`, `scripts/merge_text_boxes.py`, `configs/*`, `notebooks/01_dossr_baseline.ipynb`, `context.md` | Colab Pro GPU (`NVIDIA L4`) + local Mac |
| **Member 2** | **Huỳnh Trần Gia Hân (`25125042`)** | Text Labeling (`1–50`) & Text Evaluation Lead | `feat/text-labels-canon` (merged) | `data/text_boxes_realsr_canon.csv` (`_m2.csv`), `results/text_eval_10seed_canon/*.csv`, `scripts/make_text_contact_sheets.py`, `docs/handoff_member2.md` | CPU laptop / Free Colab |
| **Member 3** | **Hồ Quang Minh (`25125060`)** | Text Labeling (`51–100`), OCR Calibration & LoRA Lead | `feat/text-labels-nikon`, `fix/ocr-rec-only`, `feat/lora-loss` (merged) | `data/text_boxes_realsr_nikon.csv` (`_m3.csv`), `scripts/ocr_check_hr.py`, `scripts/text_eval.py` (`--rec-only --rotate-vertical`), `results/text_eval_nikon_*.csv`, `docs/lora_feasibility_and_loss_design.md`, `docs/handoff_member3.md` | CPU laptop / Free Colab (LoRA paused until Tue `11/11/2026`) |
| **Member 4** | **Nguyễn Phan Minh Nhật (`25125033`)** | Report (`A4`/`A5`), Whole-Image Metrics & Related Work Lead | `feat/report-figures` | `docs/A4_sota_limitations.tex`, `docs/A5_baseline.tex`, `docs/related_methods_ccsr_pnfs_tadisr.md`, `scripts/compute_metrics.py`, `scripts/plot_eduroam_figure.py`, `docs/handoff_member4.md` | CPU laptop / Free Colab GPU (`--device cuda`) |

## 3. Task Status & Dependency Table

> **Rule:** Only Member 1 / the agent updates this table. Everything stays `NOT TESTED` until Colab or teammate execution logs are sent back.

| Task | Description | Owner | Status | Blocked By |
|---|---|---|---|---|
| **B.5** | Contact sheet generator (`scripts/make_text_contact_sheets.py`), validator (`scripts/check_text_boxes.py`), & merger (`scripts/merge_text_boxes.py`) | Member 1, 2, 3 | **Done (`VERIFIED` on all 100 images)** | None |
| **B.5-Label** | Label all 100 `RealSRVal_crop128/test_HR` images (`1–50` Canon by Member 2; `51–100` Nikon by Member 3) and merge into `data/text_boxes_realsr.csv` | Member 2 & Member 3 & Member 1 | **Done (`VERIFIED`: `100` images, `25` images with `readable=yes` text [`10` Canon + `15` Nikon], `193` `readable=yes` boxes [`50` Canon + `143` Nikon], `0` errors)** | None |
| **B.6-OCR** | Check OCR on sharp `test_HR` crops (`scripts/ocr_check_hr.py`) and add `--rec-only --rotate-vertical` (`ca127fe`) | Member 3 | **Done (`VERIFIED`: boosts `rapidocr` HR ceiling on Nikon from `18/143 = 12.6%` to `101/143 = 70.6%` [`1.2.3`] / `108/143 = 75.5%` [`1.4.4` Colab])** | None |
| **B.6-Eval** | Run 10-seed `scripts/text_eval.py` across all 100 images (`25` readable images, `193` boxes, `1,930` box-seed crops, `HR_CEILING`, `BICUBIC_X4_FLOOR`, `default` vs `empty` prompt) | Member 2 & Member 3 | **Done (`VERIFIED` in `results/text_eval_*.csv`: `HR_CEILING = 70.5%` exact [`CER = 0.1648`], `BICUBIC_X4_FLOOR = 20.2%` [`CER = 0.7423`], `DoSSR 10-seed = 24.0%` [`CER = 0.7127`], `Cross-Seed OCR Disagreement = 32.83%`, `3.15` distinct readings/10 seeds, `Paired empty-default dWCR = -0.16%, dCER = +0.0001`)** | None |
| **C.9 & C.10** | Read-only audit of DoSSR training code (`docs/lora_feasibility_and_loss_design.md`, updated in `8565c93`) & LoRA trial design | Member 3 | **Done (`VERIFIED` in `external/DOSSR` source; runtime estimates `ESTIMATE — NOT TESTED`; LoRA coding paused until Tue `11/11/2026`)** | Paused until `11/11/2026` |
| **A.1–A.4-Run** | 100-image $\times$ 10-seed $\times$ 2-prompt (`default` vs `empty`, `SDPA OFF`, `fp16`) study (`CELL 16`–`17b`) | Member 1 | **Done (`VERIFIED` on `131e636`: `2,000` `512x512` PNGs on Drive, all 20 manifests `fp16` + `SDPA OFF`)** | None |
| **B.8** | `Canon_001_LR4` (`eduroam`) 4-panel figure (`scripts/plot_eduroam_figure.py` & `CELL 18`) | Member 1 & Member 4 | **Done (`VERIFIED` in `CELL 18` with `EDUROAM_CROP = '310,5,202,85'`, `01dc1e2`)** | None |
| **B.7** | Whole-image metrics (`scripts/compute_metrics.py`: PSNR, SSIM, LPIPS, pairwise seed spread) | Member 4 | **Done (`VERIFIED` on all 100 images $\times$ 10 seeds in `7564c85`: `default` `PSNR_RGB = 22.6353±0.5961`, `SSIM_RGB = 0.6406±0.0182`, `LPIPS = 0.3382±0.0191`, `pair_px = 10.4508`; `empty` `PSNR_RGB = 22.6998±0.6079`, `SSIM_RGB = 0.6431±0.0184`, `LPIPS = 0.3352±0.0191`, `pair_px = 10.3415`)** | None |
| **D-Report** | `docs/fact_sheet_for_report.md` (`VERIFIED`), `docs/main.tex` / `docs/A4_sota_limitations.tex` (due Tue `10/11 07:00`), 1-page CCSR/PNFS/TADiSR comparison (`docs/related_methods_ccsr_pnfs_tadisr.md`), and `A5` report (due Tue `17/11 07:00`) | Member 4 & All | **Assignment 4 Report & Related Methods Done (`VERIFIED` in `7564c85` + `docs/main.tex` / `docs/A4_sota_limitations.tex`)** | None (A5 due `17/11`) |

## 4. Repo & Key Files

```bash
git clone https://github.com/lemon4life/SC203
cd SC203
git checkout feat/baseline-runner   # main integration branch managed by Member 1
```

| Path | What it is |
|---|---|
| `docs/fact_sheet_for_report.md` | **Master verified fact sheet for A4 & A5 reports:** all verified dataset stats, hardware/VRAM profiles, 10-seed text evaluation tables (`Canon`, `Nikon`, and `Combined 100`), concrete `eduroam` / `SIEMENS` / `Nikon_002` examples, and `PENDING` checklist. |
| `data/text_boxes_realsr.csv` | Merged 100-image ground-truth text box dataset (`276` rows, `100` images, `25` images with `readable=yes` text, `193` `readable=yes` boxes). Individual camera splits: `data/text_boxes_realsr_canon.csv` (`_m2.csv`, `50` boxes) and `data/text_boxes_realsr_nikon.csv` (`_m3.csv`, `143` boxes). |
| `results/text_eval_*.csv` | Complete 10-seed text evaluation results (`summary_by_image`, `paired_comparison_by_image`, `hr_ceiling_by_image`, `hr_ceiling_per_box`, `bicubic_floor_by_image`, `bicubic_floor_per_box`, `per_box_seed`) for Combined 100 (`results/text_eval_*.csv`), Canon (`results/text_eval_10seed_canon/`), and Nikon (`results/text_eval_nikon_*.csv`). |
| `scripts/text_eval.py`, `scripts/ocr_check_hr.py` | Text restoration & HR ceiling evaluators supporting `--ocr-backend {rapidocr,easyocr,tesseract}`, `--rec-only`, `--rotate-vertical`, `--upscale {1,2,4}`, `--hr-dir`, `--lr-dir`, and `--seeds`. **`VERIFIED` on all 100 images (`1,930` box-seed crops).** |
| `scripts/check_text_boxes.py`, `scripts/merge_text_boxes.py`, `scripts/check_study_outputs.py` | CPU-only validators for text-box CSVs (`--range`), CSV merging, and 10-seed study output folders. |
| `scripts/compute_metrics.py` | Full-reference PSNR, SSIM, LPIPS, and pairwise cross-seed spread (`--output-root` / `--study-root`, `--gt-dir`, `--device auto|cpu|cuda`) with live `[i/100]` progress logging. |
| `notebooks/01_dossr_baseline.ipynb` | Standardized team Colab notebook: **`CELL 01`–`CELL 18` (including `CELL 17b`) are VERIFIED** (`251d4a5`, `810e844`, `131e636`, `01dc1e2`). |
| `docs/handoff_member2.md`, `docs/handoff_member3.md`, `docs/handoff_member4.md` | Up-to-date handoff guides for Members 2, 3, and 4. |
| `docs/ke_hoach_nhom_A4_A5.md`, `docs/plans/05_a4_a5_team_support.md` | Team coordination plan for Group Assignments 4 and 5. |

## 5. Summary of Verified Findings (`CELL 01`–`18` & 100-Image `text_eval.py`)

1. **Colab GPU (`CELL 17` / `CELL 03`):** `NVIDIA L4` (`22.5 GiB`, used for the 10-seed studies; `A100` was only used on the Nov 5 `2048x2048` discovery run), PyTorch `2.11.0+cu130`, Python `3.13.16`.
2. **RealSR Dataset & Text Box Ground Truth (`CELL 04` + `data/text_boxes_realsr.csv`):**
   - `100` paired `test_LR` (`128x128`) and `test_HR` (`512x512`) images.
   - **`25 / 100` images** contain human-readable text (`10` Canon + `15` Nikon), totaling **`193` `readable=yes` word boxes** (`50` Canon + `143` Nikon).
   - Because `25` images (`193` boxes $\times 10$ seeds = `1,930` box-seed evaluations) provide strong statistical power, `Real-CE x4` is not required for Assignment 4.
3. **1-Image `test_LR` (`128x128 -> 512x512`) FP16 Profile (`CELL 08`–`09`):**
   - `SDPA ON`: `7.32 GiB` allocated (`8.29 GiB` reserved), `1.75 s/img`.
   - `SDPA OFF` (`--disable-sdpa`): `7.32 GiB` allocated (`8.28 GiB` reserved), `1.92 s/img`.
4. **Full 10-Seed $\times$ 2-Prompt Study (`CELL 16`–`18`, `2,000` PNGs `VERIFIED`):**
   - `default` (`"clean, high-resolution, 8k"`) and `empty` (`""`) prompt studies (`1,000` `512x512` PNGs each across `seed_42`..`seed_51`, `SDPA OFF`, `fp16`) are verified on Drive.
5. **10-Seed Text Restoration & Seed-Instability Results (`VERIFIED` in `results/text_eval_*.csv`):**
   - **`HR_CEILING` (`test_HR` 512$\times$512):** `136 / 193` (**`70.5%`**) exact box match (`Canon: 70.0%`, `Nikon: 70.6%` / `75.5%` on Colab), `wrong_char_rate = 29.5%`, `CER = 0.1648`.
   - **`BICUBIC_X4_FLOOR` (`test_LR` $\times 4$):** `39 / 193` (**`20.2%`**) exact box match (`Canon: 20.0%`, `Nikon: 20.3%`), `wrong_char_rate = 79.8%`, `CER = 0.7423`.
   - **DoSSR 10-Seed (`default` prompt, `1,930` box-seeds):** `463 / 1930` (**`24.0%`**) exact match (`wrong_char_rate = 76.0%` box-weighted / `68.2%` image-mean; `CER = 0.7127` box-weighted / `0.6451` image-mean).
   - **Seed Instability Across `seed_42`..`seed_51`:** **`32.83%`** cross-seed OCR disagreement (`31.57%` image-mean), **`3.15` distinct OCR transcripts per box across 10 seeds** (`10.0 / 10` on `Canon_001` `eduroam` and `Nikon_002`), and **`20.28` pixel $L_1$ spread** (`0..255`).
   - **Paired Prompt Comparison (`empty - default`, `1,930` pairs):** $\Delta\text{WCR} = -0.16\%$, $\Delta\text{CER} = +0.0001$ (confirming text hallucination is driven by the diffusion prior and random seed rather than the positive prompt).

## 6. Remaining Open Questions (`TODO: confirm` for Assignment 5, after Tue `11/11/2026`)

1. **LoRA Training Data (`C.9/C.10`, for Assignment 5 after `11/11/2026`):** Which disjoint HR dataset with text boxes (`Real-CE`, ICDAR/TextOCR + Real-ESRGAN degradation, or RealSR train split) will we use for the short trial run of LoRA version `C`?
2. **LoRA Prompt Mode (`C.9/C.10`):** Confirm training and evaluating LoRA `A, B, C` with the empty prompt `""` (matching `realesrgan.py:179`).
