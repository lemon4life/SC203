# Plan 05 — Group Assignment 4 (SOTA's Limitations) & Assignment 5 (Baseline) Team Support

**Date:** `2026-11-08`  
**Team:** Team 1 (`LEMMEPASS`, SC203 Scientific Methods — Class 25A02)  
- **Member 1 (`Hoàng Nguyên Anh`, `25125002`):** GPU & Pipeline Lead, CSV merge, final PDF assembly & submission.  
- **Member 2 (`Huỳnh Trần Gia Hân`, `25125042`):** Text labeling (`images 1–50` in `data/text_boxes_realsr_m2.csv`) and `scripts/text_eval.py` scoring.  
- **Member 3 (`Hồ Quang Minh`, `25125060`):** Text labeling (`images 51–100` in `data/text_boxes_realsr_m3.csv`), OCR verification on sharp `test_HR` crops (`scripts/ocr_check_hr.py`), and `text_eval.py` debugging. (LoRA paused until Tue `11/11/2026`.)  
- **Member 4 (`Nguyễn Phan Minh Nhật`, `25125033`):** LaTeX report (`docs/A4_sota_limitations.tex`, `docs/A5_baseline.tex`), whole-image metrics (`scripts/compute_metrics.py` on CPU), and 1-page related work (`docs/related_methods_ccsr_pnfs_tadisr.md`).

---

## 1. Deadlines & Schedule
- **Group Assignment 4 (SOTA's Limitations):**
  - Sun `08/11 21:00`: Member 2 (`1–50`) & Member 3 (`51–100`) finish text box CSVs and image counts; Member 4 finishes LaTeX skeleton & setup section.
  - Sun `08/11 21:30`: Member 1 merges `_m2` and `_m3` into `data/text_boxes_realsr.csv` via `scripts/merge_text_boxes.py`; Member 3 selects OCR backend & upscale factor (`1x`, `2x`, `4x`) via `scripts/ocr_check_hr.py`.
  - Mon `09/11 12:00`: Member 3 verifies `text_eval.py` on `seed_42`; Member 4 finishes 1-page CCSR/PNFS/TADiSR section.
  - Mon `09/11 15:00`: Member 2 finishes 10-seed `text_eval.py` scores; Member 4 finishes 10-seed PSNR/SSIM/LPIPS table.
  - Mon `09/11 18:00`: Data freeze.
  - Mon `09/11 21:00`: Proofread complete.
  - **Mon `09/11 22:00`: Internal submit for Assignment 4** (Official course deadline: **Tue `10/11/2026 07:00`**).
- **Group Assignment 5 (Baseline):**
  - Tue `11/11`: Member 3 resumes LoRA work.
  - Fri `13/11`: Assignment 5 draft ready.
  - Mon `16/11` night: Internal submit for Assignment 5 (Official course deadline: **Tue `17/11/2026 07:00`**).

---

## 2. Ordered Task Groups (Stop & Report After Each Group)

### Step 0: Orientation & Audit (Completed with Group 1)
- Save prompt in `prompts/2026-11-08_a4_a5_team_support.md` and Vietnamese plan in `docs/ke_hoach_nhom_A4_A5.md`.
- Ignore LaTeX build artifacts (`*.aux`, `*.fdb_latexmk`, `*.fls`, `*.synctex.gz`, `*.out`, `*.toc`, `*.log`) in `.gitignore` while keeping `!data/text_boxes_*.csv` unignored.
- Read-only audit of `docs/Week01.tex` against `docs/after_01_dossr_baseline_discovery.md` (report contradictions without modifying `docs/Week01.tex`).

### Group 1: Unblock the Labelers First (`NOT TESTED` until logs returned)
- `1a.` `scripts/check_text_boxes.py`: CPU-only validator for labeler CSVs (`image,box_id,x,y,w,h,transcript,readable`), checking integer bounds within `512x512`, `w > 0`, `h > 0`, `readable in {yes, no}` (no blanks), non-empty `transcript` when `readable=yes`, no duplicate `(image, box_id)`, image existence in `test_HR`, coverage of the labeler's assigned index range (`--range 1-50` or `51-100`), and optional `--preview-dir` to draw boxes on HR images.
- `1b.` `scripts/merge_text_boxes.py`: CPU-only merger combining `data/text_boxes_realsr_m2.csv` (`1–50`) and `data/text_boxes_realsr_m3.csv` (`51–100`) into `data/text_boxes_realsr.csv`, failing on any missing/duplicate image, blank `readable`, or overlap between `_m2` and `_m3`, and printing total images with `readable=yes` text.
- `1c.` Update `docs/handoff_member2.md` and `docs/handoff_member3.md` in place with the new A4 tasks, image ranges (`1–50` vs `51–100`, shared sheet 3 split `41–50` vs `51–60`), exact commands, success signals, what to send back, and 3–4 clearly marked synthetic worked examples (`eduroam` box left as `TODO` until user sends the exact text-only crop box).

### Group 2: Member 3's OCR Check (`NOT TESTED`)
- `2a.` `scripts/ocr_check_hr.py`: Crop each `readable=yes` box from `test_HR`, optionally upscale (`--upscale 1|2|4`), run `--ocr-backend {rapidocr,easyocr,tesseract}`, normalize with `text_eval.py` rules, print exact-match rate (`exact_right / total_boxes`), and write per-box CSV.
- `2b.` Update `scripts/text_eval.py` to support `--upscale {1,2,4}`, `--hr-dir` (HR ceiling on `test_HR` crops), and `--lr-dir` (bicubic $\times 4$ floor on `test_LR` crops), and verify all CLI flags via `--help` against `docs/handoff_member2.md` and `docs/handoff_member3.md`.

### Group 3: Checks for Member 1 (`NOT TESTED`)
- `3a.` `scripts/check_study_outputs.py`: CPU-only verifier for `--root <output_root>` checking 10 seed folders (`seed_42`..`seed_51`), 100 PNGs each (`512x512`), and `run_manifest.json` (`precision == "fp16"`, `sdpa_enabled == False`, `disable_sdpa == True`, and expected `--prompt-mode`).

### Group 4: Member 4's Kit (`NOT TESTED`)
- `4a.` `docs/fact_sheet_for_report.md`: Every verified/estimated/pending number with exact log/CELL/doc citation.
- `4b.` `docs/A4_sota_limitations.tex` and `docs/A5_baseline.tex` using `docs/Week01.tex` preamble and `docs/ke_hoach_nhom_A4_A5.md` Section 9 outline.
- `4c.` Update `docs/related_methods_ccsr_pnfs_tadisr.md` with extraction checklist and mandatory page/section citation rule.
- `4d.` Update `docs/handoff_member4.md` with the exact CPU `compute_metrics.py` commands using the real `output_root` paths from `configs/seed_study_realsr_100img_10seed_{default,empty}_prompt_sdpa_off.yaml`.

### Group 5: Setup for Everyone (`NOT TESTED`)
- `5a.` Role-specific requirements files (`requirements-text-eval.txt`, `requirements-metrics-figures.txt`) with Windows/macOS `pip` commands.
- `5b.` `docs/data_access.md`: Shared Google Drive layout, permissions, Drive for Desktop / Free Colab instructions, and real `output_root` paths.
- `5c.` Vietnamese handoff notes (`docs/handoff_member2.vi.md`, `docs/handoff_member3.vi.md`, `docs/handoff_member4.vi.md`) marked `TODO: review by a Vietnamese speaker`.
- `5d.` Optional `notebooks/02_member_starter.ipynb` (CPU-friendly starter notebook).

### Group 6: Final Documentation Sync (`COMPLETED` in `feat/baseline-runner`)
- `6a.` Update `context.md` with student names/IDs, A4/A5 schedule, status table, dependency graph, and open questions (**Done**).
- `6b.` Add top note to `docs/phase2_plan.md` pointing to `docs/ke_hoach_nhom_A4_A5.md` (**Done**).
- `6c.` Update `docs/plans/05_a4_a5_team_support.md` with completion status (**Done**).
- `6d.` Document `results/` convention for small summary CSVs vs. Drive for large outputs in `results/README.md` (**Done**).

