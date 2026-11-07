# Handoff Note for Member 4 — Report, Whole-Image Metrics & Figures Lead

**Branch:** `feat/report-figures`  
**Hardware needed:** Any CPU laptop (Windows or Mac) or Free Google Colab (CPU). **No GPU required** for figures, tables, or the literature comparison (`compute_metrics.py` also runs on CPU with `--device cpu`).  
**Status of scripts:** `scripts/compute_metrics.py` is created locally (`NOT TESTED`). `scripts/plot_eduroam_figure.py` will be added in Group 3 (`NOT TESTED`).

---

## 1. Team Git & Communication Rules

1. **Work on your own branch:** `feat/report-figures`.
2. **Only edit your own files:**
   - `scripts/compute_metrics.py` (Task B.7)
   - `scripts/plot_eduroam_figure.py` (Task B.8, added in Group 3)
   - `scripts/plot_baseline_tables.py` (if you add table/plot helpers)
   - `docs/related_methods_ccsr_pnfs_tadisr.md`
   - `docs/handoff_member4.md`
3. **Do NOT edit `context.md` or `external/`.** Report your results in the team chat. Member 1 will update `context.md` and merge your Pull Request (PR).

---

## 2. Python and Package Versions That Work

All Member 4 scripts use `pathlib` and CLI flags (zero hardcoded `/content` paths) so they work on Windows, macOS, and Free Colab CPU.

- **Python:** `3.9`, `3.10`, `3.11`, `3.12`, or `3.13`
- **For figures and tables (`plot_eduroam_figure.py`, CSV summaries):**
  - `Pillow >= 10.0.0`
  - `numpy >= 1.24.0`
  - `matplotlib >= 3.7.0` (optional)
- **For `scripts/compute_metrics.py` (PSNR, SSIM, LPIPS on CPU or GPU):**
  - `Pillow >= 10.0.0`, `numpy >= 1.24.0`, `torch >= 2.0.0`, `torchvision >= 0.15.0`, `lpips >= 0.1.4`
- Install command:
  ```bash
  pip install "Pillow>=10.0.0" "numpy>=1.24.0" "torch>=2.0.0" "torchvision>=0.15.0" "lpips>=0.1.4"
  ```

---

## 3. How to Reach the Data on Shared Google Drive

- **HR ground-truth folder (`512x512`, 100 images):**
  `lemmepass/data/StableSR_testsets/StableSR_testsets/RealSRVal_crop128/test_HR`
- **LR input folder (`128x128`, 100 images):**
  `lemmepass/data/StableSR_testsets/StableSR_testsets/RealSRVal_crop128/test_LR`
- **Already-completed 5-image / 5-seed outputs on Drive (`VERIFIED`, `512x512`):**
  `lemmepass/outputs/seed-study-realsr-5img-lr128-fp16`
- **Upcoming 10-seed (`default` vs `empty` prompt) outputs on Drive (when Member 1 finishes running Group A):**
  - `lemmepass/outputs/seed-study-realsr-100img-10seed-default-prompt-sdpa-off-fp16`
  - `lemmepass/outputs/seed-study-realsr-100img-10seed-empty-prompt-sdpa-off-fp16`

You can download those folders from Google Drive to your laptop, or mount Drive in Free Colab (`/content/drive/MyDrive/lemmepass/...`) and pass the paths via CLI flags.

---

## 4. What Is Blocked vs. What You Can Do Right Now

- **What you can do right now (Not Blocked):**
  1. **Run `scripts/compute_metrics.py` on the existing 5-image / 5-seed outputs** (`seed-study-realsr-5img-lr128-fp16`) to verify PSNR, SSIM, and LPIPS on CPU or Free Colab.
  2. **Run `scripts/plot_eduroam_figure.py` on `Canon_001_LR4.png`** (`CELL 16` outputs for `default` and `empty` prompts are now **VERIFIED and saved on Drive** under `lemmepass/outputs/prompt-check-1img-lr128-{default,empty}-sdpa-off-fp16/seed_42/Canon_001_LR4_0.png`!).
  3. **Fill in the one-page comparison with CCSR, PNFS, and TADiSR** in [`docs/related_methods_ccsr_pnfs_tadisr.md`](file:///Users/lemon4life/Projects/SC203/docs/related_methods_ccsr_pnfs_tadisr.md).
     - *Important rule:* Do **not** guess or invent paper facts. Only write facts verified from the actual CCSR, PNFS, and TADiSR papers, or mark `TODO: confirm`.
- **What waits for Member 1 and Member 2:**
  - Final 10-seed whole-image metrics table waits for Member 1's 10-seed runs (`CELL 17`).
  - Final `eduroam` crop coordinates `(x, y, w, h)` in `data/text_boxes_realsr.csv` will be confirmed by Member 1 / Member 2.

---

## 5. Exact Commands and How to Check They Worked

### Step 1: Clone the repo and create your branch
```bash
git clone https://github.com/lemon4life/SC203
cd SC203
git checkout feat/baseline-runner
git checkout -b feat/report-figures
```

### Step 2: Run `scripts/compute_metrics.py` on the 5-seed study (or 10-seed study later)
Replace `<STUDY_ROOT>` and `<PATH_TO_TEST_HR>` with your local or Drive paths:

```bash
python scripts/compute_metrics.py \
  --study-root "<STUDY_ROOT>" \
  --gt-dir "<PATH_TO_TEST_HR>" \
  --device cpu
```

### Step 3: Check that `compute_metrics.py` worked (Success Signal)
1. The terminal prints a summary table with `PSNR (dB)`, `SSIM`, `LPIPS (alex)`, and `Pairwise Seed Diff (0..255)` for each image stem, plus `[OVERALL MEAN ACROSS IMAGES]`.
2. Two CSV files are created inside `<STUDY_ROOT>`:
   - `metrics_per_seed.csv`
   - `metrics_summary_by_image.csv`
3. Every image is verified as `512x512` (the script stops with an error if any output or HR image is not `512x512`).

### Step 4: Run `scripts/plot_eduroam_figure.py` (Task B.8, `NOT TESTED`)
Replace `<DATA_ROOT>` and `<OUT_BASE>` with your local or Drive paths, and `<X,Y,W,H>` with the confirmed crop box (for example `310,5,202,85` for the sign + text, or `325,22,185,60` tightly around the letters):
```bash
python scripts/plot_eduroam_figure.py \
  --lr-image "<DATA_ROOT>/test_LR/Canon_001_LR4.png" \
  --hr-image "<DATA_ROOT>/test_HR/Canon_001_LR4.png" \
  --default-image "<OUT_BASE>/prompt-check-1img-lr128-default-sdpa-off-fp16/seed_42/Canon_001_LR4_0.png" \
  --empty-image "<OUT_BASE>/prompt-check-1img-lr128-empty-sdpa-off-fp16/seed_42/Canon_001_LR4_0.png" \
  --crop "310,5,202,85" \
  --output "outputs/figures/canon_001_eduroam_default_vs_empty.png"
```
- **Success signal:** Prints `EDUROAM FIGURE SAVED | crop=(310, 5, 202, 85) | path=...` and saves the 4-panel PNG (`LR`, `HR`, `Default Prompt`, `Empty Prompt` with zoomed text region).

---

## 6. What to Send Back to Member 1

1. When you run `scripts/compute_metrics.py` on `seed-study-realsr-5img-lr128-fp16`: send the terminal output and the two CSV files (`metrics_per_seed.csv` and `metrics_summary_by_image.csv`) in the team chat so we can mark `compute_metrics.py` as `VERIFIED`.
2. Your one-page comparison draft (`docs/related_methods_ccsr_pnfs_tadisr.md`) via a Pull Request from `feat/report-figures` to `feat/baseline-runner`.
3. If a command fails: send the exact command you ran and the last 50 lines / full traceback.
