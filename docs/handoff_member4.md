# Handoff Note for Member 4 (Nguyễn Phan Minh Nhật — `25125033`) — Report, Whole-Image Metrics & Figures Lead

**Branch:** `feat/report-figures` (branched from `feat/baseline-runner`)  
**Deadlines:**
- **Group Assignment 4 (`SOTA's Limitations`):** Official due **Tue `10/11/2026 07:00`**.
- **Group Assignment 5 (`Baseline`):** Official due **Tue `17/11/2026 07:00`** (Draft Fri `13/11`, internal submit Mon `16/11` night).

**Read First (`VERIFIED` Numbers & Tables for Report):**  
👉 **[`docs/fact_sheet_for_report.md`](file:///Users/lemon4life/Projects/SC203/docs/fact_sheet_for_report.md)** has all verified dataset stats (`100` images, `25` readable images, `193` word boxes), hardware/VRAM profiles (`7.32 GiB` at `512x512`), and the complete 100-image 10-seed text evaluation tables (`results/text_eval_*.csv`) from Member 2 (Gia Hân) and Member 3 (Quang Minh)!

---

## 1. Team Git & Communication Rules

1. **Work on your own branch:** `feat/report-figures`.
2. **Only edit your own files:**
   - Your report `.tex` files (`docs/A4_sota_limitations.tex`, `docs/A5_baseline.tex`)
   - `docs/related_methods_ccsr_pnfs_tadisr.md`
   - `scripts/compute_metrics.py`, `scripts/plot_eduroam_figure.py`, or any table/plot helper scripts you add
   - `results/metrics_*.csv` (summary CSVs from `compute_metrics.py`)
3. **Do NOT edit `context.md` or `external/`.** Do **not** commit LaTeX build artifacts (`*.aux`, `*.fdb_latexmk`, `*.fls`, `*.log`, `*.out`, `*.synctex.gz`, `*.toc`).

---

## 2. Python and Package Versions That Work

All scripts use `pathlib` and CLI flags (zero hardcoded `/content` paths) so they work on Windows, macOS, and Free Colab.

- **Python:** `3.9`, `3.10`, `3.11`, `3.12`, or `3.13`
- **Install command:**
  ```bash
  pip install "Pillow>=10.0.0" "numpy>=1.24.0" "torch>=2.0.0" "torchvision>=0.15.0" "lpips>=0.1.4"
  ```

---

## 3. Verified Shared Google Drive Paths (`2,000` 10-Seed PNGs Ready!)

- **HR ground-truth folder (`512x512`, 100 images — needed by `compute_metrics.py`):**
  `/content/drive/MyDrive/lemmepass/data/StableSR_testsets/StableSR_testsets/RealSRVal_crop128/test_HR`
- **LR input folder (`128x128`, 100 images — NOT needed by `compute_metrics.py`, only needed for figures):**
  `/content/drive/MyDrive/lemmepass/data/StableSR_testsets/StableSR_testsets/RealSRVal_crop128/test_LR`
- **10-Seed Default Prompt (`"clean, high-resolution, 8k"`, `10 seeds x 100 images = 1,000` `512x512` PNGs, `VERIFIED`):**
  `/content/drive/MyDrive/lemmepass/outputs/seed-study-realsr-100img-10seed-default-prompt-sdpa-off-fp16`
- **10-Seed Empty Prompt (`""`, `10 seeds x 100 images = 1,000` `512x512` PNGs, `VERIFIED`):**
  `/content/drive/MyDrive/lemmepass/outputs/seed-study-realsr-100img-10seed-empty-prompt-sdpa-off-fp16`
- **5-Image / 5-Seed Study (`VERIFIED`):**
  `/content/drive/MyDrive/lemmepass/outputs/seed-study-realsr-5img-lr128-fp16`

---

## 4. Exact Commands for `scripts/compute_metrics.py` (Run on Free Colab GPU in ~2 Minutes!)

> **Why you should run `compute_metrics.py` on Colab GPU (`--device cuda`) instead of laptop CPU:**
> For each image in a 10-seed folder, `compute_metrics.py` computes `10` LPIPS calls against `test_HR` **plus** $\binom{10}{2} = 45$ pairwise cross-seed LPIPS calls (`55` AlexNet passes per image $\times 100$ images = **`5,500` LPIPS passes per 10-seed folder**, or `11,000` across both folders). On a laptop CPU this takes **1–2 hours**; on a Free Colab T4 GPU (`--device cuda`) it takes **~1.5–3 minutes per folder** and now logs live `[i/100]` progress!
> *(Note: `compute_metrics.py` only needs `--output-root` (or `--study-root`) and `--gt-dir` (`test_HR`). It does **not** need `test_LR`.)*

### Command 1: 10-Seed Default Prompt (`1,000` PNGs vs `test_HR`)
```bash
python scripts/compute_metrics.py \
  --output-root "/content/drive/MyDrive/lemmepass/outputs/seed-study-realsr-100img-10seed-default-prompt-sdpa-off-fp16" \
  --gt-dir "/content/drive/MyDrive/lemmepass/data/StableSR_testsets/StableSR_testsets/RealSRVal_crop128/test_HR" \
  --device cuda \
  --per-image-csv "results/metrics_10seed_default_per_seed.csv" \
  --summary-csv "results/metrics_10seed_default_summary_by_image.csv"
```

### Command 2: 10-Seed Empty Prompt (`1,000` PNGs vs `test_HR`)
```bash
python scripts/compute_metrics.py \
  --output-root "/content/drive/MyDrive/lemmepass/outputs/seed-study-realsr-100img-10seed-empty-prompt-sdpa-off-fp16" \
  --gt-dir "/content/drive/MyDrive/lemmepass/data/StableSR_testsets/StableSR_testsets/RealSRVal_crop128/test_HR" \
  --device cuda \
  --per-image-csv "results/metrics_10seed_empty_per_seed.csv" \
  --summary-csv "results/metrics_10seed_empty_summary_by_image.csv"
```

### Optional Quick Check on Laptop CPU (1 Seed Only = `100` LPIPS calls, ~30–60 seconds):
If you run on your laptop CPU, pass `--seeds 42` first to test 1 seed quickly:
```bash
python scripts/compute_metrics.py \
  --output-root "<LOCAL_OR_DRIVE_DEFAULT_10SEED_ROOT>" \
  --gt-dir "<LOCAL_OR_DRIVE_TEST_HR>" \
  --seeds 42 \
  --device cpu
```

### Success Signal for `compute_metrics.py`:
1. Prints live progress `[1/100] Canon_001_LR4 | PSNR_RGB=... | SSIM_RGB=... | LPIPS=...` through `[100/100]`.
2. Prints the final `=== Per-Image Across-Seed Summary ===` table ending with the `ALL_IMAGES_MEAN` row.
3. Writes `metrics_10seed_{default,empty}_per_seed.csv` and `metrics_10seed_{default,empty}_summary_by_image.csv` into `results/`.

---

## 5. Exact Command for the `eduroam` Figure (`scripts/plot_eduroam_figure.py`)

`CELL 18` already generated and verified this figure on Colab with `--crop "310,5,202,85"` (`01dc1e2`), saved on Drive at:
`/content/drive/MyDrive/lemmepass/outputs/figures/canon_001_eduroam_default_vs_empty.png`

To regenerate it locally or on Colab:
```bash
python scripts/plot_eduroam_figure.py \
  --lr-image "/content/drive/MyDrive/lemmepass/data/StableSR_testsets/StableSR_testsets/RealSRVal_crop128/test_LR/Canon_001_LR4.png" \
  --hr-image "/content/drive/MyDrive/lemmepass/data/StableSR_testsets/StableSR_testsets/RealSRVal_crop128/test_HR/Canon_001_LR4.png" \
  --default-image "/content/drive/MyDrive/lemmepass/outputs/prompt-check-1img-lr128-default-sdpa-off-fp16/seed_42/Canon_001_LR4_0.png" \
  --empty-image "/content/drive/MyDrive/lemmepass/outputs/prompt-check-1img-lr128-empty-sdpa-off-fp16/seed_42/Canon_001_LR4_0.png" \
  --crop "310,5,202,85" \
  --output "outputs/figures/canon_001_eduroam_default_vs_empty.png"
```

---

## 6. What to Send Back to Member 1 (Nguyên Anh)

1. **For Assignment 4 (`SOTA's Limitations` — due Tue `10/11 07:00`):**
   - Your LaTeX report (`docs/A4_sota_limitations.tex` + compiled PDF) using the verified numbers in [`docs/fact_sheet_for_report.md`](file:///Users/lemon4life/Projects/SC203/docs/fact_sheet_for_report.md) and `results/text_eval_*.csv`.
   - The `ALL_IMAGES_MEAN` rows and `results/metrics_10seed_{default,empty}_summary_by_image.csv` from `scripts/compute_metrics.py`.
2. **For Assignment 5 (`Baseline` — due Tue `17/11 07:00`, freeze Mon `16/11 21:00`):**
   - Your 1-page related methods comparison (`docs/related_methods_ccsr_pnfs_tadisr.md`).
   - `docs/A5_baseline.tex` with all verified baseline numbers:
     * **193 vs. 136 Solvable Subset Table:** Report DoSSR, Bicubic, and HR ceiling separately on all 193 boxes and the 136 solvable boxes where HR OCR matches human transcript ($0.0\%$ HR error rate vs $66.3\%$ DoSSR error rate).
     * **Exact Mathematical Formula for 32.83% Cross-Seed Disagreement:** Include $\text{Disagreement}(b) = \frac{1}{45}\sum_{i<j} \mathbf{1}(T_i \ne T_j)$ and the box-weighted average $32.83\%$.
     * **Eduroam Crop Traceability Note:** Explicitly explain why seed 42 looks different across figures: word box crop ($192\times 46$) vs full sign crop ($202\times 85$ with Wi-Fi icon). Both come from `Canon_001_LR4_0.png`.
