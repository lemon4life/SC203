# Plan 06b — Group Assignment 5 (`Baseline` Report) & Parallel LoRA Trial Plan

**Date:** Tuesday `2026-11-10`  
**Team:** Team 1 (`LEMMEPASS`, SC203 Scientific Methods — Class 25A02)  
- **Member 1 (`Hoàng Nguyên Anh`, `25125002`):** GPU & Pipeline Lead (`feat/baseline-runner`).  
- **Member 2 (`Huỳnh Trần Gia Hân`, `25125042`):** Text Evaluation & Statistical Refinement Lead (`feat/text-eval-a5`).  
- **Member 3 (`Hồ Quang Minh`, `25125060`):** LoRA & Text-Region Consistency Loss Lead (`feat/lora-loss`).  
- **Member 4 (`Nguyễn Phan Minh Nhật`, `25125033`):** Baseline Report (`docs/A5_baseline.tex`) & Figures/Tables Lead (`feat/report-a5`).

---

## 1. Strict Decoupling: Assignment 5 (`Baseline`) vs. Mentor LoRA Trial

1. **Group Assignment 5 (`A5` = Baseline Report, Official Due: Tuesday `17/11/2026 07:00`):**
   - **Scope:** Reproducible DoSSR baseline on `RealSRVal_crop128/test_LR` (`128x128 -> 512x512`), implementation & runtime compatibility patches (`SDPA`, prompt wrapper, FP16 manifest guard), 10-seed $\times$ 2-prompt whole-image metrics (`PSNR`, `SSIM`, `LPIPS`, cross-seed spread), 10-seed text restoration & seed-instability evaluation (strengthened with the A4-review statistical analyses), and full reproducibility instructions.
   - **Rule:** **`A5` must NOT depend on the LoRA trial.** `A5` stands on its own as a complete baseline report even if zero LoRA training runs are finished by `16/11`. Mention LoRA in `A5` **only** as a short future-work / preliminary-feasibility paragraph if (and only if) the trial run is already completed and verified before the `A5` freeze.
2. **Parallel Track — LoRA Trial & `Real-CE` Feasibility (For the Mentor's Next Progress Report):**
   - **Scope:** `Real-CE` (`4x`) feasibility audit, staged CPU-tested implementation of `src/lora_text_loss.py` and `scripts/run_lora_trial.py` (`NOT TESTED`), and Colab GPU smoke/trial runs (`Variant A` 20 steps $\rightarrow$ `Variant C` 50–100 steps with VRAM & step time logged).
   - Runs in parallel on `feat/lora-loss` without blocking `docs/A5_baseline.tex`.

---

## 2. Corrected Calendar & Schedule (`Tue 10/11/2026` – `Tue 17/11/2026`)

| Day & Date | Track 1: Assignment 5 (`Baseline` Report — Mandatory for `17/11`) | Track 2: LoRA Trial & `Real-CE` Audit (Parallel for Mentor Report) | Owner(s) |
| :--- | :--- | :--- | :--- |
| **Tue `10/11`** *(Today)* | Revise plan (`06b`); scaffold `docs/A5_baseline.tex` (`VERIFIED` vs `PENDING` + number-audit checklist). | Scaffold `src/lora_text_loss.py`, `scripts/run_lora_trial.py`, CPU unit tests (`a`–`f`), and `docs/lora_code_walkthrough.md` (`NOT TESTED`). | Member 1 (Agent) |
| **Wed `11/11`** | **M2:** Run unified 100-image `text_eval.py` (single RapidOCR version & flags across all 193 boxes, including full **empty-prompt 10-seed row**).<br>**M4:** Document exact metric definitions (`PSNR/SSIM/LPIPS` library, channels, border, network) and start `docs/A5_baseline.tex` Sections 1–3. | **M3:** Run CPU unit tests `(a)`–`(f)` on `src/lora_text_loss.py`.<br>**M1 & M2:** Run `Real-CE` (`4x`) feasibility check (download, counts, sizes, annotation format, license, HR OCR check, 3 pairs through DoSSR). | M1, M2, M3, M4 |
| **Thu `12/11`** | **M2:** Compute A4-review statistical analyses on the unified 10-seed CSVs: (1) OCR-solvable subset (`HR CER = 0`), (2) image-level bootstrap 95% CIs, (3) capped CER $\min(\text{CER}, 1.0)$ & median CER, (4) non-text background control boxes for pixel seed spread (`21:00`).<br>**M4:** Integrate new tables/figures into `docs/A5_baseline.tex`. | **M3 & M1:** Fix validation protocol before training. If `Real-CE` check passes, run Colab GPU smoke test of `LoRA A` (~20 steps) then trial of `LoRA C` (50–100 steps) logging VRAM & step time (`lora_train_log.csv`), plus 3-image inference sanity check (no `NaN`). | M1, M2, M3, M4 |
| **Fri `13/11`** | **INTERNAL DEADLINE (`22:00`): Full Draft of `docs/A5_baseline.tex` Ready** (Sections 1–6 complete with all `VERIFIED` baseline & A4-review numbers; `PENDING` markers resolved). | If `LoRA C` trial is verified, optionally add 1 short paragraph in `A5` Section 5; otherwise keep in separate mentor report note. | **M4** (Lead), All |
| **Sat `14/11`** | Team cross-check of `docs/A5_baseline.tex` against the **Number-Audit Checklist** (verify every number against its source CSV/log). | Inspect `LoRA C` vs `LoRA A` loss curves and 3-image qualitative crops. | All |
| **Sun `15/11`** | Revise figures, captions, and formatting in `docs/A5_baseline.tex`; verify all CLI commands in Section 6 (Reproducibility). | Buffer day for LoRA trial or `Real-CE` follow-up. | M4, M1 |
| **Mon `16/11`** | **INTERNAL SUBMIT (`21:00`):** Final proofread, PDF export, filename check, and submission of Assignment 5. | Prepare mentor meeting summary slide/note. | **M1** & All |
| **Tue `17/11`** | **OFFICIAL DEADLINE (`07:00`): Group Assignment 5 (`Baseline`) Due.** | Present baseline + LoRA trial status to mentor. | All |

---

## 3. Section-by-Section Status of Numbers for `docs/A5_baseline.tex` (`VERIFIED` vs. `PENDING`)

Every number in `docs/A5_baseline.tex` must trace to an exact file or log below. Anything marked `PENDING` must not be guessed or fabricated.

| `A5` Report Section | Metric / Fact | Value | Exact Source File or Log | Status |
| :--- | :--- | :--- | :--- | :---: |
| **1. Method & Pipeline Overview** | Sampler steps, order, domain-shift schedule, scale, color fix, preprocessor | `steps=5`, `order=3`, `trunc_12` ($t_1 = T/2 = 0.5$), `sr_scale=4`, `wavelet`, `RealESRNet` | `configs/baseline_realsr_lr128.yaml`, `src/run_dossr.py:101-110` | `VERIFIED` |
| **1. Method & Pipeline Overview** | Upstream training prompt vs. inference prompt | Training: `""` (`empty`); Inference: `"clean, high-resolution, 8k"` (`default`) | `external/DOSSR/dataset/realesrgan.py:179`, `model/shift_sampler.py:322` | `VERIFIED` |
| **2. Implementation & Equivalence Checks** | Synthetic unit test of `CrossAttention` & `VAE AttnBlock` SDPA patch (`fp16`) | `rtol = atol = 1e-2` passed ($\epsilon_{\max} = 0.000977$ on `2x4x16x8`) | `scripts/check_attention_sdpa.py`, `CELL 06` (`251d4a5`) | `VERIFIED` |
| **2. Implementation & Equivalence Checks** | 1-Image `128x128 -> 512x512` `SDPA ON` vs `SDPA OFF` pixel diff (`0..255`) | `max_abs_diff = 2.0000`, `mean_abs_diff = 0.0334` | `CELL 09` (`docs/after_01_dossr_baseline_discovery.md:139`) | `VERIFIED` |
| **2. Implementation & Equivalence Checks** | Prompt wrapper equivalence (`default` vs `unwrapped`, `Canon_001_LR4`, `seed 42`, `SDPA OFF`, `fp16`) | `max_abs_diff = 0.0000`, `mean_abs_diff = 0.0000` (`2.01 s` vs `2.02 s`) | `CELL 16` (`docs/after_01_dossr_baseline_discovery.md:175-178`) | `VERIFIED` |
| **2. Implementation & Equivalence Checks** | 3-Image same-seed determinism (`seed 42` vs `seed 42-repeat`, `0..255`) | `max_abs_diff = 0.0000`, `mean_abs_diff = 0.0000` (all 3 images) | `CELL 11` (`docs/after_01_dossr_baseline_discovery.md:147-150`) | `VERIFIED` |
| **2. Implementation & Equivalence Checks** | 3-Image pilot cross-seed diff (`seed 42` vs `seed 43`, `0..255`) | `mean_abs_diff = 8.8237`, `max_abs_diff = 255.0` (`Canon_001`: `5.1717`, `Canon_002`: `11.9893`, `Canon_003`: `9.3100`) | `CELL 11` (`docs/after_01_dossr_baseline_discovery.md:147-151`) | `VERIFIED` |
| **3. Experimental Setup** | Dataset split, pairing, and dimensions | `RealSRVal_crop128`: `100` `test_LR` (`128x128`) paired 1-to-1 with `100` `test_HR` (`512x512`) | `CELL 04` (`docs/after_01_dossr_baseline_discovery.md:124-126`) | `VERIFIED` |
| **3. Experimental Setup** | Ground-truth text annotation counts | `25 / 100` text-positive images (`10` Canon + `15` Nikon), `193` `readable=yes` word boxes (`50` Canon + `143` Nikon) | `data/text_boxes_realsr.csv` (`23bc03b`) | `VERIFIED` |
| **3. Experimental Setup** | Software stack, commits, seeds, and `512x512` VRAM / latency | Python `3.13.16`, PyTorch `2.11.0+cu130`, DoSSR `687860c`, runner `131e636`, seeds `42..51` (`configs/seeds_10.yaml`); `SDPA OFF`: `peak_allocated = 7.32 GiB`, `peak_reserved = 8.28 GiB`, `sampler = 1.42 s`, `image = 1.92 s`; `2,000` PNGs verified | `CELL 03`, `CELL 09`, `CELL 17`, `CELL 17b` (`01dc1e2`) | `VERIFIED` |
| **4.1 Whole-Image 10-Seed Metrics** | `Default` prompt (`100` imgs $\times$ `10` seeds) RGB & Y-channel `PSNR`, `SSIM`, `LPIPS` (`mean ± within-image seed SD`, `range`) | `PSNR_RGB = 22.6353 ± 0.5961 dB` (`range = 1.8966`), `SSIM_RGB = 0.6406 ± 0.0182` (`range = 0.0573`), `PSNR_Y = 24.1610 ± 0.6170 dB`, `SSIM_Y = 0.6820 ± 0.0179`, `LPIPS (alex) = 0.3382 ± 0.0191` (`range = 0.0608`) | `results/metrics_10seed_default_summary_by_image.csv` (`ALL_IMAGES_MEAN`, `7564c85`) | `VERIFIED` |
| **4.1 Whole-Image 10-Seed Metrics** | `Empty` prompt (`100` imgs $\times$ `10` seeds) RGB & Y-channel `PSNR`, `SSIM`, `LPIPS` (`mean ± within-image seed SD`, `range`) | `PSNR_RGB = 22.6998 ± 0.6079 dB` (`range = 1.9232`), `SSIM_RGB = 0.6431 ± 0.0184` (`range = 0.0574`), `PSNR_Y = 24.2262 ± 0.6295 dB`, `SSIM_Y = 0.6845 ± 0.0181`, `LPIPS (alex) = 0.3352 ± 0.0191` (`range = 0.0602`) | `results/metrics_10seed_empty_summary_by_image.csv` (`ALL_IMAGES_MEAN`, `7564c85`) | `VERIFIED` |
| **4.1 Whole-Image 10-Seed Metrics** | Whole-image pairwise cross-seed pixel $L_1$ diff (`0..255`) and pairwise LPIPS (`45` seed pairs/image) | `Default`: `pair_px_mean = 10.4508` (`max = 229.67`), `pair_lpips_mean = 0.1727` (`max = 0.2064`)<br>`Empty`: `pair_px_mean = 10.3415` (`max = 229.07`), `pair_lpips_mean = 0.1705` (`max = 0.2039`) | `results/metrics_10seed_{default,empty}_summary_by_image.csv` (`7564c85`) | `VERIFIED` |
| **4.1 Whole-Image 10-Seed Metrics** | Bicubic $\times 4$ (`test_LR` $\rightarrow$ `512x512`) whole-image `PSNR`, `SSIM`, `LPIPS` reference row | Not yet run through `compute_metrics.py` | To be generated by Member 4 | `PENDING` |
| **4.2 Text Evaluation (Current Split-Run)** | `HR ceiling` (`136/193 = 70.5%`, `CER = 0.1648`), `Bicubic floor` (`39/193 = 20.2%`, `CER = 0.7423`), `DoSSR default` (`463/1930 = 24.0%`, `WCR = 76.0%` box / `68.2%` img, `CER = 0.7127` box / `0.6451` img, `disagree = 32.83%` box / `31.57%` img) | Note: Canon (`1–50`) used `--upscale 2` while Nikon (`51–100`) used `--upscale 1` (`RapidOCR 1.2.3`) | `results/text_eval_*.csv` (`23bc03b`) | `VERIFIED` *(Split-Run)* |
| **4.2 Text Evaluation (A4-Review Tasks)** | **Unified single-OCR run** (one exact `RapidOCR` version, `--rec-only --rotate-vertical`, same `--upscale` on all 193 boxes) | All 193 boxes scored in a single invocation | Task `4.A` below (Member 2 / Member 3) | `PENDING` |
| **4.2 Text Evaluation (A4-Review Tasks)** | **Full 10-seed `Empty-Prompt` standalone row** (Exact match, WCR, CER, Seed Disagreement, Distinct Readings, Pixel Spread) | Currently only `delta_wcr` (`-0.16%`) and `delta_cer` (`+0.0001`) are saved in `paired_comparison_by_image.csv` | Task `4.B` below (Member 2) | `PENDING` |
| **4.2 Text Evaluation (A4-Review Tasks)** | **OCR-Solvable Subset Analysis** (restricting to the `136` boxes where `HR_CEILING` gets `wrong_char == 0`) | Removes the `29.5%` HR OCR failure floor so errors reflect DoSSR rather than OCR limitations | Task `4.C` below (Member 2) | `PENDING` |
| **4.2 Text Evaluation (A4-Review Tasks)** | **Image-Level Bootstrap 95% Confidence Intervals** ($N = 25$ images, $10{,}000$ resamples) | 95% CIs for image-mean WCR, CER, seed disagreement, and paired `empty - default` $\Delta$ | Task `4.D` below (Member 2) | `PENDING` |
| **4.2 Text Evaluation (A4-Review Tasks)** | **Capped CER $\min(\text{CER}, 1.0)$ and Median / IQR CER** | Prevents multi-token hallucinations (like `eduroam` `CER = 2.57` when insertions exceed GT length) from skewing the mean | Task `4.E` below (Member 2) | `PENDING` |
| **4.3 Seed Spread Control Boxes** | **Matched Non-Text Control Boxes** for pixel seed spread ($L_1$) | Compare text-box pixel spread (`20.28`) against matched-size background/non-text boxes in the same 25 images (rather than only against whole-image `10.45`, which includes flat sky/walls) | Task `4.F` below (Member 2 & Member 4) | `PENDING` |
| **5. Preliminary LoRA Feasibility (Optional)** | Short Colab trial of `LoRA C` (`50–100` steps) VRAM, step time, trainable params | Currently `ESTIMATE — NOT TESTED` (`~0.8–3.2M` params, `~12–18 GiB` VRAM) | Parallel Track (Member 3 & Member 1) | `PENDING` *(Optional in A5)* |

---

## 4. A4-Review Statistical & Methodological Upgrades (New A5 Tasks with Owners)

To make Assignment 5 (`Baseline`) bulletproof against the methodological limitations noted in `A4`, we add the following 7 concrete tasks (all CPU-only, fast to run on the existing `2,000` PNGs and CSVs):

1. **Task `4.A` — Unified Single-OCR Version & Flags Across All 193 Boxes (Owner: Member 2 `Gia Hân` / Member 3 `Quang Minh`):**
   - **Why:** In `A4`, Canon (`1–50`) was run with `--upscale 2` (`crop_upscale=2` in `results/text_eval_10seed_canon/`) while Nikon (`51–100`) was run with `--upscale 1` (`crop_upscale=1` in `results/text_eval_nikon_summary_by_image.csv`), and Colab (`RapidOCR 1.4.4`) vs local (`RapidOCR 1.2.3`) gave different HR ceilings (`75.5%` vs `70.6%`).
   - **Action:** Run `scripts/text_eval.py` **once** on the merged `data/text_boxes_realsr.csv` (`193` boxes) in a single environment, recording the exact `rapidocr_onnxruntime.__version__` and using identical flags (`--ocr-backend rapidocr --rec-only --rotate-vertical --upscale 1`, plus a second check at `--upscale 2` to pick one uniform setting for all 193 boxes).
2. **Task `4.B` — Full Standalone 10-Seed `Empty-Prompt` Row in the Text Table (Owner: Member 2 `Gia Hân`):**
   - **Why:** `Table 2` in `A4` only showed `DoSSR default, 10 seeds` and quoted `empty - default` deltas in prose, omitting the standalone `Empty-prompt` exact match, WCR, CER, and seed disagreement rate.
   - **Action:** Run `scripts/text_eval.py` with `--study-root <EMPTY_10SEED_ROOT>` as primary (or have `scripts/analyze_text_eval_a5.py` compute the full `Empty-prompt` row directly) so `Table 2` in `A5` has complete side-by-side rows for `HR ceiling`, `Bicubic x4 floor`, `DoSSR default (10 seeds)`, and `DoSSR empty (10 seeds)`.
3. **Task `4.C` — OCR-Solvable Subset Analysis (`HR_CEILING` `wrong_char == 0` Subset, Owner: Member 2 `Gia Hân`):**
   - **Why:** Because RapidOCR only gets `70.5%` (`136/193`) exact match on sharp `test_HR` crops, `29.5%` of all boxes fail even on ground truth. Reporting DoSSR accuracy on the **OCR-solvable subset** (`136` boxes where `HR_CEILING` has `wrong_char == 0` and `CER == 0`) isolates DoSSR restoration failure from OCR failure!
   - *(Preliminary check recomputed from `results/text_eval_*.csv` on 11/11 — to be re-confirmed by the unified run: on the `136` OCR-solvable boxes, `HR_CEILING` is `100.0%` exact (`CER = 0.0000`), `Bicubic x4` is `39/136 = 28.7%` (`CER = 0.6626`), and `DoSSR default` is `458/1360 = 33.7%` (wrong-char rate `66.3%`, `CER = 0.6186`). Seed disagreement on this subset has NOT been computed yet.)*
4. **Task `4.D` — Image-Level Nonparametric Bootstrap 95% Confidence Intervals (Owner: Member 2 `Gia Hân`):**
   - **Why:** The 10 seeds of one image are not independent, so the **image** ($N = 25$ text-positive images for text metrics; $N = 100$ images for whole-image metrics) is the primary sampling unit.
   - **Action:** Compute 10,000-resample percentile bootstrap 95% CIs over images ($N = 25$) for image-mean WCR, CER, capped CER, cross-seed disagreement, and paired `empty - default` differences ($\Delta\text{WCR}, \Delta\text{CER}$).
5. **Task `4.E` — Capped CER $\min(\text{CER}, 1.0)$ and Median / IQR CER (Owner: Member 2 `Gia Hân`):**
   - **Why:** Unbounded Levenshtein CER ($\text{edit\_dist} / \max(1, |\text{GT}|)$) can exceed `1.0` when OCR hallucinates extra characters on a short word (e.g., `Canon_001` `eduroam` has 7 characters, and OCR reading `"j 4 4 4 = w l odareum"` gives `CER = 2.57`, making image-mean `CER = 1.70`).
   - **Action:** Report both **unbounded mean CER**, **capped mean CER $\min(\text{CER}, 1.0) \in [0, 1]$**, and **median CER (with IQR)** across boxes and across images.
6. **Task `4.F` — Matched-Size Non-Text Control Boxes for Pixel Seed Spread (Owner: Member 2 `Gia Hân` & Member 4 `Minh Nhật`):**
   - **Why:** Comparing text-box pixel $L_1$ spread (`20.28`) only against whole-image pixel spread (`10.45`) is confounded because whole images contain smooth sky/wall regions with low texture variance, whereas text boxes are small, high-contrast regions.
   - **Action:** For each of the `193` text boxes `(x, y, w, h)` in the `25` text-positive images, sample a **non-overlapping control box of the exact same `(w, h)`** in the same image (1) at a random non-text location and (2) at a high-gradient non-text edge/texture location, and compare the cross-seed pixel $L_1$ spread (`text box` vs. `matched random box` vs. `matched high-gradient non-text box` vs. `whole image`).
7. **Task `4.G` — Explicit Metric Definitions in `docs/A5_baseline.tex` (Owner: Member 4 `Minh Nhật`):**
   - **Action:** Explicitly state in `Section 3.3` of `docs/A5_baseline.tex` the exact formula and implementation details from `scripts/compute_metrics.py` (`lines 127–189, 322–362`):
     - **PSNR:** Full-reference in `float64` on `[0, 1]` tensors, $\text{PSNR} = 10 \log_{10}(1.0 / (\text{MSE} + 10^{-8}))$, `crop_border = 0`, computed on both **RGB** (3-channel mean MSE) and **Y-channel** (ITU-R BT.601 luma $Y = (65.481 R + 128.553 G + 24.966 B)/255 + 16/255$, matching `external/DOSSR/utils/image/common.py`).
     - **SSIM:** Standard $11 \times 11$ Gaussian window ($\sigma = 1.5$), $C_1 = 0.01^2, C_2 = 0.03^2$, `padding = 0` (valid convolution, Wang et al. 2004 / BasicSR), `crop_border = 0`, in `float64` on both **RGB** (averaged across 3 channels) and **Y-channel**.
     - **LPIPS:** `lpips` package (`v0.1.4+`), **AlexNet** backbone (`net='alex'`), `normalize=True` (inputs in `[0, 1]` rescaled internally to `[-1, 1]`), evaluated in `float32`.
     - **Cross-Seed Pairwise Metrics:** Computed across all $\binom{10}{2} = 45$ unordered seed pairs per image: pixel-space mean absolute difference $\frac{1}{3HW}\|I_{s_a} - I_{s_b}\|_1$ on `uint8` `[0, 255]` arrays, and pairwise $\text{LPIPS}(I_{s_a}, I_{s_b})$.

---

## 5. Parallel Track Step 1: `Real-CE` (`4x`) Feasibility Check First & Data Split Rules

Before writing or running any LoRA training data loader, **we put a `Real-CE` (`4x`) feasibility check first** and **remove hand-labeling of RealSR training images** unless `Real-CE` fails. We do **not** decide the final dataset automatically; we list the options and run the check to give you and the mentor concrete facts.

### 5.1 `Real-CE` (`4x`) Feasibility Checklist (Owner: Member 1 & Member 2, Wed `11/11`)
1. **Download & Inspect `Real-CE` (`4x`):**
   - Locate/download the official `Real-CE` dataset (used by TADiSR, NeurIPS 2025).
   - Record total image counts per official split (`train`, `val`, `test`), file naming, and exact `LR` vs `HR` pixel dimensions (whether pre-cropped to `128x128` / `512x512` or full-size camera frames requiring crop extraction).
2. **Verify `LR`/`HR` Pairing & Annotation Format:**
   - Check if `Real-CE` provides ground-truth text annotations (`JSON` / `TXT` / `CSV` / segmentation masks), whether boxes are **word-level, line-level, or character-level**, and whether ground-truth transcripts are included (and in which script: English/Latin vs. Chinese/bilingual).
3. **License & Redistribution Check:**
   - Check the dataset license and terms of use.
4. **OCR Calibration on `Real-CE` `HR` Crops:**
   - Run `scripts/ocr_check_hr.py` (`--ocr-backend rapidocr --rec-only --rotate-vertical`) on a sample of Latin/English `Real-CE` `HR` crops to verify the HR OCR ceiling.
5. **3-Pair Smoke Test Through DoSSR Inference:**
   - Run 3 `Real-CE` `4x` `LR` crops (`128x128 -> 512x512`, `seed 42`, `FP16`, `SDPA OFF`) through `src/run_dossr.py`, recording peak allocated/reserved VRAM, per-image runtime, and qualitative visual behavior.

### 5.2 Strict Zero-Overlap Split Definition & Near-Duplicate Check
Regardless of which training dataset option is chosen by the user and mentor:
- **Test Split (Strictly Held Out — Never Used in Training or Model/Hyperparameter Selection):**
  - `RealSRVal_crop128` (`100` images, `25` text-positive images, `193` boxes).
  - *(If `Real-CE` is also used for evaluation, its official `Real-CE` test split is also strictly held out.)*
- **Train & Validation Splits (Strictly Disjoint from Test and from Each Other):**
  - Split at the **scene / physical sign / camera-capture level** (never putting two crops from the same physical parent photo into different splits).
- **Near-Duplicate Scene Audit (If RealSR Train Split Is Ever Considered):**
  - Before any RealSR training image could ever be used, run a perceptual hash / CLIP / LPIPS nearest-neighbor check between all RealSR `train` full frames and the `100` `RealSRVal_crop128` `test_HR` crops to verify zero shared physical scenes or buildings.

### 5.3 Candidate Training Data Options (For User & Mentor Decision — Do Not Auto-Decide)
- **Option 1 (Primary Candidate — Check First): `Real-CE` (`4x`) Official Train/Val Splits** (filtered to English/Latin readable text boxes if using English RapidOCR, or using its ground-truth text masks directly since $\mathcal{L}_{\text{text-rec}}$ and $\mathcal{L}_{\text{text-cons}}$ only need the binary box mask $M_{\text{HR}}$ during training!).
  - *Key insight:* During LoRA training, `A`, `B`, and `C` only need `(LR, HR, binary_text_mask M_HR)` — they do **not** run OCR during training! OCR is only needed on the validation/test splits.
- **Option 2 (Backup if `Real-CE` is unavailable or incompatible): Public Scene-Text HR Dataset (`TextOCR` / `ICDAR 2015` / `HierText`) + DoSSR Real-ESRGAN Degradation** (already has human-verified word polygons; zero overlap with `RealSRVal`).
- **Option 3 (Last Resort Only if Options 1 & 2 Fail): Disjoint `RealSR` Train Split** (after passing the scene-level near-duplicate audit against `RealSRVal_crop128`).

---

## 6. Staged LoRA Implementation & Verification Protocol (CPU Tests `(a)`–`(f)` $\rightarrow$ Colab GPU Smoke/Trial)

### 6.1 Which Modules Receive LoRA Adapters & Why `attn1` + `attn2` + Projections Are Included
1. **Why `attn2` (Cross-Attention) Alone Is Not Enough Under an Empty Prompt:**
   - In `ControlLDM`, text conditioning $c_{\text{crossattn}} = \text{FrozenOpenCLIPEmbedder}(\text{""})$ is a **constant tensor** across all images, timesteps, and noises when `prompt_mode="empty"`.
   - In `CrossAttention` (`attn2`), keys $K = W_k c_{\text{crossattn}}$ and values $V = W_v c_{\text{crossattn}}$ depend only on that constant empty-prompt embedding! Only queries $Q = W_q h$ depend on the spatial image features $h$.
   - Meanwhile, spatial stroke geometry and local neighborhood consistency are processed in **Self-Attention (`attn1`: `to_q, to_k, to_v, to_out.0` where $Q, K, V$ all come from spatial features $h$)** and **ControlNet feature injection (`zero_convs` / `input_hint_block`)**.
2. **Default Target Modules for LoRA (`rank r = 4` or `8`, `lora_alpha = r`):**
   - **Where in the network (respecting `ControlledUnetModel.forward` at `external/DOSSR/model/cldm.py:27-35`):**
     - `ControlledUnetModel.forward` wraps `diffusion_model.input_blocks` and `diffusion_model.middle_block` in `with torch.no_grad():`.
     - Therefore, without modifying `external/DOSSR`, gradients flow cleanly through:
       1. **`model.control_model` (ControlNet):** all `SpatialTransformer` blocks (**both self-attention `attn1` and cross-attention `attn2`** `to_q, to_k, to_v, to_out.0`, plus optional `proj_in` / `proj_out`).
       2. **`model.model.diffusion_model.output_blocks` (UNet Decoder):** all `SpatialTransformer` blocks (**both `attn1` and `attn2`** `to_q, to_k, to_v, to_out.0`, plus optional `proj_in` / `proj_out`).
   - All base model weights are frozen (`requires_grad = False`); only `(lora_A, lora_B)` adapter matrices have `requires_grad = True`.

### 6.2 Memory-Safe 2-Noise Execution & Explicit Detached Fallback Option
- **Primary Exact 2-Noise Implementation (Unchanged Loss Mathematics):**
  - For a batch of size $B$, at timestep $t$, we draw $\epsilon_1, \epsilon_2 \sim \mathcal{N}(0, I)$ and compute $\hat{x}_0^{(1)}$ and $\hat{x}_0^{(2)}$.
  - For **LoRA `A` and `B`**, the two terms in $\mathcal{L}_A$ and $\mathcal{L}_B$ do not acoplarse across $\epsilon_1$ and $\epsilon_2$, so we can even accumulate gradients across $\epsilon_1$ and $\epsilon_2$ (or run them in one graph at `batch_size=1` with activation checkpointing `use_checkpoint=True`, which is already enabled in `cldm_v21.yaml:28, 61`).
  - For **LoRA `C`**, $\mathcal{L}_{\text{text-cons}} = \frac{\|M_{\text{lat}} \odot (\hat{x}_0^{(1)} - \hat{x}_0^{(2)})\|_1}{\text{mean}(M_{\text{lat}}) + \varepsilon}$ couples $\hat{x}_0^{(1)}$ and $\hat{x}_0^{(2)}$. At `batch_size=1` (`512x512`, FP16 autocast, `use_checkpoint=True`), keeping both graphs in memory is estimated at `~12–18 GiB` (`ESTIMATE — NOT TESTED`), which fits on an L4 (`22.5 GiB`).
- **Explicit VRAM Fallback (`--detach-second-noise`, Labeled as a Loss Change):**
  - If keeping both computation graphs simultaneously ever exceeds VRAM on a smaller GPU (like a 15 GiB T4), `src/lora_text_loss.py` provides an explicit CLI flag `--detach-second-noise`:
    1. Compute $\hat{x}_0^{(2)}$ inside `with torch.no_grad():` (or detach $\hat{x}_0^{(2)}$ after computing its backward pass for $\frac{1}{2}\mathcal{L}_{\text{orig}}(\epsilon_2) + \frac{1}{2}\lambda_{\text{rec}}\mathcal{L}_{\text{text-rec}}(\epsilon_2)$ so only one forward activation graph is held in memory at a time!).
    2. **Important distinction:**
       - *Trick 1 (Exact gradient decomposition — ZERO change to the mathematical loss $\mathcal{L}_C$!):* By the chain rule, $\nabla_\theta \mathcal{L}_C(\hat{x}_0^{(1)}(\theta), \hat{x}_0^{(2)}(\theta)) = \nabla_\theta \mathcal{L}_C(\hat{x}_0^{(1)}(\theta), \text{sg}[\hat{x}_0^{(2)}]) + \nabla_\theta \mathcal{L}_C(\text{sg}[\hat{x}_0^{(1)}], \hat{x}_0^{(2)}(\theta))$! If you run forward without grad to get values $\bar{x}_0^{(1)}, \bar{x}_0^{(2)}$ (or run pass 1 + backward with detached $\bar{x}_0^{(2)}$, then pass 2 + backward with detached $\bar{x}_0^{(1)}$), you get the **100% exact gradient of the coupled loss** while only holding **one** forward pass in VRAM!
       - *Trick 2 (`--detach-second-noise` one-sided stop-grad fallback):* Computes $\hat{x}_0^{(2)}$ under `torch.no_grad()` as a fixed target for $\hat{x}_0^{(1)}$. This is explicitly logged in `lora_run_manifest.json` as `"loss_mode": "detached_second_noise_target"` because it changes the loss to a one-sided target consistency loss.

### 6.3 Six Mandatory CPU Unit Tests (`a`–`f`) Before Any Colab Run
Built directly into `src/lora_text_loss.py --run-cpu-tests` (`NOT TESTED` until executed):
- **Test `(a)` — Mask Downsampling & Mask/Image Alignment After Crop and Horizontal Flip:**
  - Verifies that converting boxes `(x, y, w, h)` in `512x512` HR space to $M_{\text{HR}} \in \{0, 1\}^{B \times 1 \times 512 \times 512}$ and applying `F.max_pool2d(M_HR, kernel_size=8, stride=8)` produces $M_{\text{lat}} \in \{0, 1\}^{B \times 1 \times 64 \times 64}$ where every covered $8 \times 8$ block is `1`.
  - Verifies that applying any spatial crop `(top, left, h, w)` or horizontal flip simultaneously to `(HR, LR, M_HR)` matches recomputing $M_{\text{HR}}$ from the transformed box coordinates with `0` pixel discrepancy.
- **Test `(b)` — Exact Identity Initialization (`LoRA Output == Base Output at Init`):**
  - Verifies that with `lora_A` initialized via Kaiming uniform and `lora_B` initialized to **zeros**, the output of the LoRA-wrapped module equals the unmodified base module output to `max_abs_diff == 0.0` (bit-exact at step 0).
- **Test `(c)` — Gradient Flow Audit (`Trainable Params Get Gradients, Frozen Params Do Not`):**
  - Runs a forward + backward pass on a dummy `ControlledUnetModel` + `ControlNet` hierarchy (including the `with torch.no_grad():` block on `input_blocks` / `middle_block`) and asserts:
    1. Every `lora_A` (and `lora_B` after 1 step) parameter in `control_model` and `output_blocks` has `param.grad is not None` and `grad.abs().sum() > 0`.
    2. Every base parameter has `requires_grad == False` and `param.grad is None`.
- **Test `(d)` — Shared Batch, Degraded LR (`preprocess_x`), and Timestep $t$ Across $(\epsilon_1, \epsilon_2)$:**
  - Asserts that during a single step, `predict_x0_out_mix` is called twice with the **exact same** `x` ($x_0$), `preprocess_x` ($\hat{x}_{\text{LR}}$), conditioning `c`, and timestep tensor `t`, while `noise=eps_1` and `noise=eps_2` are independent ($\|\epsilon_1 - \epsilon_2\|_1 > 0$).
- **Test `(e)` — Two-Noise Fairness Across Variants `A`, `B`, and `C`:**
  - Asserts that Variants `A`, `B`, and `C` all draw two noise samples $(\epsilon_1, \epsilon_2)$, all call `predict_x0_out_mix` twice per step, and that setting $\lambda_{\text{rec}} = 0, \lambda_{\text{cons}} = 0$ makes $\mathcal{L}_C == \mathcal{L}_B == \mathcal{L}_A$ to machine precision, while setting $\lambda_{\text{cons}} = 0$ makes $\mathcal{L}_C == \mathcal{L}_B$.
- **Test `(f)` — End-to-End CPU Dry Run (2 Optimizer Steps with Dummy Module):**
  - Runs 2 training steps (`step 1` and `step 2`) with `AdamW` on CPU, saves a tiny LoRA state dict (`.pt`), reloads it into a fresh model, and verifies that the reloaded model matches the post-step-2 model output (`max_abs_diff == 0.0`) and differs from step 0 (`max_abs_diff > 0.0`).

### 6.4 Staged Colab GPU Execution (After CPU Tests Pass)
1. **Stage 1 — Colab Smoke Run of `LoRA A` (~20 steps):**
   - Verifies checkpoint loading (`dossr_default.ckpt`), LoRA injection, 2-noise forward/backward, and checkpoint saving to Google Drive without OOM.
2. **Stage 2 — Colab Trial Run of `LoRA C` (50–100 steps):**
   - Logs in `lora_run_manifest.json` and `lora_train_log.csv`:
     - GPU model (`torch.cuda.get_device_name(0)`), total VRAM, `trainable_param_count`, `total_param_count`.
     - Per-step: `step`, `t_sampled`, `loss_total`, `loss_orig_2noise`, `loss_text_rec`, `loss_text_cons`, `mask_active_ratio`, `step_time_s`, `peak_allocated_gib`, `peak_reserved_gib`.
   - Saves resumable LoRA checkpoints (`lora_weights_step_<N>.pt` + optimizer state) to Drive.
3. **Stage 3 — Inference Sanity Check on 3 Images (`No NaN`):**
   - Loads the saved `LoRA C` weights on top of `dossr_default.ckpt`, runs 5-step inference on 3 validation images (`seed 42`), verifies finite pixel values (`no NaN/Inf`, not all-black/all-white), and logs the pixel difference against base DoSSR (`> 0.0`, confirming the adapter is active at inference).

---

## 7. Pre-Registered Validation & Training Protocol (Fixed Before Training)

To prevent overfitting or cherry-picking, the following protocol is fixed before running `LoRA A, B, C`:
1. **Validation Split:** `15–20` text-positive LR-HR pairs from the chosen dataset (`Real-CE` val split or disjoint scene-text val split), strictly disjoint from both the training split and the 100 `RealSRVal` test images.
2. **Validation Inference Seeds:** Fixed **5 seeds** (`42, 43, 44, 45, 46` — the first 5 seeds of `configs/seeds_10.yaml`) for fast validation checkpoint selection; full **10 seeds (`42..51`)** used once at the end on the held-out test set.
3. **Primary Validation Metric:**
   - Primary: **Image-mean wrong-character rate** and **cross-seed OCR disagreement rate** on the **OCR-solvable validation boxes** (with whole-image `LPIPS` and `PSNR_RGB` monitored as a guardrail to ensure whole-image quality does not degrade by $> 0.3\text{ dB}$ PSNR or $> 0.02$ LPIPS).
4. **Fixed Hyperparameters & Training Determinism Across `A`, `B`, `C`:**
   - **Training seed:** Fixed `seed = 231` (matching upstream `train_cldm.yaml:2`).
   - **Identical data order & noise stream:** `A`, `B`, and `C` iterate through the exact same shuffled batch sequence (`torch.Generator().manual_seed(231)`) and draw the exact same $(\epsilon_1, \epsilon_2, t)$ sequence at step $k$, so any difference between `A`, `B`, and `C` is caused strictly by the loss function!
   - **Default LoRA settings:** `rank r = 4`, `lora_alpha = 4`, `lr = 1e-4` (`AdamW`, `weight_decay = 1e-2`), `batch_size = 1`, `accumulate_grad_batches = 4` (effective batch size `4`), `prompt_mode = "empty"` (`""`), `lambda_rec = 1.0`, `lambda_cons = 0.5`.
5. **How Many Training Seeds We Can Afford:**
   - Under our Colab GPU budget (~2–4 minutes per 100-step trial, or ~15–25 minutes for a 500–1,000 step run), we can afford **1 fixed training seed (`seed = 231`)** for the full `A vs B vs C` comparison, plus **1–2 additional training seeds (`232, 233`)** on the winning `B vs C` pair (or short 200-step runs) to verify training-seed stability. Evaluation always uses **10 inference seeds (`42..51`)** per image.

---

## 8. Ordered Implementation Task Groups (Report After Each Group)

- **Group 1 — A4-Review Statistical Analysis Script & Unified Text Eval (`Track 1: A5 Baseline`):**
  - Create `scripts/analyze_text_eval_a5.py` (`NOT TESTED` until run on CSVs) that reads the `text_eval` CSVs and `test_HR`/`seed_*` crops to compute:
    1. The **OCR-solvable subset** (`HR_CEILING` `wrong_char == 0`) table.
    2. **Image-level bootstrap 95% CIs** ($10{,}000$ resamples over the $N=25$ images).
    3. **Capped CER $\min(\text{CER}, 1.0)$** and **median (IQR) CER**.
    4. **Standalone `Empty-prompt` 10-seed row** (by scoring or aggregating `<EMPTY_10SEED_ROOT>`).
    5. **Matched-size non-text control boxes** (random non-text & high-gradient non-text boxes in the same 25 images) for pixel seed spread ($L_1$).
- **Group 2 — `docs/A5_baseline.tex` Template & Number-Audit Checklist (`Track 1: A5 Baseline`):**
  - Create `docs/A5_baseline.tex` populated **only** with `VERIFIED` numbers from `docs/fact_sheet_for_report.md`, explicit metric definitions (`PSNR/SSIM/LPIPS` library, channels, border, network), and explicit `PENDING` placeholders for the Group 1 analyses.
  - Add a **Number-Audit Checklist** section in `docs/fact_sheet_for_report.md` mapping every table cell in `A5` to its exact CSV row/column.
- **Group 3 — LoRA & Loss Scaffold + CPU Tests `(a)`–`(f)` + Code Walkthrough (`Track 2: LoRA Trial`):**
  - Create `src/lora_text_loss.py` and `scripts/run_lora_trial.py` (`NOT TESTED` on Colab; run CPU unit tests `(a)`–`(f)` locally).
  - Create `docs/lora_code_walkthrough.md` explaining every function in simple words for Member 3 (`Quang Minh`) and the team.
- **Group 4 — `Real-CE` (`4x`) Inspection Helper & Updated Handoffs + `context.md` Sync:**
  - Create `scripts/check_realce_feasibility.py` (`NOT TESTED`) to automate the `Real-CE` (`4x`) audit (counts, pairing, sizes, annotation format, HR OCR check).
  - Update `docs/handoff_member2.md`, `docs/handoff_member3.md`, `docs/handoff_member4.md`, and `context.md`.
