# Verified Fact Sheet for Group Assignments A4 & A5

**Project:** SC203 Scientific Methods (Class 25A02) — Team 1 (`LEMMEPASS`)  
**Repository & Branch:** `https://github.com/lemon4life/SC203` (`feat/baseline-runner`)  
**Rule for Report Writers (Member 4 — Minh Nhật):**  
- Every number cited in `A4` (*SOTA's Limitations*, due **Tue 10/11/2026 07:00**) and `A5` (*Baseline*, due **Tue 17/11/2026 07:00**) **must** come from a `VERIFIED` row below (or be explicitly marked `PENDING` / `ESTIMATE` until the log/CSV arrives).  
- Do **not** invent numbers, and do **not** mix `2048x2048` accidental `test_HR` stress-test numbers with true `128x128 -> 512x512` `test_LR` baseline numbers.

---

## 1. Team Roster & Roles (`VERIFIED`)

| Member | Full Name | Student ID | Role & Verified Deliverables | Status |
| :--- | :--- | :---: | :--- | :---: |
| **Member 1** | Hoàng Nguyên Anh | `25125002` | GPU & Pipeline Lead (`notebooks/01_dossr_baseline.ipynb` `CELL 01`–`18`, `2,000` 10-seed `512x512` PNGs on Drive), CSV merge, final PDF assembly | `VERIFIED` |
| **Member 2** | Huỳnh Trần Gia Hân | `25125042` | Canon text-box labeling (`images 1–50`, `data/text_boxes_realsr_canon.csv` / `_m2.csv`: `10` readable images, `50` boxes) & 10-seed Canon `text_eval.py` (`results/text_eval_10seed_canon/`) | `VERIFIED` |
| **Member 3** | Hồ Quang Minh | `25125060` | Nikon text-box labeling (`images 51–100`, `data/text_boxes_realsr_nikon.csv` / `_m3.csv`: `15` readable images, `143` boxes), `--rec-only --rotate-vertical` OCR fix (`ca127fe`), HR OCR ceiling check, & 10-seed Nikon `text_eval.py` (`results/text_eval_nikon_*.csv`) | `VERIFIED` |
| **Member 4** | Nguyễn Phan Minh Nhật | `25125033` | LaTeX report (`A4` & `A5`), whole-image metrics (`scripts/compute_metrics.py`), 1-page related-methods comparison (`CCSR`, `PNFS`, `TADiSR`) | `IN PROGRESS` |

---

## 2. Dataset & Ground-Truth Text-Box Inventory (`VERIFIED`)

| Fact / Metric | Value | Source (`CSV` / `CELL` / `Doc`) | Status |
| :--- | :--- | :--- | :---: |
| **Evaluation Dataset** | `StableSR_testsets/RealSRVal_crop128` | `CELL 04` (`docs/after_01_dossr_baseline_discovery.md`) | `VERIFIED` |
| **Low-Resolution Input (`test_LR`)** | `100` PNGs (`Canon_001`–`050`, `Nikon_001`–`050`), all `128x128` RGB | `CELL 04` (`251d4a5`) | `VERIFIED` |
| **High-Resolution Ground Truth (`test_HR`)** | `100` PNGs (`Canon_001`–`050`, `Nikon_001`–`050`), all `512x512` RGB ($\times 4$ scale) | `CELL 04` (`251d4a5`) | `VERIFIED` |
| **Total Images with `readable=yes` Text** | **`25 / 100` images** (`10 / 50` Canon + `15 / 50` Nikon) | `data/text_boxes_realsr.csv` (`scripts/merge_text_boxes.py`) | `VERIFIED` |
| **Total `readable=yes` Ground-Truth Word Boxes** | **`193` boxes** (`50` Canon boxes + `143` Nikon boxes; `276` total CSV rows including `83` `readable=no` rows) | `data/text_boxes_realsr.csv` (`23bc03b`) | `VERIFIED` |
| **`Canon_001_LR4.png` (`eduroam`) Word Box** | `x=320, y=38, w=192, h=46`, `transcript=eduroam`, `readable=yes` | `data/text_boxes_realsr_canon.csv` (`daddf0e`) | `VERIFIED` |
| **`Canon_001_LR4.png` (`eduroam`) Full Sign Crop (Figure)** | `x=310, y=5, w=202, h=85` (`EDUROAM_CROP = '310,5,202,85'`) | `CELL 18` (`01dc1e2`), `scripts/plot_eduroam_figure.py` | `VERIFIED` |

---

## 3. DoSSR Inference Configuration & Hardware Profiling (`VERIFIED`)

| Fact / Metric | Value | Source (`Log` / `CELL` / `Config`) | Status |
| :--- | :--- | :--- | :---: |
| **Paper-Default Sampler Settings** | `steps: 5`, `solver_order: 3`, `t1_fraction: 0.5` ($t_1 = T/2$), `sr_scale: 4`, `color_fix_type: wavelet`, `preprocessor: RealESRNet`, `precision: fp16` | `configs/baseline_realsr_lr128.yaml`, `src/run_dossr.py` | `VERIFIED` |
| **Colab Hardware & Software Stack** | `NVIDIA A100-SXM4-80GB` (`79.25 GiB` VRAM), Python `3.13.16`, PyTorch `2.11.0+cu130` (`xformers` absent) | `CELL 03` (`docs/after_01_dossr_baseline_discovery.md`) | `VERIFIED` |
| **True `128x128 -> 512x512` (`test_LR`) VRAM & Latency (`SDPA ON`)** | `peak_allocated = 7.32 GiB`, `peak_reserved = 8.29 GiB`, `sampler = 1.24 s`, `image = 1.75 s` | `CELL 08` & `CELL 09` (`251d4a5`) | `VERIFIED` |
| **True `128x128 -> 512x512` (`test_LR`) VRAM & Latency (`SDPA OFF`, vanilla upstream math)** | `peak_allocated = 7.32 GiB`, `peak_reserved = 8.28 GiB`, `sampler = 1.42 s`, `image = 1.92 s` | `CELL 09` (`251d4a5`) | `VERIFIED` |
| **Pixel Difference (`SDPA ON` vs `SDPA OFF` at `512x512`, `0..255`)** | `max_abs_diff = 2.0000`, `mean_abs_diff = 0.0334` | `CELL 09` (`251d4a5`) | `VERIFIED` |
| **Why `10`-Seed Studies Use `--disable-sdpa` (`SDPA OFF`)** | At `512x512` output, sequence length is $N = 64 \times 64 = 4{,}096$, so vanilla upstream `CrossAttention` + `VAE AttnBlock` runs in `7.32 GiB` (`8.28 GiB` reserved) without any attention patch. All 10-seed studies use `disable_sdpa: true` (`sdpa_enabled: false`) so results reflect 100% unmodified upstream DoSSR math. | `configs/seed_study_realsr_100img_10seed_*_sdpa_off.yaml`, `CELL 09`, `CELL 16–17b` | `VERIFIED` |
| **Accidental `512x512 -> 2048x2048` (`test_HR`) Stress Test (Historical Diagnostic Only)** | Unpatched `fp16`: `80.00 GiB` OOM in `CrossAttention` (`ldm/modules/attention.py:178`). With `CrossAttention` SDPA only: `36.04 GiB` (`fp16`, `7.68 s/img`) and `49.29 GiB` (`fp32`, `21.05 s/img`). With `CrossAttention` + `VAE AttnBlock` SDPA (`5e6a777`): `peak_allocated = 20.51 GiB` (`-15.53 GiB` / `-43%`), `peak_reserved = 34.71 GiB`, `8.33 s/img`. *(Quarantined via `README_DO_NOT_USE_AS_BASELINE.txt`; never mix with `512x512` baseline metrics.)* | `CELL 04` & `CELL 09` (`docs/after_01_dossr_baseline_discovery.md`) | `VERIFIED` |

---

## 4. Determinism, Prompt Wrapper & 10-Seed Study Verification (`VERIFIED`)

| Experiment / Check | Verified Numbers | Source (`CELL` / `Log`) | Status |
| :--- | :--- | :--- | :---: |
| **Same-Seed Determinism (`seed 42` vs `seed 42-repeat`, 3 `test_LR` images)** | `max_abs_diff = 0.0000`, `mean_abs_diff = 0.0000` across `Canon_001_LR4`, `Canon_002_LR4`, `Canon_003_LR4` (bit-exact reproducible when seed is fixed) | `CELL 11` (`251d4a5`) | `VERIFIED` |
| **3-Image Cross-Seed Pixel Spread (`seed 42` vs `seed 43`, `0..255`)** | `Canon_001`: `mean = 5.1717, max = 185.0`; `Canon_002`: `mean = 11.9893, max = 223.0`; `Canon_003`: `mean = 9.3100, max = 255.0`; **3-image aggregate:** `mean = 8.8237, max = 255.0` | `CELL 11` (`251d4a5`) | `VERIFIED` |
| **Prompt Wrapper Equivalence (`default` vs `unwrapped`, `Canon_001_LR4`, `seed 42`, `SDPA OFF`, `fp16`)** | `max_abs_diff = 0.0000`, `mean_abs_diff = 0.0000` (proves `src/dossr_inference_compat.py` prompt wrapper is bit-exact with unmodified upstream DoSSR when `prompt_mode="default"` passes `"clean, high-resolution, 8k"`) | `CELL 16` (`Nov 7, 2026`) | `VERIFIED` |
| **1-Image Prompt Comparison (`empty` `""` vs `default` `"clean, high-resolution, 8k"`, `Canon_001_LR4`, `seed 42`)** | Whole-image pixel diff (`0..255`): `max_abs_diff = 50.0000`, `mean_abs_diff = 0.3704` | `CELL 16` (`Nov 7, 2026`) | `VERIFIED` |
| **Full 10-Seed Study Output Verification (`CELL 17` & `CELL 17b`)** | `10` seeds (`42, 43, 44, 45, 46, 47, 48, 49, 50, 51`) $\times$ `100` `test_LR` images = **`1,000` `512x512` PNGs** for `default` prompt + **`1,000` `512x512` PNGs** for `empty` prompt = **`2,000` `512x512` PNGs total**, all verified with `precision="fp16"`, `sdpa_enabled=False`, `disable_sdpa=True` | `CELL 17` & `CELL 17b` (`01dc1e2`, `git_commit=131e6368`) | `VERIFIED` |

---

## 5. Primary A4 Evidence: 10-Seed Text Restoration & Seed-Instability Evaluation (`VERIFIED`)

### A. OCR Calibration on Sharp Ground-Truth `test_HR` Crops (Member 3 — Commit `ca127fe`)
- **Finding:** Default `rapidocr` runs a 2-stage pipeline (`text_det` $\rightarrow$ `text_rec`). On tightly cropped single-word boxes, `text_det` frequently fails to detect a text polygon and returns `""` (`18 / 143 = 12.6%` exact on Nikon in Colab; `8 / 50 = 16.0%` on Canon).
- **Fix (`--rec-only --rotate-vertical` in `scripts/ocr_check_hr.py` & `scripts/text_eval.py`):** Bypassing `text_det` (`--rec-only`, feeding the ground-truth box directly to `text_rec`) and trying $\pm 90^\circ$ rotations on vertical boxes (`h > 1.5 * w`, `--rotate-vertical`) raised `test_HR` exact-match accuracy on Nikon (`143` boxes) from `12.6%` to **`70.6%` (`101/143`, `rapidocr 1.2.3`)** and **`75.5%` (`108/143`, `rapidocr 1.4.4` on Colab)**, outperforming `easyocr` (`76/143 = 53.1%` on Nikon; `35/50 = 70.0%` on Canon at `2x` upscale).

### B. 10-Seed Quantitative Text Evaluation Table (`25` Readable Images, `193` Boxes, `1,930` Box-Seed Crops, `VERIFIED`)
*Source CSVs committed in `results/`:*
- Combined 100-image tables: `results/text_eval_summary_by_image.csv`, `results/text_eval_paired_comparison_by_image.csv`, `results/text_eval_hr_ceiling_by_image.csv`, `results/text_eval_hr_ceiling_per_box.csv`, `results/text_eval_bicubic_floor_by_image.csv`, `results/text_eval_bicubic_floor_per_box.csv`, `results/text_eval_per_box_seed.csv`
- Canon subset (`1–50`, `10` images, `50` boxes): `results/text_eval_10seed_canon/*.csv` (and `results/text_eval_canon_*.csv`)
- Nikon subset (`51–100`, `15` images, `143` boxes): `results/text_eval_nikon_*.csv`

| Subset & Evaluation Target | Readable Images / Boxes | Box-Seeds Evaluated | Exact Box Match Rate (`wrong_char == 0`) | Wrong-Char Rate (Box-Wt / Img-Mean) | Mean CER (Box-Wt / Img-Mean) | Cross-Seed OCR Disagreement Rate (Box-Wt / Img-Mean) | Mean Distinct OCR Readings / 10 Seeds (Box-Wt / Img-Mean) | Crop Pixel Seed Spread $L_1$ (`0..255`) (Box-Wt / Img-Mean) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Canon (`1–50`)** — `HR_CEILING` (`test_HR`) | `10` / `50` | `50` | **`35 / 50` (`70.0%`)** | `30.0%` / `45.8%` | `0.0938` / `0.1967` | — | — | — |
| **Canon (`1–50`)** — `BICUBIC_X4_FLOOR` (`test_LR`) | `10` / `50` | `50` | `10 / 50` (`20.0%`) | `80.0%` / `72.8%` | `0.6704` / `0.5964` | — | — | — |
| **Canon (`1–50`)** — **DoSSR 10-Seed (`default`)** | `10` / `50` | `500` | **`170 / 500` (`34.0%`)** | **`66.0%`** / `61.5%` | **`0.6263`** / `0.6528` | **`46.84%`** / `40.94%` | **`4.44`** / `4.11` | **`15.70`** / `17.50` |
| **Nikon (`51–100`)** — `HR_CEILING` (`test_HR`) | `15` / `143` | `143` | **`101 / 143` (`70.6%`)** *(Colab: `75.5%`)* | `29.4%` / `40.2%` | `0.1896` / `0.2608` | — | — | — |
| **Nikon (`51–100`)** — `BICUBIC_X4_FLOOR` (`test_LR`) | `15` / `143` | `143` | `29 / 143` (`20.3%`) | `79.7%` / `70.6%` | `0.7674` / `0.6407` | — | — | — |
| **Nikon (`51–100`)** — **DoSSR 10-Seed (`default`)** | `15` / `143` | `1,430` | **`293 / 1430` (`20.5%`)** | **`79.5%`** / `72.6%` | **`0.7428`** / `0.6400` | **`27.93%`** / `25.33%` | **`2.70`** / `2.67` | **`21.88`** / `18.40` |
| **FULL 100 (`Canon + Nikon`)** — **`HR_CEILING`** | **`25` / `193`** | **`193`** | **`136 / 193` (`70.5%`)** | **`29.5%`** / `42.5%` | **`0.1648`** / `0.2352` | — | — | — |
| **FULL 100 (`Canon + Nikon`)** — **`BICUBIC_X4_FLOOR`** | **`25` / `193`** | **`193`** | **`39 / 193` (`20.2%`)** | **`79.8%`** / `71.5%` | **`0.7423`** / `0.6230` | — | — | — |
| **FULL 100 (`Canon + Nikon`)** — **DoSSR 10-Seed (`default`)** | **`25` / `193`** | **`1,930`** | **`463 / 1930` (`24.0%`)** | **`76.0%`** / **`68.2%`** | **`0.7127`** / **`0.6451`** | **`32.83%`** / **`31.57%`** | **`3.15`** / **`3.25`** | **`20.28`** / **`18.04`** |
| **FULL 100 Paired (`empty - default` prompt)** | **`25` / `193`** | **`1,930` pairs** | $\Delta = +0.16\%$ | **`-0.16%`** / `-0.32%` | **`+0.0001`** / `+0.0068` | — | — | — |

### C. Concrete Image-Level Examples of Seed Instability (`VERIFIED` in `results/text_eval_per_box_seed.csv`)
1. **`Canon_001_LR4.png` (`eduroam`, 1 box, 10 seeds):**
   - `wrong_char_rate = 1.00` (`100%`), `cer_mean = 1.70`, `ocr_seed_disagreement_rate = 1.00` (**`100%` of seed pairs disagree**), `num_distinct_ocr_mean = 10.0` (**10 different readings across 10 seeds**), `crop_pixel_seed_spread_l1 = 19.17`.
   - The 10 normalized readings across `seed_42`..`seed_51`:
     - `seed_42`: `n l n w d l odureani`
     - `seed_43`: `eaneed d ] aduronm`
     - `seed_44`: `j 4 4 4 = w l odareum`
     - `seed_45`: `4eae oduroant`
     - `seed_46`: `ys u g1 _ 0 edureain`
     - `seed_47`: `tey * ret c u cduroniv`
     - `seed_48`: `neeo adaroam`
     - `seed_49`: `l u / uuue odoroam`
     - `seed_50`: `3t l ddaroain`
     - `seed_51`: `rne/e aduronm`
2. **`Canon_002_LR4.png` (`SIEMENS` / `Ying Feng` / `Building Diagnostic Consultants`, 15 boxes, 10 seeds):**
   - `wrong_char_rate = 1.00` (`100%`, vs `HR_CEILING = 13.3%` where 13/15 boxes are read perfectly on `test_HR`!), `cer_mean = 0.9675`, `ocr_seed_disagreement_rate = 86.96%`, `num_distinct_ocr_mean = 7.80 / 10`, `crop_pixel_seed_spread_l1 = 16.75`.
   - For `box_id=1` (`SIEMENS`), DoSSR hallucinates 10 completely different strings across `seed_42`..`seed_51`: `pamenl`, `slmle`, `ealmmta`, `fumo`, `dunla`, `e:naa`, `eimill`, `bj}t`, `dimdho`, `51433`.
3. **`Nikon_002_LR4.png` (`Rhinoceros` / `Hornbill`, 2 boxes, 10 seeds):**
   - `wrong_char_rate = 1.00` (`100%`, vs `HR_CEILING = 0.0%` where both words are read with `0` errors on `test_HR`!), `cer_mean = 0.7875`, `ocr_seed_disagreement_rate = 100.0%`, `num_distinct_ocr_mean = 10.0 / 10`, `crop_pixel_seed_spread_l1 = 25.12`.
4. **`Nikon_024_LR4.png` (27 boxes, 10 seeds):**
   - `wrong_char_rate = 94.07%` (vs `HR_CEILING = 18.52%`), `cer_mean = 0.8436` (vs `HR_CEILING = 0.1101`), `ocr_seed_disagreement_rate = 68.64%`, `num_distinct_ocr_mean = 5.44 / 10`, `crop_pixel_seed_spread_l1 = 28.75`.

---

## 6. Pending Items (`PENDING` — Do Not Invent Numbers)

| Item | Owner | Current Status | Where Results Will Go |
| :--- | :--- | :---: | :--- |
| **Whole-Image `PSNR`, `SSIM`, `LPIPS`, and Pairwise Whole-Image Seed Spread on the 100-Image 10-Seed Studies (`default` and `empty` prompts)** | Member 4 (Minh Nhật) / Member 1 (Colab GPU) | `PENDING` *(Run `scripts/compute_metrics.py --device cuda` on Colab GPU: ~2 min per 10-seed folder)* | `results/metrics_10seed_default_summary_by_image.csv` & `results/metrics_10seed_empty_summary_by_image.csv` |
| **5-Image / 5-Seed `compute_metrics.py` CSV Values (`CELL 14` bundle)** | Member 1 / Member 4 | `PENDING` *(Zip bundle `sc203_baseline_results_bundle.zip` generated on Drive in `810e844`; extract exact numbers before quoting)* | `results/` |
| **LoRA Variants `(A, B, C)` Short Trial Run & VRAM Profile** | Member 3 (Quang Minh) & Member 1 | `PAUSED until Tue 11/11/2026` *(Design & feasibility verified in `docs/lora_feasibility_and_loss_design.md`; runtime numbers are `ESTIMATE — NOT TESTED`)* | For Assignment 5 (`due 17/11/2026`) |
