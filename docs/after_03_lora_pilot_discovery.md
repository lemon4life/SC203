# Discovery After `03_lora_pilot_colab.ipynb`: LoRA 2 vs. LoRA 3 Pilot on Real-CE $\times 4$ (`VERIFIED`, Nov 15, 2026)

**Status:** `VERIFIED` — Trained (`400` steps each) and evaluated across `10` seeds (`42..51`, `5` DoSSR steps, empty prompt `""`) on `30` scene-disjoint Real-CE $\times 4$ validation patches (`300` images per variant) via `notebooks/03_lora_pilot_colab.ipynb` (`CELL 05`–`CELL 07`).  
**Structured JSON:** [`results/lora_pilot_summary.json`](../results/lora_pilot_summary.json)

---

## 1. The Discovery in Simple Words (For the Whole Team)

Think of DoSSR as a skilled artist redrawing a blurry zoomed-in sign. Whenever you run DoSSR with a different random seed (`seed 42` vs `seed 43`), it is like rolling dice before drawing: **LoRA 2** fixes some spelling, *but still changes its mind wildly whenever you change the seed*.

We tested whether adding **cross-noise consistency (`LoRA 3`)** forces the model to stop rolling the dice inside text regions while keeping the exact same backbone, dataset split, random seed (`231`), step budget (`400` steps), and GPU memory limit.

### What Happened? (The 3 Big Wins + 2 Subtle Findings)

1. **Seed randomness collapsed by ~60% (Big Win):**  
   Across 10 random seeds on held-out scenes never seen during training, **LoRA 3** cut pixel-to-pixel jumping across seeds by **`-57.5%`** (`9.27 -> 3.94` gray levels out of 255), cut perceptual texture jumping (`pair_lpips`) by **`-64.1%`** (`0.2020 -> 0.0726`), cut text-crop pixel jumping by **`-55.1%`** (`15.88 -> 7.13`), and cut how often OCR changed its mind across seeds by **`-13.8%`** (`51.97% -> 44.79%` under corrected `--rec-only --upscale 2` OCR; `-28.5%` in the initial `det+rec` run).
2. **Closer to ground-truth pixels (Win):**  
   LoRA 3 improved whole-patch **PSNR_Y by `+0.5848 dB`** (`17.6980 -> 18.2828`) and **SSIM_Y by `+0.0300`** (`0.6442 -> 0.6742`).
3. **Zero extra GPU memory (`0.00 GiB` overhead):**  
   Because our two-pass chain-rule backward (`src/lora_text_loss.py`) frees pass 1's compute graph before running pass 2, **LoRA 3 used the exact same peak GPU memory as LoRA 2 (`8.04 GiB` allocated / `8.52 GiB` reserved)**, costing only `+15.2%` extra step time (`4.67s -> 5.38s/step`).
4. **Text readability improved (`CER -0.0264`, `-6.19%` error on HR-solvable boxes), while sharpness (`LPIPS +0.0480`) softened:**  
   With corrected `--rec-only --rotate-vertical --upscale 2` OCR, **LoRA 3 reduced text error rate (`-2.62%` overall, `-6.19%` on HR-solvable boxes) and reduced CER (`-0.0264` overall, `-0.0340` on HR-solvable boxes)** compared to LoRA 2. At the same time, forcing two different noise draws to agree too hard (`lambda_cons = 0.5`) across all uniform noisy timesteps $t \in [0, 999)$ caused thin strokes to soften/fade (e.g., `2024/06/17`, explaining whole-patch `LPIPS +0.0480` from `0.4231 -> 0.4711`). Notably, DoSSR's 5-step inference (`t1_fraction = 0.5`) only operates at $t \le 500$, whereas training penalized consistency up to $t = 999$ (`scripts/run_lora_trial.py`).
5. **Initial `CELL 07` `HR Ceiling` (`CER = 0.7334`) was a `DBNet` detector artifact — already resolved (`VERIFIED` Nov 16):**  
   In `CELL 06`, `scripts/text_eval.py` initially ran in default two-stage mode (`det+rec`, `--upscale 1`, without `--rec-only --rotate-vertical`), causing `DBNet` to miss tight word crops (`HR Ceiling` CER `0.7334`) and temporarily making LoRA 3 appear worse on CER (`+0.0147`). On Nov 16, Member 3 (`Quang Minh`) re-scored the saved Drive PNGs with `--rec-only --rotate-vertical --upscale 2` (`results/pilot_lora2_vs_lora3_reconly/`), dropping `HR Ceiling` CER to `0.2722` and confirming LoRA 3 beats LoRA 2 across every text metric.

---

## 2. Controlled Pilot Configuration (`VERIFIED`)

| Setting | LoRA 2 (`configs/lora2_pilot.yaml`) | LoRA 3 (`configs/lora3_pilot.yaml`) |
| :--- | :---: | :---: |
| **Train / Val Split** | Real-CE $\times 4$ ($13\text{mm} \to 52\text{mm}$), scene-disjoint (`0` overlap) | Same |
| **Validation Evaluated** | `30` patches $\times$ `10` seeds (`42..51`), `5` DoSSR steps (`t1_fraction=0.5`), prompt `""` | Same |
| **Trainable LoRA Params** | `1,659,904` (`rank=8`, `alpha=16.0`, `control_model.` + `output_blocks.`) | Same |
| **Optimizer & Budget** | `AdamW` (`lr=1e-4`), `400` steps, `batch_size=1`, `grad_accum=4`, `K=2` noise draws, `seed=231` | Same |
| **Timestep Sampling** | Uniform $t \in [0, 1000)$ (`model.num_timesteps = 1000`) | Same |
| **Text / Consistency Weights** | `alpha_text = 5.0`, `lambda_cons = 0.0` | `alpha_text = 5.0`, `lambda_cons = 0.5`, `alpha_cons_text = 5.0` |

---

## 3. Verified Quantitative Results (`CELL 05`–`CELL 07`)

### A. Training Convergence, Compute & Peak VRAM

| Metric | LoRA 2 (`lambda_cons = 0.0`) | LoRA 3 (`lambda_cons = 0.5`) | Delta (`LoRA 3 - LoRA 2`) |
| :--- | :---: | :---: | :---: |
| **Step 1 Loss** (`total` / `rec` / `text_l1` / `cons`) | `0.9776` / `0.9776` / `0.3544` / `0.0000` | `1.0561` / `0.9776` / `0.3544` / `0.1571` | Lockstep `rec` & `text_l1` identical |
| **Step 400 Loss** (`total` / `rec` / `text_l1` / `cons`) | `0.4878` / `0.4878` / `0.2456` / `0.0000` | `0.5501` / `0.4851` / `0.2475` / `0.1299` | `-17.3%` on `cons` (`0.1571 -> 0.1299`) |
| **Mean Step Time** (`p95`) | `4.67 s/step` (`4.70 s`) | `5.38 s/step` (`5.41 s`) | `+0.71 s/step` (`+15.20%`) |
| **Peak Allocated VRAM** | **`8.04 GiB`** (`8.03` at step 1) | **`8.04 GiB`** (`8.03` at step 1) | **`0.00 GiB`** |
| **Peak Reserved VRAM** | **`8.52 GiB`** | **`8.52 GiB`** | **`0.00 GiB`** |

### B. Whole-Patch Reconstruction & 10-Seed Stability (`Real-CE Val`, `n = 30` Patches)

| Model / Variant | `PSNR_Y` (dB) $\uparrow$ | `SSIM_Y` $\uparrow$ | `LPIPS` $\downarrow$ | Cross-Seed `pair_px` (`0..255`) $\downarrow$ | Cross-Seed `pair_lpips` $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LoRA 2 VAL** | `17.6980` | `0.6442` | **`0.4231`** | `9.2715` | `0.2020` |
| **LoRA 3 VAL** | **`18.2828`** | **`0.6742`** | `0.4711` | **`3.9405`** | **`0.0726`** |
| **Delta (`LoRA 3 - LoRA 2`)** | **`+0.5848` dB** | **`+0.0300`** | `+0.0480` | **`-5.3310` (`-57.50%`)** | **`-0.1294` (`-64.06%`)** |

### C. Text-Box OCR Comparison (`RapidOCR`, `Real-CE Val`, `n = 30` Patches / `154` Boxes)

#### 1. Initial Default Two-Stage Run (`det+rec`, `--upscale 1`, `CELL 07`, Nov 15)

> **Why this run was misleading:** Running RapidOCR in default `det+rec` mode at `--upscale 1` caused `DBNet` to miss tightly cropped boxes — leaving even ground-truth `HR Ceiling` at `CER = 0.7334` (`wrong_char = 0.9291`).

| Target (`n = 30`, image-mean) | `wrong_char` Rate $\downarrow$ | Mean `CER` $\downarrow$ | Cross-Seed OCR Disagreement $\downarrow$ |
| :--- | :---: | :---: | :---: |
| **HR Ceiling (`val/HR`)** | `0.9291` | `0.7334` | — |
| **Bicubic $\times 4$ (`val/LR`)** | `0.9889` | `0.9603` | — |
| **LoRA 2 VAL (`42..51`)** | `0.9838` | **`0.8908`** | `0.3456` |
| **LoRA 3 VAL (`42..51`)** | `0.9857` | `0.9055` | **`0.2470`** |
| **Paired Delta (`LoRA 3 - LoRA 2`)** | `+0.0019` | `+0.0147` | **`-0.0986` (`-28.53%`)** |

#### 2. Corrected `--rec-only --rotate-vertical --upscale 2` Re-Score (M3, `VERIFIED` Nov 16)

Re-scoring the **same saved Drive PNGs** (`outputs/pilot-lora{2,3}-realce-val-10seed/`) with `--ocr-backend rapidocr --rec-only --rotate-vertical --upscale 2` ([`docs/M3_3_pilot_reconly_and_crops_notes.md`](M3_3_pilot_reconly_and_crops_notes.md)) fixed the detector misses (`HR Ceiling` CER dropped `0.7334 -> 0.2722`) and **reversed the earlier OCR conclusion**:

| Cohort & Target | `wrong_char` Rate $\downarrow$ | Mean `CER` $\downarrow$ | OCR Seed Disagreement $\downarrow$ | Distinct Reads / 10 | Crop Pixel Spread $L_1$ $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Image-Mean (`n = 30` patches) — HR Ceiling** | `0.5791` (`±0.0491`) | `0.2722` (`±0.0538`) | — | — | — |
| **Image-Mean (`n = 30`) — Bicubic $\times 4$** | `0.7979` (`±0.0445`) | `0.5422` (`±0.0694`) | — | — | — |
| **Image-Mean (`n = 30`) — LoRA 2 VAL** | `0.9141` (`±0.0269`) | `0.6914` (`±0.0518`) | `0.5197` (`±0.0710`) | `4.88` | `15.88` |
| **Image-Mean (`n = 30`) — LoRA 3 VAL** | **`0.8878`** (`±0.0291`) | **`0.6650`** (`±0.0549`) | **`0.4479`** (`±0.0660`) | **`4.16`** | **`7.13`** |
| **Paired Delta (`LoRA 3 - LoRA 2`, `n = 30`)** | **`-0.0262 ± 0.0084`** | **`-0.0264 ± 0.0122`** | **`-0.0718` (`-13.82%`)** | **`-0.72`** | **`-8.75` (`-55.08%`)** |
| **HR-Solvable Subset (`63 / 154` boxes, HR=0)** — Bicubic | `0.5556` | `0.4184` | — | — | — |
| **HR-Solvable Subset (`63` boxes)** — LoRA 2 / LoRA 3 | `0.8222` $\to$ **`0.7603`** (`-0.0619`) | `0.6144` $\to$ **`0.5803`** (`-0.0340`) | — | — | — |

#### 3. Visual Inspection of 5 Crop Grids (`results/pilot_lora2_vs_lora3_crops/`)
- **Mask alignment confirmed:** Text bounding boxes align accurately with `52mm` HR text; failure is not caused by mask misregistration.
- **LoRA 2 vs. LoRA 3 behavior:** LoRA 2 draws sharp strokes that hallucinate differently across seeds (`crop_02`: `"深圳柏悦酒店"` -> 6 distinct readings; `crop_05`: 10 distinct readings). LoRA 3 is far more consistent (`crop_05`: 4 distinct readings, CER `0.138 -> 0.037`), but fades/blurs strokes and occasionally erases thin characters (`crop_03` `"2024/06/17"` CER `0.570 -> 0.880`), matching `LPIPS +0.0480`.
- **Both empty-prompt 400-step pilots still lag Bicubic $\times 4$ (`0.5422` vs. `0.6650` CER), especially on dense multi-stroke Chinese characters (`酒`, `馆`, `悦`).**

---

## 4. Recommended Next Actions (Track 1 vs. Track 2)

1. **Track 1 (`Assignment 5 Baseline` — Priority Before Mon `16/11 21:00` Internal / Tue `17/11 07:00` Official):**
   - Keep Assignment 5 (`RealSRVal_crop128` DoSSR baseline) completely separate from LoRA tuning; fill the `Bicubic x4` row in `docs/A5_baseline.tex` (`Table 1`) and submit `25125002_25125042_25125060_25125033.pdf`.
2. **Track 2 (`LoRA Pilot` Tuning Next — Send Update & Get Dr. Thuận's Sign-Off):**
   - **Send Mentor Update to Dr. Thuận** including the corrected `--rec-only --upscale 2` OCR results (`wrong_char -0.0262 ± 0.0084`, `CER -0.0264 ± 0.0122`, `-0.0619` on HR-solvable boxes, `seed_disagree -13.82%`), the 5 visual crop grids, and confirmed mask alignment.
   - **Proposed Next Validation Sweep (After A5 & Mentor Approval):** Lower `lambda_cons` (`0.5 -> 0.1` or `0.2`) and/or apply consistency only for timesteps $t \le 500$ (matching DoSSR's 5-step `t1_fraction = 0.5` inference schedule) to fix thin-stroke erasure and LPIPS softening while keeping seed consistency, before adding LoRA 1 (`alpha_text=1.0, lambda_cons=0.0`).
