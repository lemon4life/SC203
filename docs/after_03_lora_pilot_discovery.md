# Discovery After `03_lora_pilot_colab.ipynb`: LoRA 2 vs. LoRA 3 Pilot on Real-CE $\times 4$ (`VERIFIED`, Nov 15, 2026)

**Status:** `VERIFIED` — Trained (`400` steps each) and evaluated across `10` seeds (`42..51`, `5` DoSSR steps, empty prompt `""`) on `30` scene-disjoint Real-CE $\times 4$ validation patches (`300` images per variant) via `notebooks/03_lora_pilot_colab.ipynb` (`CELL 05`–`CELL 07`).  
**Structured JSON:** [`results/lora_pilot_summary.json`](../results/lora_pilot_summary.json)

---

## 1. The Discovery in Simple Words (For the Whole Team)

Think of DoSSR as a skilled artist redrawing a blurry zoomed-in sign. Whenever you run DoSSR with a different random seed (`seed 42` vs `seed 43`), it is like rolling dice before drawing: **LoRA 2** fixes some spelling, *but still changes its mind wildly whenever you change the seed*.

We tested whether adding **cross-noise consistency (`LoRA 3`)** forces the model to stop rolling the dice inside text regions while keeping the exact same backbone, dataset split, random seed (`231`), step budget (`400` steps), and GPU memory limit.

### What Happened? (The 3 Big Wins + 2 Subtle Findings)

1. **Seed randomness collapsed by ~60% (Big Win):**  
   Across 10 random seeds on held-out scenes never seen during training, **LoRA 3** cut pixel-to-pixel jumping across seeds by **`-57.5%`** (`9.27 -> 3.94` gray levels out of 255), cut perceptual texture jumping (`pair_lpips`) by **`-64.1%`** (`0.2020 -> 0.0726`), and cut how often OCR changed its mind across seeds by **`-28.5%`** (`34.56% -> 24.70%`).
2. **Closer to ground-truth pixels (Win):**  
   LoRA 3 improved whole-patch **PSNR_Y by `+0.5848 dB`** (`17.6980 -> 18.2828`) and **SSIM_Y by `+0.0300`** (`0.6442 -> 0.6742`).
3. **Zero extra GPU memory (`0.00 GiB` overhead):**  
   Because our two-pass chain-rule backward (`src/lora_text_loss.py`) frees pass 1's compute graph before running pass 2, **LoRA 3 used the exact same peak GPU memory as LoRA 2 (`8.04 GiB` allocated / `8.52 GiB` reserved)**, costing only `+15.2%` extra step time (`4.67s -> 5.38s/step`).
4. **Why sharpness (`LPIPS`) and OCR got slightly softer (`LPIPS +0.0480`, `CER +0.0147`):**  
   Imagine forcing two artists drawing from *completely different noisy sketches* to produce the exact same final strokes — if we penalize differences too hard (`lambda_cons = 0.5`) even at very noisy diffusion timesteps ($t$ up to `999`), the safest way for the model to agree across noise draws is to draw *softer, slightly averaged letter edges* (like overlapping sheets of tracing paper). Notably, DoSSR's 5-step inference (`t1_fraction = 0.5`) only operates at $t \le 500$, whereas training penalized consistency across all uniform timesteps $t \in [0, 999)$ (`scripts/run_lora_trial.py`).
5. **Why even `HR Ceiling` had `CER = 0.7334` in `CELL 07`:**  
   In `CELL 06`, `scripts/text_eval.py` ran in default two-stage mode (`det+rec`, `--upscale 1`, without `--rec-only --rotate-vertical`). On tight word crops touching the crop border, RapidOCR's first-stage text detector (`DBNet`) misses the text box and outputs empty text — the **exact same detector quirk** Member 3 (Quang Minh) identified on RealSR in commit `ca127fe`. Re-running `scripts/text_eval.py --rec-only --rotate-vertical --upscale 2` on the **already-saved** Drive outputs requires **zero GPU retraining** and takes ~30 seconds.

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

### C. Text-Box OCR Comparison (`RapidOCR`, `Real-CE Val`, Image-Mean over `n = 30` Patches)

> **Evaluation mode note:** Run in `CELL 06` using default RapidOCR `det+rec` (`--upscale 1`, without `--rec-only --rotate-vertical`). Re-scoring the saved Drive outputs with `--rec-only --rotate-vertical --upscale 2` is queued for Member 2 (`Gia Hân`) / Member 3 (`Quang Minh`) before deciding whether to retune `lambda_cons` or restrict consistency to $t \le 500$.

| Target (`n = 30`) | `wrong_char` Rate $\downarrow$ | Mean `CER` $\downarrow$ | Cross-Seed OCR Disagreement $\downarrow$ |
| :--- | :---: | :---: | :---: |
| **HR Ceiling (`val/HR`)** | `0.9291` | `0.7334` | — |
| **Bicubic $\times 4$ (`val/LR`)** | `0.9889` | `0.9603` | — |
| **LoRA 2 VAL (`42..51`)** | `0.9838` | **`0.8908`** | `0.3456` |
| **LoRA 3 VAL (`42..51`)** | `0.9857` | `0.9055` | **`0.2470`** |
| **Paired Delta (`LoRA 3 - LoRA 2`)** | `+0.0019` | `+0.0147` | **`-0.0986` (`-28.53%`)** |

---

## 4. Recommended Next Actions (Track 1 vs. Track 2)

1. **Track 1 (`Assignment 5 Baseline` — Priority Before Mon `16/11 21:00` Internal / Tue `17/11 07:00` Official):**
   - Do **not** derail Assignment 5 (`RealSRVal_crop128` DoSSR baseline) with LoRA tuning before submission. At most, include a short 1-paragraph summary in Future Work showing the verified `-57.5%` pixel / `-64.1%` LPIPS / `-28.5%` OCR seed-variance reduction at `0.00 GiB` VRAM overhead.
2. **Track 2 (`LoRA Pilot` Diagnostic Before Any New GPU Run):**
   - **Zero-GPU OCR re-score (~30s):** Re-run `scripts/text_eval.py --rec-only --rotate-vertical --upscale 2` on the already-saved `pilot-lora2-realce-val-10seed` and `pilot-lora3-realce-val-10seed` folders in Drive to measure true character accuracy on tight Real-CE crops.
   - **Send Mentor Update to Dr. Thuận** with the verified pilot table and diagnosis (`LPIPS +0.048`, `CER +0.0147` vs `-57.5%` / `-64.1%` / `-28.5%` seed variance reduction).
   - **Candidate Next Ablation (Once Dr. Thuận Confirms):** Test either lowering `lambda_cons` (`0.5 -> 0.1` or `0.2`) and/or gating consistency to low-to-mid noise timesteps ($t \le 500$ matching DoSSR's `t1_fraction = 0.5` inference regime) to preserve LPIPS sharpness while keeping seed stability, and add LoRA 1 (`alpha_text=1.0, lambda_cons=0.0`) after gating passes.
