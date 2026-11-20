# Plan 08 — Phase-2 LoRA Sweep ($\lambda \in \{0.1, 0.2\}$, $t \le 499$), Math & Gradient Audit, Base DoSSR Validation, & 2,000-Step Curve

**Date:** November 21, 2026  
**Team:** LEMMEPASS (Team 1, SC203 Scientific Methods — Class 25A02)  
**Target Notebook:** [`notebooks/04_lora_sweep_and_2000step_colab.ipynb`](../../notebooks/04_lora_sweep_and_2000step_colab.ipynb)  
**Audit Output JSON:** [`results/realce_val_split_and_category_audit.json`](../../results/realce_val_split_and_category_audit.json)

---

## 1. Exact Mathematical Audit of DoSSR Loss, $\hat{x}_0$ Conversion, $\eta_t / (1-\eta_t)$ Schedule, & FP32 Backward Equivalence

### 1.1. How DoSSR Converts Model Output $\hat{\epsilon}_\theta$ into Clean HR Latent Prediction $\hat{x}_0$
In [`external/DOSSR/ldm/models/diffusion/ddpm.py:909-917`](../../external/DOSSR/ldm/models/diffusion/ddpm.py#L909-L917) (`predict_x0_out_mix`, `parameterization == "eps"`), given clean HR VAE latent $x_0 = E(\text{HR}) \cdot 0.18215$, preprocessed LR latent $\hat{x}_{\text{LR}} = E_{\text{cond}}(\text{RealESRNet}(\text{LR})) \cdot 0.18215$, conditioning $c$, timestep $t \in \{0, \dots, 999\}$, and Gaussian noise $\epsilon \sim \mathcal{N}(0, I)$:

1. **Domain-shifted clean interpolation (`ddpm.py:912-913`):**
   $$x_{\text{rect}}(t) = \eta_t \, \hat{x}_{\text{LR}} + (1 - \eta_t) \, x_0$$
2. **Forward diffusion noising (`ddpm.py:914`, `q_sample`):**
   $$x_t(\epsilon) = \sqrt{\bar{\alpha}_t} \, x_{\text{rect}}(t) + \sqrt{1 - \bar{\alpha}_t} \, \epsilon$$
3. **UNet + ControlNet $\epsilon$-prediction (`ddpm.py:915`):**
   $$\hat{\epsilon}_\theta = \text{ControlledUNet}\big(x_t(\epsilon), t, c_{\text{crossattn}}, \text{ControlNet}(x_t(\epsilon), \hat{x}_{\text{LR}}, t, c_{\text{crossattn}})\big)$$
4. **Conversion from $\hat{\epsilon}_\theta$ to predicted clean HR latent $\hat{x}_0$ (`ddpm.py:916`, `predict_start_from_noise` at `ddpm.py:293-297`):**
   $$\hat{x}_0(\epsilon) = \frac{1}{\sqrt{\bar{\alpha}_t}} \, x_t(\epsilon) - \sqrt{\frac{1}{\bar{\alpha}_t} - 1} \; \hat{\epsilon}_\theta(x_t(\epsilon), t, c)$$

### 1.2. Behavior of $1 - \eta_t$ vs. $\sqrt{\bar{\alpha}_t}$ Near $t = 500$ and Selection of Timestep Cutoff
From [`external/DOSSR/ldm/modules/diffusionmodules/util.py:32-36`](../../external/DOSSR/ldm/modules/diffusionmodules/util.py#L32-L36) (`make_eta_schedule("trunc_12", 1000)`) and [`external/DOSSR/model/shift_sampler.py:48-63, 118-158`](../../external/DOSSR/model/shift_sampler.py#L48-L63), verified via `python src/lora_text_loss.py --audit-eta-schedule`:

- **Does training divide by $1 - \eta_t$?** **No.** In `predict_x0_out_mix`, conversion to $\hat{x}_0$ divides only by $\sqrt{\bar{\alpha}_t}$ (where $\sqrt{\bar{\alpha}_{499}} = 0.526944$, $\sqrt{\bar{\alpha}_{999}} = 0.068265$), so training never encounters a $0/0$ singularity in the loss formula itself.
- **Where does $1 - \eta_t$ vanish, and what happens near $\eta_t = 1$?**
  - For $t \in [0, 499]$ ($t < t_1 = 500$): $s_t = \text{linspace}(0.001, 0.999, 500)$ and $\eta_t = \frac{1 - \cos(\pi s_t)}{2} \in [2.4674 \times 10^{-6}, 0.99999753]$. At the active boundary $t = 499$, $\eta_{499} = 0.9999975326$ and **$1 - \eta_{499} = 2.4674 \times 10^{-6} > 0$**.
  - For $t \in [500, 999]$: **$\eta_t \equiv 1.0$ and $1 - \eta_t \equiv 0.0$ identically**. Consequently, $x_{\text{rect}}(t) = \hat{x}_{\text{LR}}$ contains **zero ground-truth HR latent $x_0$**, and inference (`SpacedSampler` in `shift_sampler.py:157, 236`, which divides by $1 - \eta_t$) excludes all $t \ge 500$.
- **Exact 5-Step Inference Timesteps (`space_timesteps(500, "6")`):**
  - `shift_sampler.py:48-61` uses `frac_stride = 499 / 5 = 99.8`, yielding spaced knots `[0, 100, 200, 299, 399, 499]` (`round(299.4) = 299`, `round(399.2) = 399`). The 5 reverse-sampling UNet evaluations occur strictly at **`t in [499, 399, 299, 200, 100]`** (all $\le 499$).
- **Chosen Timestep Cutoff for Consistency Ablation:** **`cons_t_min = 0, cons_t_max = 499` ($t \le 499$, i.e. $t < 500$)**. Reconstruction loss remains active over all $t \in [0, 999]$ while cross-noise consistency loss $\mathcal{L}_{\text{cons}}$ is enabled exclusively on $[0, 499]$ (where $1 - \eta_t > 0$ and where the 5-step solver operates).

### 1.3. Verified FP32 Gradient Check: Custom Two-Pass Backward vs. Simultaneous Joint Graph
Executed via `/opt/miniconda3/envs/sc203-local/bin/python src/lora_text_loss.py --run-cpu-tests --verify-fp32-gradients`:

| Test Mode (`FP32`, `lambda_cons=0.2`, `alpha_text=5.0`) | Joint Loss | Two-Pass Loss | Loss Abs Diff | Max Abs Grad Diff | Mean Abs Grad Diff | Relative $L_2$ Grad Diff |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`grad_accum = 1`** (single minibatch) | `2.50256610` | `2.50256628` | `1.79e-07` | **`0.0`** | **`0.0`** | **`0.0`** |
| **`grad_accum = 4`** (4-minibatch accumulation) | `2.54860532` | `2.54860538` | `5.66e-08` | **`1.49e-08`** | **`1.70e-09`** | **`4.11e-08`** |

Additionally, passing `--verify-fp32-grad` to `scripts/run_lora_trial.py` executes `run_controlldm_fp32_grad_check(...)` on the loaded `ControlLDM` in FP32 across all `1,659,904` LoRA parameters before step 1 (restoring exact initial weights and RNG states afterward).

---

## 2. Minibatch Accounting & Sweep Matrix (`grad_accum = 4`, `batch_size = 1`, `K = 2`)

- **400-step sweep runs (`lora3_lambda01_pilot.yaml`, `lora3_lambda02_pilot.yaml`, `lora3_t499_pilot.yaml`):**
  - **400 optimizer updates** $\times$ `grad_accum = 4` = **`1,600` minibatches** ($\times K=2$ = **`3,200` noisy forward passes**).
  - All runs start from identical base checkpoint (`dossr_default.ckpt`), identical LoRA init (`rank=8, alpha=16.0`), identical training seed (`231`), identical dataset (`data/realce_train_manifest.csv`), identical batch sequence, and empty prompt `""`.
- **2,000-step full runs (`lora2_full2000_s231.yaml`, `lora3_full2000_s231.yaml`):**
  - **2,000 optimizer updates** $\times$ `grad_accum = 4` = **`8,000` minibatches** ($\times K=2$ = **`16,000` noisy forward passes**).
  - Intermediate checkpoints saved every 400 optimizer steps: `step_0400.pt`, `step_0800.pt`, `step_1200.pt`, `step_1600.pt`, and `lora_weights_final.pt` (`step 2000`).

---

## 3. Verified Validation Baseline & Category Breakdown (`Real-CE Val`: 30 Patches, 154 Boxes, 63 HR-Solvable)

Computed via `python3 scripts/analyze_realce_lora_sweep.py` (`results/realce_val_split_and_category_audit.json`):

### 3.1. Existing Methods vs. Phase-2 Placeholders (Fixed 10 Validation Seeds `42..51`, `per_image_seed: true`)

| Method | All-154 BoxErr | All-154 CER | All-154 Disag | Solv-63 BoxErr | Solv-63 CER | Solv-63 Disag | Solv CER W/T/L | Solv BoxErr W/T/L | Patch LPIPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **HR Ceiling** | 59.09% | 0.2908 | --- | 0.00% | 0.0000 | --- | --- | --- | 0.0000 |
| **Bicubic $\times 4$** | 80.52% | 0.5770 | --- | 55.56% | 0.4184 | --- | --- | --- | *Run CELL 03* |
| **Base DoSSR (unmodified)** | *Run CELL 03* | *Run CELL 03* | *Run CELL 03* | *Run CELL 03* | *Run CELL 03* | *Run CELL 03* | *Run CELL 03* | *Run CELL 03* | *Run CELL 03* |
| **LoRA 2 ($\lambda=0.0$, 400st)** | 91.82% | 0.7241 | 47.69% | 82.22% | 0.6144 | 51.71% | 0 / 63 / 0 | 0 / 63 / 0 | 0.4231 |
| **LoRA 3 ($\lambda=0.1$, 400st)** | *Run CELL 04* | *Run CELL 04* | *Run CELL 04* | *Run CELL 04* | *Run CELL 04* | *Run CELL 04* | *Run CELL 04* | *Run CELL 04* | *Run CELL 04* |
| **LoRA 3 ($\lambda=0.2$, 400st)** | *Run CELL 04* | *Run CELL 04* | *Run CELL 04* | *Run CELL 04* | *Run CELL 04* | *Run CELL 04* | *Run CELL 04* | *Run CELL 04* | *Run CELL 04* |
| **LoRA 3 ($\lambda=0.5$, 400st)** | 89.42% | 0.6942 | 41.21% | 76.03% | 0.5803 | 41.73% | 31 / 22 / 10 | 17 / 46 / 0 | 0.4711 |
| **LoRA 3 ($\lambda^*, t \le 499$)** | *Run CELL 05* | *Run CELL 05* | *Run CELL 05* | *Run CELL 05* | *Run CELL 05* | *Run CELL 05* | *Run CELL 05* | *Run CELL 05* | *Run CELL 05* |

### 3.2. Verified Diagnostic Category Breakdown (63 HR-Solvable Cohort)
*Why lowering $\lambda_{\text{cons}}$ is needed:* In `numeric_or_date` (24 solvable boxes / 50 total), Pilot LoRA 3 ($\lambda=0.5$) actually edged worse on mean CER (`0.5854` $\to$ `0.5865`) despite reducing box error (`81.2%` $\to$ `76.2%`), matching the visual erasure of thin slashes/digits in `"2024/06/17"`. Meanwhile, `latin_only` (`16` solvable / `67` all), `border_touching` (`27` / `73`), and `cjk_signage` (`35` / `68`) all improved in both CER and Box Error Rate.

---

## 4. Real-CE 945-Pair Provenance & Validation Scene IDs

- **Why `945` total vs. `645` train / `300` val (`results/realce_feasibility_summary.json`):**
  - The full `Real-CE` dataset contains **945 paired $(13\text{mm}, 52\text{mm})$ images** for $4\times$:
    - **645 pairs** in official `RealCE/train/`
    - **300 pairs** in official `RealCE/val/`
  - `scripts/split_realce_by_scene.py --split train --scale x4 --val-ratio 0.20 --seed 42` partitions **only the 645 official `RealCE/train/` pairs** by `scene_key` (`Train scenes ∩ Val scenes = ∅`), while keeping all **300 official `RealCE/val/` pairs completely untouched** for final test evaluation.
- **Populated Scene ID Export:** Running CELL 02 in [`notebooks/04_lora_sweep_and_2000step_colab.ipynb`](../../notebooks/04_lora_sweep_and_2000step_colab.ipynb) exports the complete `train_scenes`, `val_scenes`, `train_pairs`, and `val_pairs` ID lists (`data/realce_splits_manifest.json` & Drive archive).
- **The 30 Validation Patches & Their 30 Distinct Parent Scenes (`results/realce_val_split_and_category_audit.json`):**
  Each validation patch corresponds 1-to-1 to a unique parent scene ID (`ip11pro_output_*` and `ip12pro_output_*`), archived in `results/realce_val_split_and_category_audit.json`.

---

## 5. Team Task Division for Phase-2 Deliverables

| Member | Assigned Tasks | Verification / Outputs |
| :--- | :--- | :--- |
| **M1 (Hoàng Nguyên Anh)** | Run [`notebooks/04_lora_sweep_and_2000step_colab.ipynb`](../../notebooks/04_lora_sweep_and_2000step_colab.ipynb) on Colab L4 (CELL 01–06): FP32 `ControlLDM` check, Base DoSSR val run, $\lambda \in \{0.1, 0.2\}$ sweep, $t \le 499$ gating run, 2,000-step runs + checkpoint evaluations, and archive `realce_splits_manifest_populated.json`. | Drive outputs, `lora_train_log.csv`, `lora_run_manifest.json`, peak VRAM & `s/step` logs, checkpoint curve JSONs. |
| **M2 (Huỳnh Trần Gia Hân)** | Execute manual qualitative inspection on date/numeric (`2024/06/17`), Latin (`paas`), and border-touching (`ip11pro_output_TextSR_renew_101_patch000` boxes 3 & 5, `renew_114` box 5) crop grids across `Base DoSSR`, `0.0`, `0.1`, `0.2`, `0.5`, `t<=499`, and `step 2000`; log concrete improved vs. degraded cases. | Updated crop inspection notes & failure/success comparison PNGs. |
| **M3 (Hồ Quang Minh)** | Verify `src/lora_text_loss.py` & `scripts/run_lora_trial.py` math/FP32 gradient logs and `cons_active_ratio` / sampled-$t$ stats; run `scripts/analyze_realce_lora_sweep.py` on all Phase-2 Drive CSVs to verify zero loss bugs and extract checkpoint curves. | `phase2_lambda_sweep_comparison.json`, `phase2_timestep_gating_comparison.json`, `phase2_2000step_curves_s231.json`. |
| **M4 (Nguyễn Phan Minh Nhật)** | Assemble the mentor Phase-2 evaluation report integrating the 6 sections required by Dr. Thuận (math/FP32 check, minibatch accounting, full validation comparison table, timestep-gating table, 2,000-step curve plot + VRAM/timing, and 945-pair split provenance + ID lists). | Final Phase-2 Mentor Report (Markdown/PDF + plots). |
