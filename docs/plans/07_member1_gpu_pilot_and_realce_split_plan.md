# Plan 07 — Member 1 (Team Lead & GPU) Implementation Plan: Real-CE Split & LoRA Pilot

**Role:** Member 1 (Hoàng Nguyên Anh — `25125002`) — Team Lead, GPU Runner & Submission Lead  
**Branch:** `feat/baseline-runner`  
**Internal Deadline for A5:** **Monday 16/11/2026 21:00**  
**Pilot Target Date:** **Sunday 15/11/2026 18:00**  

---

## 1. Overview & Context

This implementation plan specifies Member 1's next four concrete tasks derived from the mentor's latest guidance (Dr. Thuận):
1. **Task 1: Real-CE $\times 4$ Scene-Level Split Script (`scripts/split_realce_by_scene.py`)**  
   Split Real-CE $\times 4$ train and validation sets by root scene / original image before cropping patches to guarantee zero patch-level leakage.
2. **Task 2: Colab GPU Pilot of LoRA 2 vs. LoRA 3**  
   Run training for LoRA 2 (text-weighted) and LoRA 3 (text-weighted + cross-noise latent consistency) on Colab GPU, recording peak VRAM (allocated and reserved) and average step time (sec/step).
3. **Task 3: 10-Seed, 5-Step Validation Inference**  
   Run inference on the fixed validation set for both LoRA checkpoints using the identical 10 seeds (`42`–`51`) and 5 inference steps.
4. **Task 4: A5 Final PDF Compilation & Submission**  
   Coordinate the Number-Audit review with Member 4, compile `25125002_25125042_25125060_25125033.pdf`, and submit to the portal.

---

## 2. Step-by-Step Task Guides & Verification

### Task 1: Real-CE $\times 4$ Scene-Level Train/Val Split

#### Objective
Ensure that all patches originating from the same camera shot / scene remain strictly within either `train` or `val`. Under no circumstance should one crop of a scene be in `train` while another crop of the same scene is in `val`.

#### Execution Command
```bash
python scripts/split_realce_by_scene.py \
  --data-dir "/content/drive/MyDrive/lemmepass/data/Real-CE" \
  --split train \
  --scale x4 \
  --val-ratio 0.20 \
  --seed 42 \
  --output-manifest "data/realce_splits_manifest.json" \
  --output-train-csv "data/realce_train_manifest.csv" \
  --output-val-boxes "data/realce_val_boxes.csv" \
  --extract-patches \
  --patch-dir "/content/drive/MyDrive/lemmepass/data/RealCE_patches" \
  --hr-patch-size 512 \
  --lr-patch-size 128 \
  --stride 256
```

#### Verification Criteria
- [x] $\text{scenes}(\text{train}) \cap \text{scenes}(\text{val}) = \emptyset$ (strictly 0 overlapping scenes, verified in Colab `CELL 04`).
- [x] Output manifest records total images, total scenes, train image count, and val image count.
- [x] Text annotation boxes corresponding to validation images are exported to `data/realce_val_boxes.csv` (`154` boxes across `30` val patches).

---

### Task 2: Colab GPU Pilot Run (LoRA 2 vs. LoRA 3)

#### Objective
Execute the pilot run comparing LoRA 2 (text-weighted) and LoRA 3 (text-weighted + cross-noise latent consistency loss) to determine if the consistency loss reduces text error on the validation set.

#### Protocol
- **Common Settings:**
  - Base Checkpoint: upstream DoSSR (`687860c`)
  - Precision: `FP16`
  - Positive Prompt: Empty prompt `""`
  - Timestep sampling: Same timestep $t$ per batch
  - Noise samples: Exactly 2 independent noise samples $(\epsilon_1, \epsilon_2)$ per step
  - Steps: Fixed pilot budget (`400` optimizer steps executed in `notebooks/03_lora_pilot_colab.ipynb`; initially estimated at `100`)
  - Optimizer: AdamW, lr = $1\times 10^{-4}$
- **LoRA 2 Config (`configs/lora2_pilot.yaml`):**
  - `alpha_text = 5.0`, `lambda_cons = 0.0` *(executed setting; early draft used `lambda_text = 2.0`)*
- **LoRA 3 Config (`configs/lora3_pilot.yaml`):**
  - `alpha_text = 5.0`, `lambda_cons = 0.5`, `alpha_cons_text = 5.0` *(executed setting; early draft used `lambda_text = 2.0, lambda_cons = 1.0`)*

#### Colab Execution Commands
```python
# 0. Offline CPU dry-run verification (optional quick check before GPU run)
!python scripts/run_lora_trial.py --dry-run-cpu

# 1. Run LoRA 2 (Text-weighted base, lambda_cons=0.0)
!python scripts/run_lora_trial.py \
  --config configs/lora2_pilot.yaml \
  --train-manifest data/realce_train_manifest.csv \
  --output-dir /content/drive/MyDrive/lemmepass/outputs/lora2-pilot

# 2. Run LoRA 3 (Text-weighted + Cross-Noise Consistency, lambda_cons=0.5)
!python scripts/run_lora_trial.py \
  --config configs/lora3_pilot.yaml \
  --train-manifest data/realce_train_manifest.csv \
  --output-dir /content/drive/MyDrive/lemmepass/outputs/lora3-pilot
```

#### Metrics to Log During Run
In `lora_run_manifest.json` and `lora_train_log.csv` inside each output directory:
- `trainable_lora_params`: Total trainable LoRA parameters
- `peak_vram_allocated_gib`: `torch.cuda.max_memory_allocated() / (1024**3)`
- `peak_vram_reserved_gib`: `torch.cuda.max_memory_reserved() / (1024**3)`
- `mean_step_time_sec` & `p95_step_time_sec`: Optimizer step time (sec/step)
- Per-step `total_loss`, `rec_loss`, `global_l1`, `text_l1`, `cons_loss` in `lora_train_log.csv`

#### Verification Criteria
- [x] Checkpoints (`lora_weights_final.pt`) save cleanly to Google Drive without corruptions.
- [x] Loss values decrease smoothly without `NaN` or `Inf`.
- [x] Peak VRAM is recorded (`8.04 GiB` allocated / `8.52 GiB` reserved for both LoRA 2 and LoRA 3) and confirmed within the L4 GPU budget (< 22.5 GiB).

---

### Task 3: 10-Seed, 5-Step Inference on Fixed Validation Set

#### Objective
Generate super-resolved validation images from both LoRA checkpoints using the standard DoSSR 10 seeds (`42`–`51`), deterministic per-image seeding (`--per-image-seed`), empty prompt `""`, and 5 inference steps.

#### Execution Commands
```bash
# 10-seed validation study for LoRA 2 Checkpoint
python scripts/run_seed_study.py --config configs/lora2_pilot.yaml

# 10-seed validation study for LoRA 3 Checkpoint
python scripts/run_seed_study.py --config configs/lora3_pilot.yaml

# Or single-seed manual invocation via src/run_dossr.py:
python src/run_dossr.py \
  --config configs/lora3_pilot.yaml \
  --lora-weights /content/drive/MyDrive/lemmepass/outputs/lora3-pilot/lora_weights_final.pt \
  --per-image-seed
```

#### Verification Criteria
- [x] All validation images are generated for all 10 seeds (`seed_42`..`seed_51`) in `outputs/pilot-lora{2,3}-realce-val-10seed/` (`300` PNGs per variant).
- [x] Output images are non-blank, non-NaN, and exact $4\times$ resolution (`512x512`).
- [x] OCR evaluation (`--rec-only --rotate-vertical --upscale 2` completed by M3 on Nov 16) and `scripts/compute_metrics.py` (PSNR, SSIM, LPIPS) logged in `results/lora_pilot_summary.json`.

---

### Task 4: Assist Member 4 & Submit Assignment 5 PDF

#### Objective
Ensure Assignment 5 baseline report is 100% verified, compiled, and submitted before the internal deadline (**Monday 16/11 21:00**).

#### Verification Checklist
- [x] Review `docs/A5_baseline.tex` tables against `docs/fact_sheet_for_report.md`:
  - 10-seed whole-image table: `PSNR_RGB 22.6353 dB` (`22.64 dB`) / `PSNR_Y 24.1610 dB`, `SSIM_RGB 0.6406` / `SSIM_Y 0.6820`, `LPIPS 0.3382` (Default prompt); `Bicubic x4` whole-image row **`VERIFIED` Nov 16** on Colab (`PSNR_RGB 25.48 dB`, `SSIM_RGB 0.7189`, `PSNR_Y 27.11 dB`, `SSIM_Y 0.7561`, `LPIPS 0.4563`, saved in `results/metrics_realsr_bicubic_x4_summary.csv`).
  - 193-box vs 136-box solvable vs 57-box unsolvable breakdown table verified.
  - Pairwise disagreement formula ($32.83\%$ on 193 boxes, $34.05\%$ on 136 solvable) included.
  - Eduroam `seed 42` source audit (1-img vs 100-img RNG stream + crop framing) included.
- [x] Compile PDF locally cleanly (`pdflatex -output-directory=/tmp docs/A5_baseline.tex` -> exit 0, `/tmp/A5_baseline.pdf` 4 pages).
- [x] Fill `Bicubic x4` whole-image row after Colab `compute_metrics.py --bicubic-lr-dir` run (done in `9c78cb6`); [ ] Final Overleaf re-compile to `25125002_25125042_25125060_25125033.pdf`, upload to course portal, and archive screenshot.

---

## 3. Implementation Checklist for Member 1

```text
[x] Author scripts/split_realce_by_scene.py (--split train, LR/HR pairing, safe scene grouping, --extract-patches).
[x] Run split script on Real-CE on Drive and verify zero train/val scene overlap + populate data/realce_val_boxes.csv.
[x] Launch Colab Pilot via notebooks/03_lora_pilot_colab.ipynb for LoRA 2 and LoRA 3 (400 steps each).
[x] Record peak allocated VRAM, reserved VRAM, and step time from lora_run_manifest.json.
[x] Run 10-seed 5-step validation inference via scripts/run_seed_study.py --config configs/lora{2,3}_pilot.yaml.
[x] Compute Bicubic x4 whole-image baseline via scripts/compute_metrics.py --bicubic-lr-dir for A5 Table 1.
[x] Obtain pilot text error evaluation (`--rec-only --upscale 2` completed by M3 in `results/pilot_lora2_vs_lora3_reconly/`).
[x] Cross-check A5 numbers with Member 4 against docs/fact_sheet_for_report.md & results/text_eval_a5_summary.json.
[ ] Compile 25125002_25125042_25125060_25125033.pdf on Overleaf and submit before Tue 17/11 07:00.
```
