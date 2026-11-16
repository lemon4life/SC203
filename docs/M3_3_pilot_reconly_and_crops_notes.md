# M3-3: LoRA 2 vs LoRA 3 pilot — corrected OCR re-score + crop check

Hồ Quang Minh (M3) · 16/11/2026 · Real-CE ×4 val: 30 patches / 154 boxes, 10 seeds (42–51)

No new training. This re-scores the saved pilot outputs (`outputs/pilot-lora2-realce-val-10seed`, `outputs/pilot-lora3-realce-val-10seed`).

## OCR re-score

Command: `scripts/text_eval.py --ocr-backend rapidocr --rec-only --rotate-vertical --upscale 2`

HR ceiling CER dropped from 0.73 to **0.27**, so the earlier numbers (det+rec, upscale 1) were wrong: the detector missed text in tight crops.

| | wrong_char | CER | seed disagree |
|---|---|---|---|
| HR | 0.579 | 0.272 | – |
| Bicubic ×4 | 0.798 | 0.542 | – |
| LoRA 2 | 0.914 | 0.691 | 0.520 |
| LoRA 3 | 0.888 | 0.665 | 0.448 |
| **LoRA 3 − LoRA 2** | **−0.026** | **−0.026** | **−0.072 (−13.8%)** |

- With correct OCR, **LoRA 3 is better than LoRA 2** (−0.026 CER, −13.8% seed disagreement). The earlier conclusion "LoRA 3 makes CER worse (+0.0147)" came from the OCR error.
- Caveat: n = 30, so the difference is still small compared with the SE (±0.03–0.05 per row).

## Crop check

Images in `results/pilot_lora2_vs_lora3_crops/` (row 1: LR / LoRA 2 / LoRA 3 / HR; rows 2–3: 5 seeds of LoRA 2 and LoRA 3). Per-seed OCR reads in [`results/pilot_lora2_vs_lora3_crops/crop_notes.txt`](../results/pilot_lora2_vs_lora3_crops/crop_notes.txt).

- LoRA 2 is sharp but **invents strokes** that change with the seed (e.g. #2 "深圳柏悦酒店").
- LoRA 3 is more stable (#5 "G+24h生活馆": 4 distinct reads vs 10 for LoRA 2) but its **strokes are faded/blurred**, and thin text is sometimes erased (#3 date "2024/06/17"). This matches LPIPS +0.048.
- Boxes line up with the text; no mask problem seen.
- Both are worse than bicubic on complex Chinese characters (酒, 馆, 悦).

### Additional Paired-SE & HR-Solvable Cohort Breakdown (Added During Repo Integration)
- **Paired image-level SE (`n = 30` patches):** $\Delta\text{wrong\_char} = -0.0262 \pm 0.0084$, $\Delta\text{CER} = -0.0264 \pm 0.0122$, and text-crop pixel seed spread `crop_pixel_seed_spread_l1` drops **`-55.08%`** (`15.8839 -> 7.1348`). Because images are paired (`LoRA 3 - LoRA 2` on the same 30 patches), the paired SE (`±0.0084` / `±0.0122`) is much smaller than the per-row unpaired SE (`±0.03–0.05`).
- **HR-solvable subset (`63 / 154` boxes = `40.9%` where `HR wrong_char == 0`):** `Bicubic x4` gets `wrong_char = 0.5556` (`CER = 0.4184`), `LoRA 2` gets `0.8222` (`0.6144`), and `LoRA 3` gets `0.7603` (`0.5803`), giving **`-0.0619`** (`-6.19` pp) lower box error rate and **`-0.0340`** lower CER for LoRA 3 vs. LoRA 2.

## Proposal (needs Dr. Thuận's approval)

- Lower λ_cons from 0.5 to 0.1–0.2, and/or
- apply the consistency loss only for t ≤ 500 (matching inference t1_fraction = 0.5).

## Files (Relocated into Canonical Repo Hierarchy)

- [`results/pilot_lora2_vs_lora3_reconly/`](../results/pilot_lora2_vs_lora3_reconly/): HR ceiling, bicubic floor, LoRA 2, paired delta
- [`results/pilot_lora3_reconly/`](../results/pilot_lora3_reconly/): LoRA 3 standalone (seed disagreement)
- [`data/realce_val_boxes.csv`](../data/realce_val_boxes.csv): the 154 val boxes (rebuilt from the public Real-CE zip, seed 42)
- [`results/pilot_lora2_vs_lora3_crops/`](../results/pilot_lora2_vs_lora3_crops/): 5 crop grids + `crop_notes.txt`
