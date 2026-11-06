# Implementation Plan 04 — 4-Member Handoffs, Tasks B.5 & C.9/C.10, and Group A (Seeds & Prompts)

**Status:** Approved by user on Nov 6, 2026 with 8 required revisions (incorporated below).

---

## 1. Approved Execution Order (Stop and Report After Each Group)

1. **Group 1 (Immediate — Unblock Teammates 2, 3, and 4 Today):**
   - **Task B.5:** `scripts/make_text_contact_sheets.py` + `data/text_boxes_realsr_template.csv` (with header `image,box_id,x,y,w,h,transcript,readable`).
   - **Task C.9 & C.10 (Read-Only Audit + Estimates):** `docs/lora_feasibility_and_loss_design.md` covering full training entry point (`scripts/train.py`, `configs/train_cldm.yaml`), training prompt (`""` in `dataset/realesrgan.py:179`) vs inference prompt (`"clean, high-resolution, 8k"`), `shared_step` and `predict_x0_out_mix` equations, `_dequeue_and_enqueue` and `torch.no_grad()` caveats, and LoRA trial estimates (`ESTIMATE — NOT TESTED`).
   - **Handoff Docs:** `docs/handoff_member2.md`, `docs/handoff_member3.md`, `docs/handoff_member4.md` (pure Python + `pathlib` + CLI paths, zero `/content` hardcoding, compatible Python/package versions, Drive instructions, text labeling rules).
   - **Status Table:** Update `context.md` (members report in chat; only Member 1 / agent updates `context.md`; Member 1 merges via PRs).
2. **Group 2 (Next — After Reporting Group 1):**
   - **Task A.1:** `configs/seeds_10.yaml` (`42..51`).
   - **Task A.3:** `--prompt-mode {default,empty}` in `src/dossr_inference_compat.py` and `src/run_dossr.py` (recording prompt mode, exact prompt string, and SDPA flags in `run_manifest.json`, and stopping with an error if `run_manifest.json` does not say `precision: "fp16"`).
   - **Task A.4:** 100-image $\times$ 10-seed $\times$ 2-prompt configs (`SDPA OFF`) + `CELL 16` (prints exact prompt string in each mode; verifies `prompt_mode=default` gives `0.0` pixel diff vs no wrapper and `empty` gives a non-zero pixel diff) and `CELL 17` in `notebooks/01_dossr_baseline.ipynb`.
3. **Group 3 (After Group 2):**
   - **Tasks B.6, B.7, B.8 & Group D:** Present 2–3 OCR options and write `scripts/text_eval.py` (using the exact same text normalization rules as `docs/handoff_member2.md`), write `scripts/plot_eduroam_figure.py` (and ask user for the `eduroam` crop box), and finish remaining doc updates.

---

## 2. Canonical Text-Box CSV Schema & Text Rules

```csv
image,box_id,x,y,w,h,transcript,readable
```

- `image`: Filename in `test_HR` (e.g., `Canon_001_LR4.png`).
- `box_id`: Integer starting at `1` per image (`0` if an image has no text boxes).
- `x,y,w,h`: Integer pixel bounding box in `512x512` HR image coordinates.
- `transcript`: Human ground-truth transcription (1 word or 1 short single-line span per box).
- `readable`: `yes` or `no`.
- **Shared Normalization Rules (`docs/handoff_member2.md` & `scripts/text_eval.py`):**
  - Strip leading/trailing whitespace and punctuation, collapse multiple spaces to one space, and evaluate both case-insensitive (`lower()`, primary) and case-sensitive (secondary).
