# Implementation Plan 01: Standardized `01_dossr_baseline.ipynb`, `scripts/check_attention_sdpa.py`, Docs & `context.md`

**Status:** Approved & Implemented (`dfc8d96`)

## 1. Summary of Approved Plan
1. **`CELL 11` Exact Gate Rule & Named Constant (`SEED_SPREAD_MIN_RATIO = 3.0`):**
   - If `mean_abs_diff(42 vs 42-repeat) == 0.0`, pass when `mean_abs_diff(42 vs 43) > 0.0` (and `max_abs_diff(42 vs 43) > 0.0`).
   - Otherwise, pass when `mean_abs_diff(42 vs 43) >= SEED_SPREAD_MIN_RATIO * mean_abs_diff(42 vs 42-repeat)`.
   - On failure, write `cell11_seed_sanity_status.json` with `"passed": False` and all recorded numbers first, then raise `RuntimeError`.
2. **Safe Pixel Diff Arithmetic (Float32 / Int16, Reported in 8-Bit `[0..255]` Levels):**
   - Convert loaded RGB images with `np.asarray(im, dtype=np.float32)` before subtracting — never subtract `uint8` arrays directly. Report `max_abs_diff` and `mean_abs_diff` in 8-bit pixel levels (`0..255`).
3. **Provenance Fields & Commit Hash Check in Status JSONs (`CELL 10`, `CELL 11`, `CELL 12`, `CELL 13`):**
   - Both `cell10_inspection_status.json` and `cell11_seed_sanity_status.json` record `"passed"`, `"git_commit"`, `"timestamp_utc"`, and `"checked_image_paths"`.
   - `CELL 12` and `CELL 13` read both JSON files from disk, check `"passed" is True`, and verify `"git_commit" == current_head_hash`, refusing to run on any mismatch.
4. **How `seed 43` and `seed 42-repeat` Are Configured (`CELL 11`):**
   - Both extra runs write dedicated YAML config files with separate `output_root`s and invoke `bash scripts/run_baseline.sh --config <yaml> --fp16`.
5. **`scripts/check_attention_sdpa.py` GPU Flag & Larger VAE `AttnBlock` Shape:**
   - Adds `--device` (`choices=("cpu", "cuda", "auto")`, default `"auto"`).
   - Tests `CrossAttention`, tiny `VAE AttnBlock` (`2 x 16 x 6 x 5`), and a larger `VAE AttnBlock` shape (`1 x 512 x 64 x 64`) on the selected device.
6. **All Cells & New Checks Labeled `NOT TESTED`:**
   - Every cell in `notebooks/01_dossr_baseline.ipynb` and every new check in `docs/` and `context.md` is explicitly labeled `NOT TESTED` until Colab logs are returned.
