# After `01_dossr_baseline.ipynb` Discovery Report (`2026-11-05`)

**Team:** LEMMEPASS (Team 1, SC203 Scientific Methods — Class 25A02)  
**Branch / Commit Executed in Colab:** `feat/baseline-runner` @ `5e6a777`  
**Hardware Confirmed (`CELL 03`):** `NVIDIA A100-SXM4-80GB` (`79.25 GiB` total VRAM)

---

## 1. Executive Summary

Running `CELL 01` through `CELL 11` of `notebooks/01_dossr_baseline.ipynb` on Google Colab established three key facts:

1. **Dataset Subfolder Discovery (`test_HR` vs. `test_LR` in `RealSRVal_crop128`):**
   - `CELL 04` and a follow-up inspection revealed that `RealSRVal_crop128` is not a flat folder of 200 low-resolution images. It contains two subdirectories with identical filenames (`Canon_001_LR4.png`, etc.):
     - `RealSRVal_crop128/test_HR`: **100 high-resolution ground-truth images (`512x512`)**
     - `RealSRVal_crop128/test_LR`: **100 low-resolution input images (`128x128`)**
   - Because `configs/baseline_realsr.yaml` originally set `input_dir` to the parent directory `RealSRVal_crop128` and `scripts/prepare_baseline_smoke.py` used `rglob("*")` sorted alphabetically, `test_HR/Canon_001_LR4.png` (`512x512`) was staged before `test_LR/Canon_001_LR4.png` (`128x128`).
   - Feeding a `512x512` `test_HR` image into DoSSR with `sr_scale: 4` upscaled it to **`2048x2048`**.
   - *(Note on earlier runs: only the very first, unpatched FP16 run on `2048x2048` hit the 80 GiB OOM in `CrossAttention`. After the `CrossAttention` SDPA patch, `2048x2048` runs succeeded at `36.04 GiB` allocated in FP16 and `49.29 GiB` allocated in FP32.)*
2. **VAE `AttnBlock` SDPA Patch Verified on `2048x2048` (`-15.53 GiB` Peak Allocated VRAM):**
   - `CELL 09` proved that patching the single `AttnBlock` in the VAE middle block (`self.mid.attn_1`) with PyTorch SDPA reduced FP16 `peak_allocated` on the `2048x2048` run from **`36.04 GiB` down to `20.51 GiB`** (`-43%`) and `peak_reserved` from **`52.70 GiB` down to `34.71 GiB`** (`-34%`).
3. **1-Image Same-Seed Control (`0.0`) vs. Cross-Seed Spread (`mean = 2.9309 px`, `max = 168.0 px`) Verified:**
   - `CELL 11` confirmed that running `seed 42` twice (`42` vs `42-repeat`) in separate `output_root` folders produced **bit-exact identical outputs (`max_abs_diff = 0.0`, `mean_abs_diff = 0.0`)**, while `seed 42` vs `seed 43` produced **`max_abs_diff = 168.0`** and **`mean_abs_diff = 2.9309`** (in 8-bit pixel levels `0..255`).

---

## 2. Exact Colab Outputs & Code Explanation

### A. `CELL 03` Output: Hardware & PyTorch Stack
```text
Python  : 3.13.16
Platform: Linux-6.6.159+-x86_64-with-glibc2.39
PyTorch : 2.11.0+cu130
CUDA available: True
SDPA available: True
GPU 0   : NVIDIA A100-SXM4-80GB | total=79.25 GiB | free=78.83 GiB
CELL 03 READY
```

### B. `CELL 04` Output & Follow-Up Check: `RealSRVal_crop128` Structure
```text
=== Directories under StableSR_testsets (max depth 3) ===
  StableSR_testsets (direct images: 0)
  StableSR_testsets/DIV2K_V2_val (direct images: 0)
  StableSR_testsets/DIV2K_V2_val/gt (direct images: 3000)
  StableSR_testsets/DIV2K_V2_val/lq (direct images: 3000)
  StableSR_testsets/DPEDiphoneValSet_crop128 (direct images: 113)
  StableSR_testsets/DrealSRVal_crop128 (direct images: 0)
  StableSR_testsets/DrealSRVal_crop128/test_HR (direct images: 93)
  StableSR_testsets/DrealSRVal_crop128/test_LR (direct images: 93)
  StableSR_testsets/RealSRVal_crop128 (direct images: 0)
  StableSR_testsets/RealSRVal_crop128/test_HR (direct images: 100)
  StableSR_testsets/RealSRVal_crop128/test_LR (direct images: 100)

RealSRVal_crop128 total images: 200
First image : Canon_001_LR4.png | size=(512, 512) (WxH) | mode=RGB
```
And inspecting `test_HR` vs. `test_LR` directly:
```text
test_HR first 3: ['Canon_001_LR4.png', 'Canon_002_LR4.png', 'Canon_003_LR4.png'] size: (512, 512)
test_LR first 3: ['Canon_001_LR4.png', 'Canon_002_LR4.png', 'Canon_003_LR4.png'] size: (128, 128)
```

### C. `CELL 09` Output: Memory Profile with VAE `AttnBlock` SDPA Patch (`2048x2048`)
```text
2026-11-05 16:08:23,946 INFO DoSSR | DoSSR compatibility: using PyTorch scaled_dot_product_attention for vanilla CrossAttention; original projections, heads, scale, and mask semantics are retained.
2026-11-05 16:08:23,946 INFO DoSSR | DoSSR compatibility: using PyTorch scaled_dot_product_attention for vanilla VAE AttnBlock; original norm, q/k/v projections, single-head scale, proj_out, and residual semantics are retained.
2026-11-05 16:12:34,542 INFO DoSSR | DoSSR memory profile | after model device transfer | allocated=6.44 GiB | reserved=6.52 GiB | allocator_cache=0.08 GiB | peak_allocated=6.44 GiB | peak_reserved=6.52 GiB | inactive_split=0.08 GiB | alloc_retries=0 | ooms=0
2026-11-05 16:12:35,328 INFO DoSSR | DoSSR memory profile | before sampler | allocated=6.48 GiB | reserved=10.68 GiB | allocator_cache=4.19 GiB | peak_allocated=9.64 GiB | peak_reserved=10.68 GiB | inactive_split=0.08 GiB | alloc_retries=0 | ooms=0
2026-11-05 16:12:40,605 INFO DoSSR | DoSSR sampler profile | elapsed=5.28s
2026-11-05 16:12:40,606 INFO DoSSR | DoSSR memory profile | after sampler | allocated=6.51 GiB | reserved=34.71 GiB | allocator_cache=28.19 GiB | peak_allocated=20.51 GiB | peak_reserved=34.71 GiB | inactive_split=0.12 GiB | alloc_retries=0 | ooms=0
2026-11-05 16:12:42,952 INFO DoSSR | Execution time: 8.33 seconds
```

| Precision & Patches (`2048x2048` output) | `peak_allocated` | `peak_reserved` | Sampler time | Image time |
|---|---|---|---|---|
| `fp32` (`CrossAttention` SDPA only) | `49.29 GiB` | `60.48 GiB` | `18.17 s` | `21.05 s` |
| `fp16` (`CrossAttention` SDPA only, before VAE patch) | `36.04 GiB` | `52.70 GiB` | `4.61 s` | `7.68 s` |
| **`fp16` (`CrossAttention` + `VAE AttnBlock` SDPA, `5e6a777`)** | **`20.51 GiB`** | **`34.71 GiB`** | **`5.28 s`** | **`8.33 s`** |

### D. `CELL 10` & `CELL 11` Status JSON Outputs (`2048x2048`)
```json
{
  "passed": true,
  "git_commit": "5e6a777e9abde8d3cb4de829d90a42c713ead350",
  "timestamp_utc": "2026-11-05T16:15:17.322012+00:00",
  "seed_spread_min_ratio": 3.0,
  "gate_rule": "mean_same == 0.0 -> require mean_cross > 0.0",
  "diff_42_vs_42_repeat_px": {
    "max_abs_diff": 0.0,
    "mean_abs_diff": 0.0
  },
  "diff_42_vs_43_px": {
    "max_abs_diff": 168.0,
    "mean_abs_diff": 2.9309139251708984
  }
}
```

---

## 3. Quarantine Rule for Old `2048x2048` Output Folders (`README_DO_NOT_USE_AS_BASELINE.txt`)

Any output folder generated prior to the `test_LR` fix used `test_HR` (`512x512`) as input and produced `2048x2048` images. Those folders are valid **only** as hardware/VRAM stress-test benchmarks for `2048x2048` attention scaling, **not** as RealSR super-resolution baseline outputs.

Every such folder (`baseline-smoke-l4-fp16`, `baseline-smoke-vae-sdpa-fp16`, `seed-sanity-repeat42-fp16`, `seed-sanity-seed43-fp16`, `baseline-realsr-5img`) must contain a `README_DO_NOT_USE_AS_BASELINE.txt` file stating:
```text
DO NOT USE AS REALSR BASELINE.
These 2048x2048 outputs were generated when input_dir pointed to the parent
RealSRVal_crop128 folder, which caused rglob('*') to stage 512x512 images from
test_HR instead of 128x128 images from test_LR.
Use the *-lr128-* output folders (generated from RealSRVal_crop128/test_LR,
128x128 input -> 512x512 output) for all baseline and seed-study evaluations.
```

---

## 4. Safeguards & Next Steps Implemented (`Status: NOT TESTED` on `test_LR`)

1. **`scripts/prepare_baseline_smoke.py`:** Now stops with an error if `input_dir` has subdirectories or mixed image sizes, and logs the size (`WxH`) of every staged image.
2. **`src/run_dossr.py`:** Refuses to run if `input_dir` is named `test_HR` or contains a `test_HR` subdirectory; accepts `gt_dir` in YAML configs without passing it to DoSSR; supports `--disable-sdpa` and `--disable-vae-sdpa`.
3. **`scripts/run_seed_study.py` & `notebooks/01_dossr_baseline.ipynb`:** Verify that every staged input image is `128x128` and every generated output image is `512x512`.
4. **Estimated `test_LR` (`128x128 -> 512x512`) performance (`NOT TESTED`):** Rough engineering estimates (~7–9 GiB VRAM, ~1–2 s per image, plausible on smaller GPUs like T4/L4) are **estimates only (`NOT TESTED`)** until `CELL 08`–`CELL 11` are executed on `test_LR` in Colab.
