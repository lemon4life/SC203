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

## 4. Verified Results on True `RealSRVal_crop128/test_LR` (`128x128 -> 512x512`, Commit `251d4a5`)

Running `CELL 01` through `CELL 13` of the updated `notebooks/01_dossr_baseline.ipynb` (`251d4a5`) on Google Colab verified the corrected `test_LR` (`128x128 -> 512x512`) pipeline end-to-end:

### A. `CELL 04` (`VERIFIED`): Dataset Pairing & Quarantine Markers
```text
Verified 100 image(s) in /content/drive/MyDrive/lemmepass/data/StableSR_testsets/StableSR_testsets/RealSRVal_crop128/test_LR all have size (128, 128).
Verified 100 image(s) in /content/drive/MyDrive/lemmepass/data/StableSR_testsets/StableSR_testsets/RealSRVal_crop128/test_HR all have size (512, 512).
Verified 1-to-1 filename pairing between test_LR (128x128) and test_HR (512x512) for all 100 images.
First 3 paired filenames: ['Canon_001_LR4.png', 'Canon_002_LR4.png', 'Canon_003_LR4.png']
Wrote /content/drive/MyDrive/lemmepass/outputs/baseline-smoke-vae-sdpa-fp16/README_DO_NOT_USE_AS_BASELINE.txt
Wrote /content/drive/MyDrive/lemmepass/outputs/seed-sanity-repeat42-fp16/README_DO_NOT_USE_AS_BASELINE.txt
Wrote /content/drive/MyDrive/lemmepass/outputs/seed-sanity-seed43-fp16/README_DO_NOT_USE_AS_BASELINE.txt
CELL 04 READY
```

### B. `CELL 08` & `CELL 09` (`VERIFIED`): 1-Image `test_LR` (`128x128 -> 512x512`) SDPA ON vs. SDPA OFF
```text
=== 1-Image test_LR (128x128 -> 512x512) SDPA ON vs. OFF Comparison ===
SDPA ON  : peak_allocated=7.32 GiB | peak_reserved=8.29 GiB | sampler=1.24 s | image=1.75 s
SDPA OFF : peak_allocated=7.32 GiB | peak_reserved=8.28 GiB | sampler=1.42 s | image=1.92 s
Pixel diff (SDPA ON vs SDPA OFF, 8-bit levels 0..255): max_abs_diff=2.0000 | mean_abs_diff=0.0334
CELL 09 READY
```
- **Why VRAM is `7.32 GiB` (`8.29 GiB` reserved) at `512x512`:** At `512x512` output, the largest attention sequence length is $N = 64 \times 64 = 4{,}096$ ($256\times$ smaller in $O(N^2)$ attention matrix area than $N = 65{,}536$ at `2048x2048`). Because `peak_reserved = 8.29 GiB`, `128x128 -> 512x512` runs fit comfortably on standard Colab **T4 (`14.56 GiB`)** or **L4 (`22.5 GiB`)** GPUs (with High-RAM enabled for host checkpoint loading) without requiring an A100.

### C. `CELL 11` (`VERIFIED`): Resume Check & 3-Image `test_LR` Seed Sensitivity (`42` vs `42-repeat` vs `43`)
```text
=== 3-Image Seed Sensitivity Results (8-bit pixel levels 0..255) ===
  Canon_001_LR4_0.png: (42 vs 42-repeat) max=0.0000, mean=0.0000 | (42 vs 43) max=185.0000, mean=5.1717
  Canon_002_LR4_0.png: (42 vs 42-repeat) max=0.0000, mean=0.0000 | (42 vs 43) max=223.0000, mean=11.9893
  Canon_003_LR4_0.png: (42 vs 42-repeat) max=0.0000, mean=0.0000 | (42 vs 43) max=255.0000, mean=9.3100
Aggregate (3 images): (42 vs 42-repeat) max=0.0000, mean=0.0000
Aggregate (3 images): (42 vs 43)        max=255.0000, mean=8.8237
Wrote status file: /content/sc203_status/cell11_seed_sanity_status.json
SEED SANITY GATE PASSED
CELL 11 READY
```

### D. `CELL 12` & `CELL 13` (`VERIFIED`): 5-Image Batch & 5-Seed Study (`seeds 42..46`)
```text
Disk gate check passed (HEAD=251d4a5).
Verified 5 image(s) in /content/realsr_batch_5_lr128 all have size (128, 128).
=== 5-Seed Study (test_LR 128x128 -> 512x512) Output Verification ===
  seed_42: 5 PNGs verified at (512, 512) | precision=fp16
  seed_43: 5 PNGs verified at (512, 512) | precision=fp16
  seed_44: 5 PNGs verified at (512, 512) | precision=fp16
  seed_45: 5 PNGs verified at (512, 512) | precision=fp16
  seed_46: 5 PNGs verified at (512, 512) | precision=fp16
CELL 13 READY
```

### E. Next Step (`Status: NOT TESTED`): Full-Reference & Cross-Seed Metrics (`CELL 14` & `CELL 15`)
- `scripts/compute_metrics.py` (`CELL 14`) pairs each `<stem>_0.png` (`512x512`) in `seed_42`..`seed_46` with `test_HR/<stem>.png` (`512x512`), computes **PSNR**, **SSIM**, and **LPIPS** per image and seed plus across-seed `mean`, `std`, `min`, `max`, `range`, and pairwise cross-seed spread, and writes `metrics_per_seed.csv` and `metrics_summary_by_image.csv`. (`NOT TESTED` until `CELL 14` Colab logs are returned.)
