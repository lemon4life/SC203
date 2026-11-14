# Handoff Note for Member 3 (Hồ Quang Minh, `25125060`) — LoRA Architecture & Consistency Loss Lead

> [!IMPORTANT]
> **LoRA Pilot Update (Saturday 14/11/2026):**
> 1. **Prioritize LoRA 2 vs. LoRA 3 Pilot (Skip LoRA 1 for now!):** Per Dr. Thuận's directive, we only pilot LoRA 2 (text-weighted) and LoRA 3 (text-weighted + cross-noise latent consistency loss). Defer LoRA 1 (baseline loss) until gating condition passes.
> 2. **Loss Formulation & Fair Comparison:** Ensure both variants draw 2 independent noise samples $(\epsilon_1, \epsilon_2)$ per step at the exact same timestep $t$ and empty prompt `""`.
> 3. **CPU Unit Tests:** Confirm tests `(a)`–`(f)` in `src/lora_text_loss.py` pass cleanly.
> 4. **Pair with Member 1:** Provide the training loss wrapper and monitor peak VRAM & training step time during the Colab run.

**Original Scope (A4 History):** Labeling Images `51–100` (`#051` to `#100`) in `data/text_boxes_realsr_m3.csv` (Nikon, 143 readable boxes) and discovering the RapidOCR `--rec-only --rotate-vertical` fix (`ca127fe`).

---

## 1. Deadlines & No-Git Rule (Assignment 4)

- **Sun `08/11 21:00`:** Finish labeling images `51–100` in `data/text_boxes_realsr_m3.csv`, validate with `scripts/check_text_boxes.py`, report how many images out of `51–100` have `readable=yes` text, and send `data/text_boxes_realsr_m3.csv` to Member 1 (Nguyên Anh) or upload to shared Google Drive (`lemmepass/`).
- **Sun `08/11` night (after `21:30` CSV merge):** Run the OCR check on the sharp `test_HR` crops (`1x`, `2x`, `4x` crop upscale) and tell the team which OCR backend (`rapidocr` vs `easyocr`) and upscale factor read the most `test_HR` boxes accurately.
- **Mon `09/11 12:00`:** Run `scripts/text_eval.py` on `seed_42` to verify there are no runtime errors and help Member 2 debug if needed.
- **Tue `11/11`:** Resume LoRA work (`Section 6` below).
- **No Git required:** Save files to shared Google Drive (`lemmepass/`) or send them directly to Member 1 in chat. Do **not** edit `context.md` or `external/`.

---

## 2. Task A (Sunday Before 21:00): Label Images 51–100 in `data/text_boxes_realsr_m3.csv`

### Step 1: Clone the repo and install Pillow
```bash
git clone https://github.com/lemon4life/SC203
cd SC203
git checkout feat/baseline-runner
pip install "Pillow>=10.0.0"
```

### Step 2: Generate Contact Sheets & Your CSV (`data/text_boxes_realsr_m3.csv`, Images 51–100)
Replace `<PATH_TO_TEST_HR>` with your local or Drive path to `RealSRVal_crop128/test_HR` (on Windows, use `py` if `python` is not on PATH):
```bash
python scripts/make_text_contact_sheets.py \
  --hr-dir "<PATH_TO_TEST_HR>" \
  --out-dir "outputs/text_contact_sheets" \
  --csv-out "data/text_boxes_realsr_m3.csv" \
  --range 51-100
```
- **Success signal:**
  1. Prints: `DONE | total_images=100 | sheets_written=5 | csv_range=51-100 (50 images) | csv=...`
  2. `outputs/text_contact_sheets/` has `contact_sheet_01` through `contact_sheet_05` (with `25% / 50% / 75%` pixel ticks at `128, 256, 384` px).
  3. `data/text_boxes_realsr_m3.csv` has **50 rows** for images `#051` through `#100` with `readable` left blank (`""` = not yet labeled).

---

## 3. Canonical CSV Format, Worked Examples & Text Labeling Rules (MUST FOLLOW)

You and Member 2 must use the **exact same CSV format and text rules** so Member 1 can merge `data/text_boxes_realsr_m2.csv` (`1–50`) and `data/text_boxes_realsr_m3.csv` (`51–100`) into `data/text_boxes_realsr.csv`.

### Exact CSV Header
```csv
image,box_id,x,y,w,h,transcript,readable
```

### Column Definitions
- `image`: Exact file name in `test_HR` (for example `Nikon_001_LR4.png`).
- `box_id`: Integer starting at `1` for each image (`1`, `2`, `3`, ...). If an image has **no** readable text at all, keep **one** row with `box_id=0`, leave `x,y,w,h,transcript` empty, and set `readable=no`. If an image has multiple readable text boxes, create one row per box (`box_id=1`, `box_id=2`, ...).
- `x,y,w,h`: Integer pixel box in the **`512x512` `test_HR` image**:
  - `x`: Left pixel column (`0` to `511`).
  - `y`: Top pixel row (`0` to `511`).
  - `w`: Box width in pixels (`> 0`, with `x + w <= 512`).
  - `h`: Box height in pixels (`> 0`, with `y + h <= 512`).
  - *Tip:* Add a `4` to `8` pixel margin around the letters so the crop does not cut off strokes.
- `transcript`: Exact text read from `test_HR` (do **not** guess).
- `readable`: Must be `yes` or `no` for every row (no blanks).

### Worked Label Examples (`EXAMPLE ONLY — NOT REAL DATA`)

> **IMPORTANT:** The 4 rows below are **synthetic worked examples** to illustrate the format. They are **NOT** real data from `test_HR`.
> - **Confirmed real `Canon_001_LR4.png` (`eduroam`) box (from `data/text_boxes_realsr_m2.csv`):** `Canon_001_LR4.png,1,320,38,192,48,eduroam,yes` (and full sign crop for figure: `EDUROAM_CROP = '310,5,202,85'`).

```csv
image,box_id,x,y,w,h,transcript,readable
EXAMPLE_sign_one_word.png,1,120,40,160,55,STOP,yes
EXAMPLE_sign_two_words.png,1,80,100,110,42,CAFE,yes
EXAMPLE_sign_two_words.png,2,205,100,140,42,OPEN,yes
EXAMPLE_nature_no_text.png,0,,,,,no
EXAMPLE_blurry_unreadable_plate.png,0,,,,,no
```

### Text Transcription & Normalization Rules (Shared with Member 2 and `scripts/text_eval.py`)
1. **One word or one short single-line phrase per box:** Prefer **one word per box**. If words are tightly joined on a single line, you may put that single line in one box, **never multiple lines in one box**.
2. **Case:** Write the exact characters as they appear in `test_HR` (`eduroam`, `STOP`, etc.). `scripts/text_eval.py` evaluates both case-insensitive (primary) and case-sensitive (secondary).
3. **Spaces:** Strip leading/trailing spaces and use a single space `" "` between words if a box has more than one word.
4. **Punctuation and Symbols:** Only include standard ASCII/Latin letters and digits (`A-Z`, `a-z`, `0-9`) in `transcript` unless punctuation is a core part of a number/code. Do **not** include decorative logos or icons.

### Step 3: Validate Your CSV & Draw Box Previews (`scripts/check_text_boxes.py`)
After labeling images `51–100` in `data/text_boxes_realsr_m3.csv`, run:
```bash
python scripts/check_text_boxes.py \
  --csv "data/text_boxes_realsr_m3.csv" \
  --hr-dir "<PATH_TO_TEST_HR>" \
  --range 51-100 \
  --preview-dir "outputs/previews_m3"
```
- **Visual check:** Inspect `outputs/previews_m3/<image>_boxes.png` to verify your green boxes fit the words in `test_HR`.
- **Success signal:** Prints:
  `CHECK PASSED | file=.../data/text_boxes_realsr_m3.csv | range=51-100 (50 images) | total_rows=... | images_with_readable_text=... | readable_yes_boxes=... | readable_no_rows=... | previews_saved=...`

---

## 4. Task B (Can Run Right Now!): Check OCR on Sharp `test_HR` Crops (`scripts/ocr_check_hr.py`)

> **Update:** Your `data/text_boxes_realsr_nikon.csv` (`6b4cca6`) is **merged and verified** (`50` Nikon images, `15` images with readable text, `143` `readable=yes` boxes, `0` errors; also saved as `data/text_boxes_realsr_m3.csv`). You do **not** need to wait for Member 2 to run Task B and Task C — run them right now on `data/text_boxes_realsr_m3.csv`!

Why this matters: If the OCR tool cannot even read the sharp ground-truth `test_HR` crops, then its scores on DoSSR outputs are meaningless.
1. Install `rapidocr-onnxruntime` (and optionally `easyocr` to compare):
   ```bash
   pip install rapidocr-onnxruntime
   ```
2. Run `scripts/ocr_check_hr.py` on `data/text_boxes_realsr_m3.csv` (and later on the merged `data/text_boxes_realsr.csv`) across `--upscale 1`, `2`, and `4`:
   ```bash
   python scripts/ocr_check_hr.py \
     --boxes-csv "data/text_boxes_realsr_m3.csv" \
     --hr-dir "<PATH_TO_TEST_HR>" \
     --ocr-backend rapidocr \
     --upscale 1 \
     --output-csv "outputs/ocr_check_hr_rapidocr_x1.csv"

   python scripts/ocr_check_hr.py \
     --boxes-csv "data/text_boxes_realsr_m3.csv" \
     --hr-dir "<PATH_TO_TEST_HR>" \
     --ocr-backend rapidocr \
     --upscale 2 \
     --output-csv "outputs/ocr_check_hr_rapidocr_x2.csv"

   python scripts/ocr_check_hr.py \
     --boxes-csv "data/text_boxes_realsr_m3.csv" \
     --hr-dir "<PATH_TO_TEST_HR>" \
     --ocr-backend rapidocr \
     --upscale 4 \
     --output-csv "outputs/ocr_check_hr_rapidocr_x4.csv"
   ```
- **Success signal:** Each command prints:
  `OCR HR CHECK DONE | backend=rapidocr | upscale=<1|2|4>x | exact_right=<K>/<N> (...) | exact_case_sensitive=... | box_mean_cer=... | csv=...`
- Report in chat: which `--ocr-backend` (`rapidocr` or `easyocr`) and which `--upscale` (`1`, `2`, or `4`) read the highest number of `test_HR` boxes **exactly right** (`exact_right / 143`).

---

## 5. Task C (Can Run Right Now!): Smoke-Test `scripts/text_eval.py` on `seed_42`

All 2,000 10-seed images are already generated on shared Google Drive:
- Default prompt: `/content/drive/MyDrive/lemmepass/outputs/seed-study-realsr-100img-10seed-default-prompt-sdpa-off-fp16`
- Empty prompt: `/content/drive/MyDrive/lemmepass/outputs/seed-study-realsr-100img-10seed-empty-prompt-sdpa-off-fp16`

Run `scripts/text_eval.py` on `seed_42` (`--seeds 42`) using your Nikon CSV (`data/text_boxes_realsr_m3.csv`), including `--hr-dir` (HR ceiling) and `--lr-dir` (bicubic $\times 4$ floor):
```bash
python scripts/text_eval.py \
  --study-root "<DEFAULT_10SEED_ROOT>" \
  --compare-study-root "<EMPTY_10SEED_ROOT>" \
  --hr-dir "<PATH_TO_TEST_HR>" \
  --lr-dir "<PATH_TO_TEST_LR>" \
  --boxes-csv "data/text_boxes_realsr_m3.csv" \
  --ocr-backend rapidocr \
  --rec-only \
  --rotate-vertical \
  --upscale 1 \
  --seeds 42 \
  --output-dir "outputs/text_eval_nikon_seed42"
```
- **Success signal:** Prints `HR CEILING`, `BICUBIC x4 FLOOR`, `TEXT EVAL SUMMARY`, and `PAIRED COMPARISON`, and writes the 5 summary CSVs into `outputs/text_eval_nikon_seed42/`.

**What to Send Back to Member 1 (Nguyên Anh):**
1. **Task A (`data/text_boxes_realsr_nikon.csv`):** **DONE & MERGED (`15` readable images, `143` boxes, `0` errors).**
2. **Task B:** The 3 `OCR HR CHECK DONE` lines (`1x`, `2x`, `4x`), the output CSVs, and your recommended `--ocr-backend` + `--upscale` factor.
3. **Task C:** Confirmation that `scripts/text_eval.py --seeds 42` succeeded (plus its output lines, or full traceback if any error occurred).

---

## 6. LoRA & Text-Region Loss Reference (PAUSED Until Tuesday 11/11/2026)

Read [`docs/lora_feasibility_and_loss_design.md`](file:///Users/lemon4life/Projects/SC203/docs/lora_feasibility_and_loss_design.md) first. Here is the short summary:

1. **Full training entry point exists in `external/DOSSR`:**
   - [`external/DOSSR/scripts/train.py:10-30`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/scripts/train.py#L10-L30) and [`external/DOSSR/configs/train_cldm.yaml:1-47`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/configs/train_cldm.yaml#L1-L47).
2. **Training prompt vs. Inference prompt (`TODO: confirm with user & mentor`):**
   - Upstream training ([`dataset/realesrgan.py:179`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/dataset/realesrgan.py#L179)) always passes `"txt": ""` (empty prompt).
   - Upstream inference ([`model/shift_sampler.py:322`](file:///Users/lemon4life/Projects/SC203/external/DOSSR/model/shift_sampler.py#L322)) passes `"clean, high-resolution, 8k"`.
   - **Proposal for LoRA `(A, B, C)` (`TODO: decision for Member 1 and the mentor`):** Train `A`, `B`, and `C` with the **empty prompt `""`** (as DoSSR training does) and evaluate them with the same empty-prompt mode.
3. **Where the loss and clean-latent prediction $\hat{x}_0$ live (and why `A` and `B` also draw 2 noises):**
   - [`ControlLDM.shared_step` (`model/cldm.py:346-356`)](file:///Users/lemon4life/Projects/SC203/external/DOSSR/model/cldm.py#L346-L356) calls `pred_x0 = self.predict_x0_out_mix(x, preprocess_x, c, t, **kwargs)` and computes `loss_preprocess_diff = (x - pred_x0).abs().mean()`.
   - [`LatentDiffusion.predict_x0_out_mix` (`ldm/models/diffusion/ddpm.py:909-917`)](file:///Users/lemon4life/Projects/SC203/external/DOSSR/ldm/models/diffusion/ddpm.py#L909-L917) already takes an optional `noise=` tensor!
   - To draw two independent noise samples $\epsilon_1, \epsilon_2 \sim \mathcal{N}(0, I)$ for the same $(x, \text{preprocess\_x}, c, t)$, we call `predict_x0_out_mix` twice (`noise=eps_1` and `noise=eps_2`).
   - **Fairness rule:** Versions `A` and `B` **also** draw two noise samples ($\epsilon_1, \epsilon_2$) and average the normal loss over both predictions, so `C` does not receive extra signal.
   - **Phase 1 target:** For the next report, we aim at a **short trial run of version `C` on Colab Pro with VRAM logged** (`ESTIMATE — NOT TESTED`).
4. **Two critical code traps we found in `external/DOSSR`:**
   - **Trap 1 (`batch_transform.py:98-134`):** `RealESRGANBatchTransform._dequeue_and_enqueue` shuffles a 256-image GPU queue (`torch.randperm`) without shuffling extra batch keys like a text mask. Our LoRA data loader must **not** call `_dequeue_and_enqueue` unless the mask is queued with identical indices (or better: use a direct paired dataset without the queue).
   - **Trap 2 (`model/cldm.py:27-35`):** `ControlledUnetModel.forward` wraps `self.input_blocks` and `self.middle_block` in `with torch.no_grad():`. Put LoRA adapters in `self.control_model` (ControlNet) and `diffusion_model.output_blocks` (or override `ControlledUnetModel.forward` in our wrapper).

---

## 4. How the Shared Text-Box CSV Connects to Your Mask

Member 2 produces `data/text_boxes_realsr.csv` with header:
```csv
image,box_id,x,y,w,h,transcript,readable
```
- Coordinates `(x, y, w, h)` are in `512x512` HR pixel space.
- For each image, union all boxes with `readable == yes` into a binary mask $M_{\text{HR}} \in \{0, 1\}^{1 \times 512 \times 512}$.
- Downsample $M_{\text{HR}}$ by $8\times$ to latent size $(1, 64, 64)$ using `F.max_pool2d(M_HR, kernel_size=8, stride=8)`.

---

## 5. What to Do First & What to Send Back to Member 1

1. Clone the repo and create your branch:
   ```bash
   git clone https://github.com/lemon4life/SC203
   cd SC203
   git checkout feat/baseline-runner
   git checkout -b feat/lora-loss
   ```
2. Read [`docs/lora_feasibility_and_loss_design.md`](file:///Users/lemon4life/Projects/SC203/docs/lora_feasibility_and_loss_design.md).
3. Send back to Member 1 in chat:
   - Your confirmation of the loss weights ($\lambda_{\text{rec}}$ for LoRA `B` and `C`, $\lambda_{\text{cons}}$ for LoRA `C`).
   - Your choice of LoRA rank (`r = 4` vs `r = 8`) and target modules (`ControlNet` only vs `ControlNet + UNet output_blocks`).
   - Which dataset we will use for the trial LoRA run on Colab Pro (so we never train on test images).
