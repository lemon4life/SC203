# Handoff Note for Member 2 — Text Evaluation Lead

**Branch:** `feat/text-eval`  
**Hardware needed:** Any CPU laptop (Windows or Mac) or Free Google Colab (CPU). **No GPU needed.**  
**Status of scripts:** `scripts/make_text_contact_sheets.py` is created locally (`NOT TESTED` on full `test_HR`).

---

## 1. Team Git & Communication Rules

1. **Work on your own branch:** `feat/text-eval`.
2. **Only edit your own files:**
   - `scripts/make_text_contact_sheets.py`
   - `scripts/text_eval.py` (after you pick the OCR tool)
   - `data/text_boxes_realsr_template.csv`
   - `data/text_boxes_realsr.csv`
   - `docs/handoff_member2.md`
3. **Do NOT edit `context.md` or `external/`.** Report your progress and numbers in the team chat. Member 1 will update `context.md` and merge your branch via Pull Request (PR).

---

## 2. Python and Package Versions That Work

All Member 2 scripts use standard `pathlib` paths and work on Windows, macOS, and Linux (Free Colab CPU) with no `/content` paths hardcoded.

- **Python:** `3.9`, `3.10`, `3.11`, `3.12`, or `3.13`
- **Required package for Task B.5 (`make_text_contact_sheets.py`):**
  - `Pillow >= 10.0.0`
-Install on Windows / Mac / Colab:
  ```bash
  pip install "Pillow>=10.0.0"
  ```

---

## 3. How to Reach the Data on Shared Google Drive

You need the folder `RealSRVal_crop128/test_HR` (100 images, `512x512` PNGs).

- **Option A — Laptop (Windows or Mac):**
  1. Open the team shared Google Drive folder `lemmepass/data/StableSR_testsets/StableSR_testsets/RealSRVal_crop128/test_HR` (or extract it from `StableSR_testsets.zip`).
  2. Save it to a local folder on your computer, for example `data/RealSRVal_crop128/test_HR`.
- **Option B — Free Google Colab (CPU runtime):**
  1. Mount Drive in Colab:
     ```python
     from google.colab import drive
     drive.mount("/content/drive")
     ```
  2. Pass `--hr-dir "/content/drive/MyDrive/lemmepass/data/StableSR_testsets/StableSR_testsets/RealSRVal_crop128/test_HR"` to the script.

---

## 4. What to Do First (Task B.5: Contact Sheets & Text Box CSV)

### Step 1: Clone the repo and create your branch
```bash
git clone https://github.com/lemon4life/SC203
cd SC203
git checkout feat/baseline-runner
git checkout -b feat/text-eval
```

### Step 2: Run `scripts/make_text_contact_sheets.py`
Replace `<PATH_TO_TEST_HR>` with your local or Drive path to `RealSRVal_crop128/test_HR`:

```bash
python scripts/make_text_contact_sheets.py \
  --hr-dir "<PATH_TO_TEST_HR>" \
  --out-dir "outputs/text_contact_sheets" \
  --csv-out "data/text_boxes_realsr.csv"
```

### Step 3: Check that it worked (Success Signal)
1. The terminal prints:
   `DONE | total_images=100 | sheets_written=5 | csv=...`
2. `outputs/text_contact_sheets/` contains `contact_sheet_01_imgs_001_to_020.png` through `contact_sheet_05_imgs_081_to_100.png` (20 `512x512` HR images per sheet, with `25% / 50% / 75%` pixel tick marks on every tile border).
3. `data/text_boxes_realsr.csv` exists with 100 rows and the exact header below. Notice that `readable` is left **blank (`""`)** by default for every image (`blank = not yet labeled`).

---

## 5. Canonical CSV Format and Text Labeling Rules (MUST FOLLOW)

Members 2, 3, and 4 all use `data/text_boxes_realsr.csv`. Do **not** change column names.

### Exact CSV Header
```csv
image,box_id,x,y,w,h,transcript,readable
```

### Column Definitions
- `image`: Exact file name in `test_HR` (for example `Canon_001_LR4.png`).
- `box_id`: Integer starting at `1` for each image (`1`, `2`, `3`, ...). If an image has **no** text at all, keep one row with `box_id=0`, leave `x,y,w,h,transcript` empty, and set `readable=no`. If an image has 3 text boxes, make 3 rows for that image (`box_id=1`, `box_id=2`, `box_id=3`).
- `x,y,w,h`: Integer pixel box in the **`512x512` `test_HR` image**:
  - `x`: Left pixel column (`0` to `511`).
  - `y`: Top pixel row (`0` to `511`).
  - `w`: Box width in pixels (`> 0`, with `x + w <= 512`).
  - `h`: Box height in pixels (`> 0`, with `y + h <= 512`).
  - *Tip:* Add a small margin (`4` to `8` pixels) around the letters so the crop does not cut off edges.
- `transcript`: The ground-truth text written in `test_HR` (see Text Rules below).
- `readable`:
  - **Blank (`""`)** when first generated (`not yet labeled`). You **must** fill `yes` or `no` for every row before submitting.
  - `yes` if a human looking at `test_HR` can read the text clearly and unambiguously.
  - `no` if the image has no text, or if the text in `test_HR` is too blurry/tiny for a human to read.

### Step 4: Verify Your Completed CSV with `--check`
After you label all 100 images in `data/text_boxes_realsr.csv`, run the built-in checker:
```bash
python scripts/make_text_contact_sheets.py --check data/text_boxes_realsr.csv
```
- **If any image still has a blank `readable` column** (or if a `readable=yes` row is missing `box_id >= 1`, `x, y, w, h`, or `transcript`), the script lists every unlabeled/invalid row and fails with exit code `1`.
- **Success signal:** Prints `CSV CHECK PASSED | total_rows=... | unique_images=100 | images_with_readable_text=... | readable_yes_boxes=... | readable_no_rows=... | unlabeled_blank=0`.

### Text Transcription & Normalization Rules (Used by Member 2 and `scripts/text_eval.py`)
1. **One word or one short single-line phrase per box:**
   - Prefer **one word per box** (for example `eduroam`).
   - If words are tightly joined on a single line, you may put the single line in one box, **never multiple lines in one box**.
2. **Case:**
   - Write the exact characters as they appear in `test_HR` (for example `eduroam` or `STOP`).
   - During automatic evaluation, `scripts/text_eval.py` will compare both **case-insensitive (primary)** and **case-sensitive (secondary)** after normalizing with `lower()`.
3. **Spaces:**
   - Strip leading and trailing spaces.
   - Use a single space `" "` between words if a box has more than one word. `scripts/text_eval.py` collapses multiple spaces into a single space (or strips spaces when comparing single-word boxes).
4. **Punctuation and Symbols:**
   - Only include standard ASCII/Latin letters and digits (`A-Z`, `a-z`, `0-9`) in `transcript` unless punctuation is a core part of a number/code.
   - Do **not** include decorative logos or icons (like the Wi-Fi arcs above `eduroam`).
   - `scripts/text_eval.py` strips leading/trailing punctuation and keeps alphanumeric characters (`a-z0-9`) for our primary Character Error Rate (CER) and wrong-character metric.

---

## 6. Next Decisions for Member 2 (Task B.6 & Real-CE x4 Check)

1. **Count readable text boxes in RealSR `test_HR`:**
   - How many images out of 100 have at least one `readable=yes` text box?
   - How many total `readable=yes` boxes exist across all 100 images?
   - Starting coordinate for `Canon_001_LR4.png` (`eduroam`): check around `x=325, y=22, w=185, h=60` (tight around the word `eduroam`) or `x=310, y=5, w=202, h=85` (with the logo arcs) on the `512x512` HR image and confirm the exact `(x, y, w, h)` box.
2. **Check if we need `Real-CE x4`:**
   - If RealSR `test_HR` has too few readable text boxes (for example fewer than 15–20 readable boxes), our mentor asked us to check **`Real-CE x4`** for text evaluation.
3. **Pick 1 OCR tool and run `scripts/text_eval.py` (Task B.6, `NOT TESTED`):**
   - `scripts/text_eval.py` is ready and keeps the OCR choice in one place (`DEFAULT_OCR_BACKEND` / `--ocr-backend {rapidocr,easyocr,tesseract}`):
     - **Option 1 — `rapidocr` (`pip install rapidocr-onnxruntime`):** `~50 MB` total download, pure CPU ONNXRuntime (no PyTorch/CUDA needed), fast (`~20–50 ms/crop` on CPU), strong scene-text recognition (PP-OCRv4).
     - **Option 2 — `easyocr` (`pip install easyocr`):** `~100 MB` model weights + PyTorch (`~800 MB–2 GB`), supports 80+ languages, runs on CPU or GPU.
     - **Option 3 — `tesseract` (`pip install pytesseract` + system `tesseract-ocr` binary):** `~30 MB`, CPU-only, classic document OCR (weaker on curved/noisy scene signs unless tightly cropped).
   - Command to evaluate a 10-seed study (and optionally compare `empty` vs `default` paired on `(image, seed)` with `IMAGE` as the statistical unit):
     ```bash
     python scripts/text_eval.py \
       --study-root "<PATH_TO_DEFAULT_10SEED_STUDY>" \
       --compare-study-root "<PATH_TO_EMPTY_10SEED_STUDY>" \
       --boxes-csv "data/text_boxes_realsr.csv" \
       --ocr-backend rapidocr
     ```

---

## 7. What to Send Back to Member 1

When you finish Step 2 and your labeling pass, send in the team chat:
1. Total count of `test_HR` images with `readable=yes` text and total count of `readable=yes` boxes.
2. The exact `(x, y, w, h)` box you chose for `eduroam` in `Canon_001_LR4.png` (Member 4 needs this for the `eduroam` figure!).
3. Your committed `data/text_boxes_realsr.csv` on branch `feat/text-eval` (open a PR to `feat/baseline-runner` for Member 1 to merge).
4. If the script fails: send the exact command you ran and the full error traceback.
