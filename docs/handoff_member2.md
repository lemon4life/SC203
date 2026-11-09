# Handoff Note for Member 2 (Huỳnh Trần Gia Hân, `25125042`) — Text Labeling (Images 1–50) & Text Evaluation

**Assigned Range:** **Images `1–50` (`#001` to `#050`)** in `data/text_boxes_realsr_m2.csv`  
- Contact sheet 1 (`#001`–`#020`), Contact sheet 2 (`#021`–`#040`), and the **first half of Contact sheet 3 (`#041`–`#050`)**.  
- *(Note: Contact sheet 3 is shared — you label `#041`–`#050`, and Member 3 Quang Minh labels `#051`–`#060`.)*  
**Hardware needed:** Any CPU laptop (Windows or Mac) or Free Google Colab (CPU). **No GPU needed.**  
**Status of scripts:** `scripts/make_text_contact_sheets.py`, `scripts/check_text_boxes.py`, and `scripts/text_eval.py` are created (`NOT TESTED` on full `test_HR`).

---

## 1. Deadlines & No-Git Rule (Assignment 4 — Due Tue 10/11 07:00)

- **Sun `08/11 21:00`:** Finish labeling images `1–50` in `data/text_boxes_realsr_m2.csv`, run `scripts/check_text_boxes.py`, report how many images out of `50` have `readable=yes` text, and send `data/text_boxes_realsr_m2.csv` to Member 1 (Nguyên Anh) or upload it to the shared Google Drive folder `lemmepass/`.
- **Sun `08/11 21:30`:** Member 1 merges `data/text_boxes_realsr_m2.csv` (`1–50`) and `data/text_boxes_realsr_m3.csv` (`51–100`) into `data/text_boxes_realsr.csv` and sends the merged file back to you.
- **Mon `09/11 15:00`:** Run `scripts/text_eval.py` (test on `seed_42` first, then all 10 seeds) on the two 10-seed output folders on Drive and send the output CSVs to Member 1.
- **Mon `09/11 18:00`:** Data freeze. **Mon `09/11 21:00`:** Proofread. **Mon `09/11 22:00`:** Internal submit for Assignment 4.
- **No Git required:** You do **not** need to commit or push to Git. Save your CSV and output files to the shared Google Drive (`lemmepass/`) or send them directly to Member 1 in chat. Do **not** edit `context.md` or `external/`.

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

## 4. What to Do Today (Sunday, Before 21:00: Contact Sheets & Labeling Images 1–50)

### Step 1: Clone the repo and switch to `feat/baseline-runner`
```bash
git clone https://github.com/lemon4life/SC203
cd SC203
git checkout feat/baseline-runner
```

### Step 2: Generate Contact Sheets & Your CSV (`data/text_boxes_realsr_m2.csv`, Images 1–50)
Replace `<PATH_TO_TEST_HR>` with your local or Drive path to `RealSRVal_crop128/test_HR` (note: on Windows you may type `py` instead of `python`):

```bash
python scripts/make_text_contact_sheets.py \
  --hr-dir "<PATH_TO_TEST_HR>" \
  --out-dir "outputs/text_contact_sheets" \
  --csv-out "data/text_boxes_realsr_m2.csv" \
  --range 1-50
```

### Step 3: Check that Step 2 Worked (Success Signal)
1. The terminal prints:
   `DONE | total_images=100 | sheets_written=5 | csv_range=1-50 (50 images) | csv=...`
2. `outputs/text_contact_sheets/` contains `contact_sheet_01_imgs_001_to_020.png` through `contact_sheet_05_imgs_081_to_100.png` (20 `512x512` HR images per sheet, with `25% / 50% / 75%` pixel tick marks at `128, 256, 384` px on every tile border).
3. `data/text_boxes_realsr_m2.csv` exists with **50 rows** (images `#001` through `#050`, corresponding to sheets 1, 2, and the first 10 images `#041`–`#050` of sheet 3). Notice that `readable` is left **blank (`""`)** by default for every row (`blank = not yet labeled`).

---

## 5. Canonical CSV Format, Worked Examples & Text Labeling Rules (MUST FOLLOW)

Member 2 (`data/text_boxes_realsr_m2.csv`, images `1–50`) and Member 3 (`data/text_boxes_realsr_m3.csv`, images `51–100`) must follow the **exact same rules** so Member 1 can merge them into `data/text_boxes_realsr.csv`. Do **not** change column names.

### Exact CSV Header
```csv
image,box_id,x,y,w,h,transcript,readable
```

### Column Definitions
- `image`: Exact file name in `test_HR` (for example `Canon_001_LR4.png`).
- `box_id`: Integer starting at `1` for each image (`1`, `2`, `3`, ...). If an image has **no** readable text at all, keep **one** row with `box_id=0`, leave `x,y,w,h,transcript` empty, and set `readable=no`. If an image has 3 readable text boxes, make 3 rows for that image (`box_id=1`, `box_id=2`, `box_id=3`).
- `x,y,w,h`: Integer pixel box in the **`512x512` `test_HR` image**:
  - `x`: Left pixel column (`0` to `511`).
  - `y`: Top pixel row (`0` to `511`).
  - `w`: Box width in pixels (`> 0`, with `x + w <= 512`).
  - `h`: Box height in pixels (`> 0`, with `y + h <= 512`).
  - *Tip:* Add a small margin (`4` to `8` pixels) around the letters so the crop does not cut off edges.
- `transcript`: The ground-truth text written in `test_HR` (do **not** guess; see Text Rules below).
- `readable`:
  - **Blank (`""`)** when first generated (`not yet labeled`). You **must** fill `yes` or `no` for every row before submitting.
  - `yes` if a human looking at `test_HR` can read the text clearly and unambiguously.
  - `no` if the image has no text, or if the text in `test_HR` is too blurry/tiny for a human to read.

### Worked Label Examples (`EXAMPLE ONLY — NOT REAL DATA`)

> **IMPORTANT:** The 4 rows below are **synthetic worked examples** to show the CSV format. They are **NOT** real data from `test_HR`.
> - **Confirmed real `Canon_001_LR4.png` (`eduroam`) box (from `data/text_boxes_realsr_m2.csv`):** `Canon_001_LR4.png,1,320,38,192,48,eduroam,yes` (and full sign crop for figure: `EDUROAM_CROP = '310,5,202,85'`).

```csv
image,box_id,x,y,w,h,transcript,readable
EXAMPLE_sign_one_word.png,1,120,40,160,55,STOP,yes
EXAMPLE_sign_two_words.png,1,80,100,110,42,CAFE,yes
EXAMPLE_sign_two_words.png,2,205,100,140,42,OPEN,yes
EXAMPLE_nature_no_text.png,0,,,,,no
EXAMPLE_blurry_unreadable_plate.png,0,,,,,no
```
- **Example 1 (`EXAMPLE_sign_one_word.png`):** Has 1 clear word (`STOP`) inside `(x=120, y=40, w=160, h=55)`. Set `box_id=1`, `transcript=STOP`, `readable=yes`.
- **Example 2 (`EXAMPLE_sign_two_words.png`):** Has 2 separate readable words (`CAFE` and `OPEN`). Create 2 rows with the same `image` name and `box_id=1`, `box_id=2`.
- **Example 3 (`EXAMPLE_nature_no_text.png`):** Has no text at all. Keep 1 row with `box_id=0`, empty `x,y,w,h,transcript`, and `readable=no`.
- **Example 4 (`EXAMPLE_blurry_unreadable_plate.png`):** Has tiny background letters that even a human cannot read in `test_HR`. Keep 1 row with `box_id=0`, empty `x,y,w,h,transcript`, and `readable=no` (or if you want to record the unreadable box coordinates, set `box_id=1`, fill `x,y,w,h`, leave `transcript` empty, and set `readable=no`).

### Step 4: Validate Your CSV & Draw Box Previews (`scripts/check_text_boxes.py`)
After you finish labeling images `1–50` in `data/text_boxes_realsr_m2.csv`, run the validator with `--preview-dir`:
```bash
python scripts/check_text_boxes.py \
  --csv "data/text_boxes_realsr_m2.csv" \
  --hr-dir "<PATH_TO_TEST_HR>" \
  --range 1-50 \
  --preview-dir "outputs/previews_m2"
```
- **What it checks:** Exact header, integer coordinates inside `512x512`, `w > 0` and `h > 0`, `readable` is `yes` or `no` (no blanks), non-empty `transcript` when `readable=yes`, no duplicate `(image, box_id)`, all image names exist in `test_HR`, and all 50 images in `1–50` are covered.
- **Visual check:** Open `outputs/previews_m2/<image>_boxes.png` to visually check that your green boxes sit around the right words!
- **Success signal:** Prints:
  `CHECK PASSED | file=.../data/text_boxes_realsr_m2.csv | range=1-50 (50 images) | total_rows=... | images_with_readable_text=... | readable_yes_boxes=... | readable_no_rows=... | previews_saved=...`
  *(You can also still run `python scripts/make_text_contact_sheets.py --check data/text_boxes_realsr_m2.csv` as a quick syntax check.)*

### Text Transcription & Normalization Rules (Used by Member 2, Member 3, and `scripts/text_eval.py`)
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

## 6. Monday Task (Before Mon 09/11 15:00): Run `scripts/text_eval.py` on the 10-Seed Outputs

Once Member 1 sends you the merged `data/text_boxes_realsr.csv` (Sun `21:30`) and Member 3 confirms the OCR tool and crop upscale factor (Sun night):

1. **Install the chosen OCR tool** (for example `rapidocr`):
   ```bash
   pip install rapidocr-onnxruntime
   ```
2. **Real 10-seed output roots on Shared Google Drive** (from `configs/seed_study_realsr_100img_10seed_{default,empty}_prompt_sdpa_off.yaml`):
   - **Default prompt (`"clean, high-resolution, 8k"`):**
     `/content/drive/MyDrive/lemmepass/outputs/seed-study-realsr-100img-10seed-default-prompt-sdpa-off-fp16`
   - **Empty prompt (`""`):**
     `/content/drive/MyDrive/lemmepass/outputs/seed-study-realsr-100img-10seed-empty-prompt-sdpa-off-fp16`
3. **Test on 1 seed (`--seeds 42`) first, then run on all 10 seeds (`42..51`) with HR ceiling (`--hr-dir`) and bicubic floor (`--lr-dir`):**
   Replace `<DEFAULT_10SEED_ROOT>`, `<EMPTY_10SEED_ROOT>`, `<PATH_TO_TEST_HR>`, and `<PATH_TO_TEST_LR>` with your local or Drive paths (and set `--upscale` to the value Member 3 recommends from `ocr_check_hr.py`, e.g. `1`, `2`, or `4`):
   ```bash
   # Step A: Quick 1-seed smoke test on seed_42
   python scripts/text_eval.py \
     --study-root "<DEFAULT_10SEED_ROOT>" \
     --boxes-csv "data/text_boxes_realsr.csv" \
     --ocr-backend rapidocr \
     --rec-only \
     --rotate-vertical \
     --upscale 1 \
     --seeds 42 \
     --output-dir "outputs/text_eval_smoke_seed42"

   # Step B: Full 10-seed evaluation (default vs empty + HR ceiling + bicubic x4 floor)
   python scripts/text_eval.py \
     --study-root "<DEFAULT_10SEED_ROOT>" \
     --compare-study-root "<EMPTY_10SEED_ROOT>" \
     --hr-dir "<PATH_TO_TEST_HR>" \
     --lr-dir "<PATH_TO_TEST_LR>" \
     --boxes-csv "data/text_boxes_realsr.csv" \
     --ocr-backend rapidocr \
     --rec-only \
     --rotate-vertical \
     --upscale 1 \
     --output-dir "outputs/text_eval_10seed"
   ```
   - **Success signal:** Prints `HR CEILING`, `BICUBIC x4 FLOOR`, `TEXT EVAL SUMMARY`, and `PAIRED COMPARISON`, and writes `text_eval_per_box_seed.csv`, `text_eval_summary_by_image.csv`, `text_eval_paired_comparison_by_image.csv`, `text_eval_hr_ceiling_by_image.csv`, and `text_eval_bicubic_floor_by_image.csv` inside `--output-dir`.

---

## 7. What to Send Back to Member 1 (Nguyên Anh)

1. **By Sunday `08/11 21:00`:**
   - Your validated `data/text_boxes_realsr_m2.csv` (upload to shared Drive or send in chat).
   - The `CHECK PASSED` line from `scripts/check_text_boxes.py` showing `images_with_readable_text` and `readable_yes_boxes` in images `1–50`.
2. **By Monday `09/11 15:00`:**
   - The terminal output and generated CSV files (`text_eval_per_box_seed.csv`, `text_eval_summary_by_image.csv`, `text_eval_paired_comparison_by_image.csv`, `text_eval_hr_ceiling_by_image.csv`, `text_eval_bicubic_floor_by_image.csv`) from `scripts/text_eval.py`.
3. **If any command fails:** Send the **exact command** you ran and the **full error traceback** immediately in chat.
