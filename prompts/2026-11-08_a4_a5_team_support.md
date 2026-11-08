# Prompt — 2026-11-08: Group Assignment 4 & 5 Team Support

```text
Context: our deadlines changed. Group Assignment 4 (SOTA's Limitations) is due
Tue 10/11/2026 07:00. Group Assignment 5 (Baseline) is due Tue 17/11/2026
07:00. Internal submit for A4: Mon 09/11 22:00. A4 needs experiments that show
a limitation of DoSSR: does it restore text on signs consistently when only
the seed changes?

Already done by me: 10 seeds x 100 test_LR images x 2 prompt modes (default,
empty) were run, and the eduroam figure (CELL 18) was made. Take the two output
roots from configs/seed_study_realsr_100img_10seed_{default,empty}_prompt_
sdpa_off.yaml (output_root). Do not guess them.

STEP 0: orient first (do not skip)
- My editor shows this tree: data/ (text_boxes_realsr_template.csv), docs/
  (plans/, after_01_dossr_baseline_discovery.md, colab_workflow.md,
  handoff_member2.md, handoff_member3.md, handoff_member4.md,
  lora_feasibility_and_loss_design.md, phase2_plan.md,
  related_methods_ccsr_pnfs_tadisr.md, setup_notes.md, Week01.tex and its
  build files Week01.aux, Week01.fdb_latexmk, Week01.fls, Week01.log),
  external/ (DOSSR, InstructIR), notebooks/ (01_dossr_baseline,
  colab_l4_smoke, colab_runner, kaggle_runner), prompts/README.md, results/.
  The screenshot is cut off below results/. Run git ls-files and list the
  folders yourself, and use the real names for scripts/, src/, configs/.
- Read: prompts/README.md (follow it; save THIS prompt as
  prompts/2026-11-08_a4_a5_team_support.md), the results/ folder (what it
  holds and any README), docs/Week01.tex (preamble and structure), .gitignore.
- Housekeeping: Week01.aux, Week01.fdb_latexmk and Week01.fls are untracked
  LaTeX build files. Add LaTeX build artifacts to .gitignore (*.aux,
  *.fdb_latexmk, *.fls, *.synctex.gz, *.out, *.toc, *.log). Do not git add
  them. Check that data/text_boxes_realsr*.csv are NOT ignored.
- Do NOT edit or delete docs/Week01.tex (it is mine). Audit it against
  docs/after_01_dossr_baseline_discovery.md and list every contradiction
  (the early runs used test_HR at 512 to 2048 by mistake; the correct input is
  test_LR 128 to 512; the memory story and numbers). Report only.
- Check that docs/ke_hoach_nhom_A4_A5.md exists (it is the Vietnamese team
  plan). If it is missing, tell me and continue.
- Also read context.md and the current docs/handoff_member{2,3,4}.md.

New roles and deadlines (times are a proposal):
- Member 1 (me): check the 2,000 outputs, merge the label CSVs, collect student
  IDs, assemble and submit the PDF. Merge CSVs Sun 21:30, submit Mon 22:00.
- Member 2: label images 1-50 in data/text_boxes_realsr_m2.csv, report how many
  have readable text, then run text_eval.py on the 10-seed outputs. Labels and
  count Sun 21:00, scores Mon 15:00.
- Member 3: LoRA work is paused until Tue 11/11. Label images 51-100 in
  data/text_boxes_realsr_m3.csv (sheet 3 is shared: Member 3 takes 51-60,
  Member 2 takes 41-50), check the OCR on the sharp HR crops, and help debug
  text_eval.py. Labels Sun 21:00, OCR choice Sun night, checks Mon 12:00.
- Member 4: LaTeX skeleton, write the sections that already have data, run
  PSNR/SSIM/LPIPS on the 10-seed folders (CPU), one page on CCSR, PNFS and
  TADiSR. Skeleton Sun 21:00, one page Mon 12:00, metrics Mon 15:00.
Everyone: data freeze Mon 18:00, proofread Mon 21:00.
Members use no git: they save to the shared Drive or send files to me.

TASKS, in this order. Stop and report after each numbered group.

1. Unblock the labelers first (they start today)
 a. scripts/check_text_boxes.py: validate a labeler's CSV. Header must be
    image,box_id,x,y,w,h,transcript,readable. Check that coordinates are
    integers inside the 512x512 HR image, w and h are above 0, readable is
    yes or no (no blanks), transcript is not empty when readable is yes, no
    duplicate (image, box_id), image names exist in test_HR, and every image in
    the labeler's range has at least one row. Add --preview-dir that draws each
    box on its HR image and saves a picture, so labelers can see their boxes.
 b. scripts/merge_text_boxes.py: merge the _m2 (images 1-50) and _m3 (51-100)
    files into data/text_boxes_realsr.csv. Fail with a clear error on a missing
    or duplicate image, a blank readable, or an overlap between the two files.
    Print the number of images with readable text.
 c. UPDATE (do not recreate) docs/handoff_member2.md and handoff_member3.md:
    the new tasks, ranges, per-person CSV names, the exact commands, the
    success signal for each command, what to send back, and 3 or 4 worked label
    examples (clearly marked as examples, not real data). Keep the existing
    text rules. Put the eduroam example in only after I send the crop box; until
    then write TODO.

2. Member 3's OCR check
 a. scripts/ocr_check_hr.py: read the labeled CSV, crop each readable box from
    the HR image, run the chosen OCR backend (rapidocr or easyocr via a flag),
    optionally enlarge the crop 1x, 2x or 4x, compare with the transcript using
    the same normalization rules as text_eval.py, and print: boxes read
    exactly right out of the total, and a CSV of each box with the OCR text.
 b. text_eval.py: add the HR ceiling (the same scores on test_HR crops) and a
    bicubic floor (test_LR enlarged x4), and the crop enlarge option, if not
    done yet. Confirm that all arguments match the commands in the handoff
    notes by running --help, and fix the docs where they differ.

3. Checks for me
 a. scripts/check_study_outputs.py --root <output_root>: check 10 seed folders,
    100 PNGs each, all 512x512, and read run_manifest.json to confirm precision
    fp16, sdpa off and the expected prompt_mode. Print a short table and exit
    with an error if anything is wrong. CPU only.

4. Member 4's kit
 a. docs/fact_sheet_for_report.md: every number we can already use, with its
    source (log, CELL or doc) and status VERIFIED or ESTIMATE or PENDING.
    Include: hardware (A100), data (100 LR-HR pairs, 128 to 512), settings, the
    SDPA equivalence checks, speed and memory on test_LR, seed results (same
    seed 0.0, 42 vs 43 on 3 images), prompt facts (default prompt text,
    training uses the empty prompt, the one-image effect), the 2048 stress test
    (80 GiB request, 36.04 to 20.51 GiB), and the early wrong-folder story.
    Mark every number you cannot find in a log or doc as PENDING. Do not invent.
 b. LaTeX skeletons docs/A4_sota_limitations.tex and docs/A5_baseline.tex next
    to Week01.tex (figures in docs/figures/). Reuse the Week01.tex preamble and
    style. Sections from docs/ke_hoach_nhom_A4_A5.md section 9, with TODO
    markers. Our topic-submission template (topic-3.tex) is not in the tree: if
    the course template differs from Week01.tex, ask me. Tell me before you
    create any new folder. Do not commit LaTeX build files.
 c. docs/related_methods_ccsr_pnfs_tadisr.md: add a checklist of what to extract
    from each paper (problem, how it treats stability or text, inputs needed,
    training or not, main limitation as stated by the authors, how ours
    differs) and require a page or section for every fact. Do not fill in any
    facts about the papers.
 d. Confirm the compute_metrics.py command for both 10-seed folders (CPU) and
    write it with the real paths in docs/handoff_member4.md.

5. Setup for everyone
 a. Requirements files per role (text labeling and OCR; metrics and figures).
    First check which requirements files already exist and propose names.
    Include pip commands for Windows and Mac. Mark versions you have not
    tested NOT TESTED. Do not claim they work on Windows.
 b. docs/data_access.md: the Drive folder layout (data, outputs, figures, an
    inbox folder for teammates), who needs view access to what, how to use it
    from a laptop (Drive for Desktop) or free Colab, and the real output roots
    from the configs.
 c. Vietnamese copies of the three handoff notes (docs/handoff_member2.vi.md
    and so on): simple words, same commands. Keep commands, file names and the
    CSV header in English. Mark them "TODO: review by a Vietnamese speaker".
 d. Optional, only if time allows: notebooks/02_member_starter.ipynb (CPU
    friendly, same notebook guidelines as 01_dossr_baseline.ipynb): mounts
    Drive, clones the repo, installs the requirements, one cell per member
    task. Do not touch the four existing notebooks.

6. Update the docs
 a. context.md: the status table (task, owner Member 1-4, status, blocked by),
    the new deadlines, who waits for whom, and the open questions (student IDs,
    number of readable text images, OCR choice, LoRA trial date, LoRA training
    data that does not overlap the evaluation images, LoRA prompt choice).
    Keep the rules.
 b. Add a short note at the top of docs/phase2_plan.md saying it is replaced by
    docs/ke_hoach_nhom_A4_A5.md. Do not rewrite it.
 c. Save a short plan for this work as docs/plans/05_a4_a5_team_support.md.
 d. If results/ has a convention, say where small summary CSVs should go. Large
    outputs stay on Drive.

Rules:
- Do not edit external/. Do not edit docs/Week01.tex. All new scripts for
  members: Python only, pathlib, CLI paths, no /content paths, CPU only, work
  on Windows, Mac and free Colab.
- Mark everything NOT TESTED until I send logs. Run only what you can run here
  (syntax checks, --help). Do not claim a script works.
- Do not invent facts about papers, numbers, names, student IDs or paths. Write
  TODO and ask me.
- Simple English, short sentences.
- Before every commit, show git status. Commit only the files you changed, not
  LaTeX build files. At the end give: the list of new and changed files, the
  commit hash, and the push command. Then list what you still need from me.

Member 2 is Gia Hân, 3 is Quang Minh, 4 is Minh Nhật, 1 is Nguyên Anh
```
