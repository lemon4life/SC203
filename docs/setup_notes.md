# Setup notes: DoSSR and InstructIR

These notes reflect the checked-out repositories and documentation at the recorded commits. No packages were installed and no additional model weights were downloaded during this step.

## Source revisions

| Repository | Commit |
| --- | --- |
| [DoSSR](https://github.com/AMD-AIG-AIMA/DOSSR) | `687860c85b3e31dc44b691950c129d3bbc181361` |
| [InstructIR](https://github.com/mv-lab/InstructIR) | `5959746f21d2eb17c27aec5818536cf24f289138` |

## DoSSR

### Runtime and dependencies

- **Python:** README creates a Conda environment with Python 3.10.
- **PyTorch:** `requirements.txt` does not pin or list `torch`; a specific PyTorch version is not stated in the README or requirements file.
- **CUDA:** no CUDA toolkit/runtime version is specified. The README inference examples use `--device cuda`; the checked-in inference script also supports CPU and MPS and falls back to CPU when CUDA is unavailable.
- **Main requirements:** `xformers==0.0.16`, `pytorch_lightning==1.4.2`, `einops`, `open-clip-torch==2.24.0`, `omegaconf`, `torchmetrics==0.6.0`, `triton==2.0.0`, `opencv-python-headless`, `scipy`, `matplotlib`, `lpips`, `gradio`, `chardet`, `transformers`, and `facexlib`.

### Pretrained weights

- The DoSSR inference checkpoint is either `dossr_default.ckpt` (multi-step) or `dossr_onestep.ckpt` (one-step), linked from the [Google Drive folder in the README](https://drive.google.com/drive/folders/1BTGRbAqXDSNWjyUnTLgDfmjd2xyu8GvP?usp=sharing). The README does not give file sizes.
- The README explains that the multi-step model uses Real-ESRNet preprocessing and the one-step model uses Real-ESRGAN preprocessing. Its training setup additionally names `RealESRNet_x4plus.pth` from [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN), and Stable Diffusion 2.1 Base weights from [Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-2-1-base). Those are described for constructing initial training weights, not as separate inference CLI arguments.
- No checkpoint or weight file is tracked in the DoSSR clone. Weight sizes are not stated in its README.

### Inference

The README shows `python inference.py ...`; in this checkout the entry point is `scripts/inference.py`. From `external/DOSSR`, run (fill in paths):

```bash
python scripts/inference.py \
  --input /path/to/input_images \
  --config configs/model/cldm_v21.yaml \
  --ckpt /path/to/dossr_default.ckpt \
  --steps 5 --sr_scale 4 --color_fix_type wavelet \
  --output /path/to/output --device cuda
```

For one-step inference, use `dossr_onestep.ckpt` and `--steps 1`.

## InstructIR

### Runtime and dependencies

- **Python:** README says Python >=3.10 for its Gradio demo.
- **PyTorch:** no version is specified. `requirements_gradio.txt` has a commented `# torch` line and does not pin a PyTorch version.
- **CUDA:** no CUDA version is specified. The checked-in `gradio_demo.py` sets `device = torch.device("cuda")`, so this demo expects CUDA despite the README not documenting a CUDA requirement.
- **Main requirements:** `requirements_gradio.txt` lists Pillow, sentence-transformers, gradio, and gradio_imageslider; Python >=3.10 and torch appear only as comments. The README also offers `demo.ipynb`, but gives no separate environment file or full inference dependency list.
- **Other model fetched by code:** `configs/eval5d.yml` selects `TaylorAI/bge-micro-v2`, which `text/models.py` loads through Hugging Face Transformers at runtime. Its size is not stated in the README.

### Pretrained weights

- The clone already contains `models/im_instructir-7d.pt` (63,627,895 bytes; about 60.7 MiB) and `models/lm_instructir-7d.pt` (403,275 bytes; about 0.38 MiB). These are tracked in the upstream Git repository; the README does not provide a separate download URL or state their sizes.
- `gradio_demo.py` loads both files by these local paths. The README does not document a separate checkpoint-download step.

### Inference

The README points to `demo.ipynb` and the local Gradio demo but does not provide a shell command. From `external/InstructIR`, the checked-in app can be launched with:

```bash
pip install -r requirements_gradio.txt
python gradio_demo.py
```

The script starts a Gradio interface that accepts an image and a text instruction. It loads the two local `.pt` files and the Hugging Face text encoder. This is the repository's demo path; it is not presented as a command-line batch inference tool.

## Dependency compatibility assessment

There is no definitive named-version conflict stated by the two repositories: both target Python 3.10 or newer, and neither declares a PyTorch version. However, the environment is not yet demonstrated to be jointly installable. DoSSR pins older `xformers==0.0.16` and `triton==2.0.0`, while InstructIR leaves PyTorch and its Gradio dependency versions unpinned. Since those components can impose platform and PyTorch compatibility constraints, a concrete conflict cannot be confirmed or ruled out from the supplied dependency files alone. DoSSR's README targets a CUDA inference setup; InstructIR's demo hardcodes CUDA. Resolve a single compatible PyTorch/CUDA stack before installing anything.

## Apple Silicon / no-CUDA assessment

This is a source inspection, not a tested run. The machine context for this project is Apple Silicon macOS without an NVIDIA GPU/CUDA. PyTorch MPS availability depends on its installed build and macOS; the code itself offers an MPS path for DoSSR.

### DoSSR findings

- **An explicit CPU/MPS option exists:** `scripts/inference.py:119` defines `--device` choices `cpu`, `cuda`, and `mps`, defaulting to `cuda`. `scripts/inference.py:123-145` disables xformers on CPU/MPS, checks MPS support, and falls back to CPU if MPS is unavailable. The model is moved to the selected device at `scripts/inference.py:155`. For this Mac, launch with `--device mps` (or `--device cpu`).
- **xformers is optional in the code path, but listed as a requirement:** `requirements.txt:1` pins `xformers==0.0.16`. `ldm/xformers_state.py:1-7` catches import failure and records xformers unavailable; `scripts/inference.py:131-132` explicitly disables it for CPU/MPS. The regular attention implementation is selected when xformers is unavailable (`ldm/modules/attention.py:247-258`). The actual xformers attention call is at `ldm/modules/attention.py:234`. Thus xformers is a CUDA-oriented optional speedup for this inference path, not an unconditional requirement to execute inference.
- **Triton is listed but no project Python import/use was found:** `requirements.txt:7` pins `triton==2.0.0`; code search found no Triton API calls. It is still a likely platform/install obstacle on macOS, but whether it can be omitted without affecting a particular setup is an inference from the repository source, not a tested install result.
- **Hard-coded CUDA references:** `ldm/modules/encoders/modules.py:45,60,95,200` has CUDA defaults on encoder/helper constructors, but the configured DoSSR inference path moves the model and inputs through the selected device. `ldm/models/diffusion/ddim.py:17-21` and `ldm/models/diffusion/plms.py:19-23` forcibly move sampler buffers to CUDA; these are potential blockers if those legacy samplers are used. The README-selected inference uses `model/shift_sampler.py`, whose timestep extraction includes an MPS float32 fallback (`model/shift_sampler.py:75-80`). `ldm/modules/diffusionmodules/util.py:159` uses `torch.cuda.amp.autocast` in a custom backward/checkpoint path; whether that path is reached during inference is not established here.
- **No `.cuda()` method calls were found** in Python source. CUDA mentions also include conditional cache/device code, not all of which means the selected inference path requires CUDA.

**Conclusion for DoSSR:** source supports a CPU route and an intended MPS route. MPS is plausible, with xformers disabled; however, I cannot guarantee the entire checkpoint/model path is MPS-clean because there are CUDA-forcing legacy samplers and CUDA autocast code elsewhere in the repository. CPU is the safer compatibility fallback, likely much slower. No inference was run.

### InstructIR findings

- **The checked-in Gradio demo hard-codes CUDA:** `gradio_demo.py:38` creates `torch.device("cuda")`, then sends the image and both networks to that device (`:41`, `:49`, `:67`, `:70`). There is no CLI device flag or CPU/MPS selection in this demo. On this Mac, as written, it will fail when trying to use CUDA.
- **CUDA utility calls exist:** `utils.py:16-18` calls `torch.cuda.manual_seed`, `manual_seed_all`, and sets `torch.backends.cudnn.benchmark`; the demo imports are separate and do not call this utility in the inspected file. These calls are not evidence by themselves that the model needs CUDA, but they would need review if that helper is used on CPU/MPS.
- **No `.cuda()` method calls or bitsandbytes/custom CUDA kernel dependencies were found.** `requirements_gradio.txt` lists Pillow, sentence-transformers, Gradio, and gradio_imageslider; it does not list xformers, Triton, or bitsandbytes.
- **Underlying model operations appear device-generic:** the model and tensors use `.to(device)` when called, but the application hard-codes the device. An adaptation to `mps` or `cpu` looks plausible from this code, but is untested and would require changing the app's device selection and checking the text encoder's placement as well.

**Conclusion for InstructIR:** the repository's Gradio demo does not run on this Mac unchanged because it explicitly selects CUDA. A CPU/MPS port appears feasible at the application level, but is not currently exposed or verified.

### Runtime estimate for a ~1.7B-parameter model

This estimate is general; the two checked-in models should not be assumed to have 1.7B parameters. At 16-bit weights, 1.7B parameters require about 3.4 GB decimal (roughly 3.2 GiB) just for weights; at 32-bit, about 6.8 GB (roughly 6.3 GiB). Inference activations, attention buffers, framework state, and input/output memory add overhead. **My estimate:** CPU execution may fit on a Mac with sufficient unified memory but is likely too slow for iterative research runs; MPS is more plausible for usable single-image experiments if the model and operators fit and the implementation supports them. These speed and fit judgments are guesses without the exact checkpoint, model dtype, input sizes, Mac memory, or a benchmark. For DoSSR's diffusion sampling, multiple denoising steps further add latency. No timing or memory test was performed.

### Confidence

- **Directly verified in source:** the DoSSR CLI exposes CPU/MPS and disables xformers off CUDA; InstructIR's demo selects CUDA unconditionally; no `.cuda()` calls were found. DoSSR requirements include `xformers==0.0.16` and `triton==2.0.0`; InstructIR requirements list neither package.
- **Not established by source inspection:** that full DoSSR inference succeeds on MPS, that an edited InstructIR app succeeds on MPS/CPU, and the actual speed or memory use on this Mac. The ~1.7B run-time estimate is a rough engineering estimate, not a benchmark.

## Files inspected

- DoSSR: `README.md`, `requirements.txt`, `scripts/inference.py`, `utils/realesrgan/realesrganer.py`
- InstructIR: `README.md`, `requirements_gradio.txt`, `gradio_demo.py`, `configs/eval5d.yml`, `text/models.py`

## DoSSR implementation details and licensing

### Seed control and random noise

- `scripts/inference.py:118` accepts `--seed` (default `42`), and `scripts/inference.py:149` calls `pl.seed_everything(args.seed)`. Inference accepts a seed and explicitly seeds random generators through PyTorch Lightning.
- The README CLI does not pass `--seed`; the option is available in the script. In the invoked `SpacedSampler.shift_sample_3_order` path, inference passes its preprocessed input as `x_T` (`scripts/inference.py:75-78`), but the sampler encodes that image and adds initial latent Gaussian noise at `model/shift_sampler.py:345-347` with `torch.randn_like(x0_hat)`.
- The sampler also draws per-step noise at `model/shift_sampler.py:363` with `torch.randn_like(img)`, used in the subsequent update at lines 364-377. Both draws are covered by the inference seeding call. These line references are code facts; no noise injection or code change was made.

### Training code and expected data

- Training entry point: `scripts/train.py:10-29`, invoked by the README as `python scripts/train.py --config configs/train_cldm.yaml` (`README.md:93-95`). It loads the config, seeds training, builds the Lightning data module and model, and calls `trainer.fit`.
- Main training config: `configs/train_cldm.yaml`. It points to `configs/dataset/general_deg_realesrgan_train.yaml` (:5) and `configs/model/cldm_v21.yaml` (:11). The default config uses DDP with GPU IDs `[1, 5, 6, 7]` (`configs/train_cldm.yaml:19-23`); this is the repository's example setup, not a Mac-ready configuration.
- Data format: `scripts/make_file_list.py:15-33` scans for `.jpg`, `.png`, and `.jpeg` images and writes newline-delimited image paths to `train.list` and `val.list`. `configs/dataset/general_deg_realesrgan_train.yaml:1-7` configures `RealESRGANDataset` to read `train_set/train.list` and create 512-pixel crops. The dataset opens each path as an RGB image (`dataset/realesrgan.py:87-108`), then synthesizes low-quality examples via blur/noise/degradation transforms. Each training item contains `hq`, degradation kernels, and `"txt": ""` (`dataset/realesrgan.py:170-180`); it does not require a parallel caption or prompt file. Training was not run.

### Text prompt behavior and conditioning path

- The active `scripts/inference.py` call at lines 75-78 does not expose or pass a text prompt argument. The invoked sampler method `shift_sample_3_order` defaults to positive prompt `"clean, high-resolution, 8k"` and negative prompt `""` (`model/shift_sampler.py:316-327`). It uses those defaults to create embeddings at lines 336-343. Inference uses a fixed positive prompt and empty negative prompt, not user-supplied text.
- `LatentDiffusion.get_learned_conditioning` in `ldm/models/diffusion/ddpm.py:675-682` delegates text strings to the configured conditioning model. `configs/model/cldm_v21.yaml:84-88` selects a frozen OpenCLIP text encoder.
- In `model/shift_sampler.py:336-343`, positive and negative prompt embeddings are placed in `c_crossattn`. `ControlLDM.apply_model` concatenates that context at `model/cldm.py:376-380`, passes it as `context=cond_txt` into ControlNet at lines 385-388, then passes the same text context to the diffusion model/UNet at line 390 (or directly to the UNet if control is absent, lines 382-384). These source locations confirm the conditioning path.

### Licenses

- DoSSR repository: its root `LICENSE` declares MIT License (copyright AMD-AGI, 2025).
- InstructIR repository: its root `LICENSE` declares MIT License (copyright Marcos V. Conde, 2024).
- Weights: DoSSR's README links its checkpoints from Google Drive and identifies upstream Stable Diffusion / Real-ESRGAN weights, but I found no separate license statement for the DoSSR checkpoints in the inspected README. InstructIR's `.pt` weights are tracked in the repository, but its README does not specify a separate weights license. The repositories' MIT files are verified; whether those files grant the same terms for each associated weight artifact is not stated, so I cannot infer that they do.

### Verification versus inference

The seed argument and seeding call, the two random-noise lines, training entry point/config/data format, fixed prompt defaults, OpenCLIP embedding and UNet/ControlNet routing, and repository license declarations above are directly verified in the checked-out files at the cited paths and lines. No training or inference was run. The docs do not state a separate license for weight artifacts, so their licensing beyond the repository declarations is unknown from this inspection.

## Phase 1 baseline assets (Colab only)

The approved Phase 1 scope is `dossr_default.ckpt`, `RealESRNet_x4plus.pth`, and the StableSR test-set archive. Stable Diffusion 2.1-base and RealLR200 are explicitly deferred. **RealLR200 is not in the StableSR bundle**; SeeSR publishes it separately in its [Google Drive test-data folder](https://drive.google.com/drive/folders/1L2VsQYQRKhWJxe6yWZU9FgBWSgBCk6mz?usp=drive_link). Do not fetch that folder for Phase 1.

### Asset manifest

Published sizes are included where available. Actual byte sizes, SHA-256 values, and image counts below come from the successful Colab verification log dated 2026-10-28; assets remain in Drive and were not downloaded to this Mac.

| Asset | Source | Published size | Colab path | Verified size / SHA-256 |
| --- | --- | ---: | --- | --- |
| DoSSR multi-step checkpoint `dossr_default.ckpt` | [DoSSR Google Drive folder](https://drive.google.com/drive/folders/1BTGRbAqXDSNWjyUnTLgDfmjd2xyu8GvP?usp=sharing) | Not published for the individual file | `/content/drive/MyDrive/lemmepass/weights/dossr_default.ckpt` | 6,871,491,886 bytes (6,553.17 MiB); SHA-256 `442cbfabb7597144ac42ba6a5c126647e983078bdbc9997a9261c39b9cb18cf8` |
| `RealESRNet_x4plus.pth` | [Official Real-ESRGAN v0.1.1 release](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth) | About 67 MB | `/content/drive/MyDrive/lemmepass/weights/RealESRNet_x4plus.pth` | 67,040,989 bytes (63.94 MiB); SHA-256 `a820b9bde89a874d7599d545567308ce6c128fc8754a53208eda016d40aa81df` |
| StableSR test bundle `StableSR_testsets.zip` | [StableSR-TestSets on Hugging Face](https://huggingface.co/datasets/Iceclear/StableSR-TestSets) | 3.06 GB for the complete archive | `/content/drive/MyDrive/lemmepass/data/StableSR_testsets.zip` | 3,055,396,861 bytes (2,913.85 MiB); SHA-256 `32a6344e2a0d394ffcaa7a9245df14105c590dab3ef0d3540e940add1ef9b803` |
| RealSR Val image folder | In the StableSR archive | Per-folder size not published | `/content/drive/MyDrive/lemmepass/data/StableSR_testsets/StableSR_testsets/RealSRVal_crop128` | 200 images |
| DRealSR Val image folder | In the StableSR archive | Per-folder size not published | `/content/drive/MyDrive/lemmepass/data/StableSR_testsets/StableSR_testsets/DrealSRVal_crop128` | 186 images |

The same run logged `VERIFICATION PASSED` and successfully loaded `dossr_default.ckpt` using `torch.load(map_location='cpu')`. This verifies file integrity checks, image presence, and checkpoint deserialization; it does not verify GPU model loading or inference.

### Colab download and verification steps

These commands run in Colab only. They do not download Stable Diffusion 2.1-base or RealLR200. The repo changes may not yet be pushed, so upload `scripts/download_assets.sh` and `scripts/verify_assets.py` from this working tree into the Colab runtime using the file upload cell below.

1. Mount Drive:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. Create the destination and put **only** `dossr_default.ckpt` there. The source is the DoSSR Drive folder linked above. In its Drive UI, select `dossr_default.ckpt` (not `dossr_onestep.ckpt`), choose **Make a copy**, and move that copy into `MyDrive/lemmepass/weights/`. This avoids downloading an unapproved checkpoint. If direct `gdown --id FILE_ID -O /content/drive/MyDrive/lemmepass/weights/dossr_default.ckpt` fails with a quota error, stop retrying; make the copy in Drive as described, then continue. A copy already in the mounted destination is used as-is by the downloader.

   ```python
   from pathlib import Path
   Path('/content/drive/MyDrive/lemmepass/weights').mkdir(parents=True, exist_ok=True)
   Path('/content/drive/MyDrive/lemmepass/data').mkdir(parents=True, exist_ok=True)
   ```

3. Upload the two scripts from this repo into the Colab runtime (choose both files in the picker):

   ```python
   from google.colab import files
   uploaded = files.upload()
   ```

   Upload `scripts/download_assets.sh` and `scripts/verify_assets.py`; Colab places them in the current directory (`/content` in a fresh runtime).

4. Download RealESRNet and the 3.06 GB StableSR archive, then extract it. The explicit flag enables the already-approved large archive download:

   ```bash
   !bash /content/download_assets.sh --target-dir /content/drive/MyDrive/lemmepass --allow-large-assets
   ```

   The script downloads only the approved RealESRNet and StableSR assets. It checks for the manually copied DoSSR checkpoint and prints its expected location if missing. It does not fetch SD 2.1-base or RealLR200. Ensure the Drive has enough free space for the archive and extracted images.

5. Verify files, image counts, checkpoint loading, byte sizes, and SHA-256 hashes:

   ```bash
   !python /content/verify_assets.py --assets-dir /content/drive/MyDrive/lemmepass
   ```

   The verifier discovers folders named RealSR and DRealSR under the extraction directory. If the archive uses different folder names, locate its image folders with:

   ```bash
   !find /content/drive/MyDrive/lemmepass/data/StableSR_testsets -type d
   ```

   Then pass each actual path explicitly, for example:

   ```bash
   !python /content/verify_assets.py \
     --assets-dir /content/drive/MyDrive/lemmepass \
     --image-dir 'RealSR=/actual/path/to/RealSR' \
     --image-dir 'DRealSR=/actual/path/to/DRealSR'
   ```

   Copy the verifier's `size_bytes`, full paths, SHA-256 hashes, image counts, and final `VERIFICATION PASSED` line back into the manifest above. A passing run requires all three asset files, at least one image in each detected/provided image folder, and successful `torch.load(..., map_location='cpu')` of the DoSSR checkpoint with top-level keys printed. The verifier never reads or prints tokens.

### RealSR / DRealSR / RealLR200 source note

The [StableSR dataset card](https://huggingface.co/datasets/Iceclear/StableSR-TestSets) lists RealSR Val and DRealSR Val, and gives 3.06 GB as the total archive size; the publisher does not list per-folder byte sizes. It does not include RealLR200. SeeSR's project documentation identifies its separate [RealLR200 Drive folder](https://drive.google.com/drive/folders/1L2VsQYQRKhWJxe6yWZU9FgBWSgBCk6mz?usp=drive_link). RealLR200 remains deferred and must not be downloaded until separately approved.

## Colab workflow implementation notes

- The repository had no `SC203_Phase1_RealSR.ipynb`; `notebooks/` contained only `.gitkeep`. The existing asset downloader defaults to `/content/drive/MyDrive/lemmepass` and the verifier checks the DoSSR checkpoint, RealESRNet weights, StableSR archive, image counts, sizes, hashes, and CPU checkpoint loading.
- `notebooks/colab_runner.ipynb` is a thin template with Drive mount, plain-HTTPS branch checkout/pull, runtime report, setup-script invocation, and a tee-based script/log cell. Outputs are cleared in the committed notebook.
- `scripts/colab_setup.py` reads the DoSSR requirements file, excludes `xformers` and `triton`, runs `pip check` and a constrained pip dry run first, and stops on conflicts or any plan to change installed `torch`/`torchvision`. If the ignored external clone is absent in Colab, it clones upstream source only; it does not download weights.
- `src/env_info.py` provides shared timestamped logging and a start-of-run platform/device/PyTorch/seed/arguments line. It redacts argument names that look like credentials. The setup script and asset verifier use this helper; runtime logs default to `/content/drive/MyDrive/lemmepass/logs`.
- The setup script and notebook were initially untested on Colab. The later setup log below verifies `colab_setup.py`; the full runner notebook, DoSSR model load, and inference remain **UNTESTED on Colab**. No package installation or asset download was performed locally.
- Kernel/team decision: use Colab's managed Python kernel rather than creating a venv/Conda environment inside the runtime; the setup cell now invokes scripts with `sys.executable` so pip targets that kernel. For four teammates, use separate personal runtimes, shared Drive access for the single asset copy, and GitHub branches/commits as the code and result provenance source. This is workflow guidance, not a tested Colab run.
- **Colab dependency setup verified (2026-10-28):** after repairing the earlier missing `jedi` dependency, `colab_setup.py` completed in a CUDA runtime with Python 3.13.16, torch `2.11.0+cu130`, and torchvision `0.26.0+cu130`. `pip check` and the constrained dry run passed; setup installed DoSSR's filtered requirements and confirmed torch/torchvision unchanged. `xformers` and DoSSR's pinned `triton==2.0.0` were excluded. The log shows Triton `3.6.0` already present as a dependency of the existing torch installation. This verifies environment setup only; no DoSSR import, model load, or inference was run.

## Vanilla DoSSR RealSR baseline runner

- `scripts/run_baseline.sh` delegates to `src/run_dossr.py`; `configs/baseline_realsr.yaml` holds the Colab asset, code, output, and log paths plus seed/device and sampling settings. The defaults select the verified RealSR Val folder, `dossr_default.ckpt`, 5 steps, CUDA, x4 scale, and wavelet color correction. Results go to `/content/outputs/baseline/seed_42` by default; changing `output_root` in YAML can place them on Drive.
- Direct source inspection of `external/DOSSR/scripts/inference.py:91-119` verified the actual flags `--ckpt`, `--config`, `--input`, `--steps`, `--sr_scale`, `--color_fix_type`, `--output`, `--skip_if_exist`, `--seed`, and `--device`. The runner uses these exact flags. It does not invent separate RealESRNet-weight or solver/t1 flags.
- The checked-in inference process calls `shift_sample_3_order` (`scripts/inference.py:73-77`), which is the third-order solver. The sampler uses `trunc_12` (`model/shift_sampler.py:117-123`), and `ldm/modules/diffusionmodules/util.py:32-38` sets its truncation point to `floor(T/2)`. These are fixed upstream defaults; the YAML records `solver_order: 3` and `t1_fraction: 0.5` as validated expectations, not controls passed on the CLI.
- The multi-step checkpoint's preprocessing model is RealESRNet, as stated in the DoSSR README and declared by `configs/model/cldm_v21.yaml:90-98`. `inference.py` loads the model state from the checkpoint with `strict=True`; it has no standalone RealESRNet weight CLI option. The separate RealESRNet file is therefore not passed by this inference wrapper.
- Device source inspection verified DoSSR accepts `cpu`, `cuda`, or `mps` (`inference.py:119`) and disables xformers on CPU/MPS; it is not CUDA-only. The new wrapper checks the requested device is available and fails rather than silently allowing upstream's CUDA-to-CPU fallback. This source finding does not establish practical speed or successful MPS execution.
- The runner is designed to skip already-produced images through upstream `--skip_if_exist`, records a run manifest, rejects mismatched existing output, and only replaces a seed output folder when explicitly called with `--overwrite`. No inference was run locally. Python 3.13 + the logged Colab torch build has only been verified for dependency setup; the baseline runner, model load, and inference remain **UNTESTED** until a Colab log is supplied.
- **Colab import-path failure and wrapper fix (2026-10-28):** the first Colab attempt reached the wrapper but upstream inference failed at `from ldm.xformers_state import disable_xformers` with `ModuleNotFoundError: No module named 'ldm'`. Since the inference file lives in `external/DOSSR/scripts/` while `ldm/` is at the repo root, the wrapper now adds the configured DoSSR root to the child process `PYTHONPATH`. This does not modify upstream files. The pasted runtime also emitted `Could not find cuda drivers ... GPU will not be used`; that is a separate environment signal to inspect if CUDA/model initialization fails after the import fix. The source-path change is syntax-checked locally but still needs a new Colab run to verify.
- **Colab OpenCLIP initialization failure and compatibility launcher (2026-10-28):** after the import-path fix, DoSSR constructed its UNet but failed before loading the DoSSR checkpoint because `external/DOSSR/ldm/modules/encoders/modules.py:149` hard-codes `/group/ossmodelzoo/qinpecui/downloads/open_clip_pytorch_model.bin`, a file on the original author's machine. This is not one of the approved assets and must not be fetched from an inferred source. `src/dossr_inference_compat.py` intercepts only this exact missing path and asks OpenCLIP to construct the architecture without separate pretrained weights. Upstream inference then strictly loads the full `dossr_default.ckpt` (`scripts/inference.py:153`, `utils/common.py:20-43`); absent model keys will still fail strict loading. The launcher forwards the original inference CLI flags and does not edit `external/`. It is syntax-checked locally only and needs a Colab run to verify. The `cudart_stub` warning preceded this OpenCLIP failure; actual CUDA model execution has not yet been confirmed.
- **Colab initialization SIGKILL (2026-10-28):** the latest runner attempt printed `Global seed set to 42`, then shell reported `Killed` with exit 137 and no Python traceback. This confirms SIGKILL during setup, before image processing; the log shows T4 GPU memory was unused. Host RAM pressure during construction/loading of the large model is a plausible cause, but not confirmed because the earlier `memory.events` output was collected before this attempt and showed zero counters. The attempted one-image smoke setup copied an image but did not override `input_dir`; the runner still counted the full 200-image dataset. Changing input count would not reduce model initialization memory. Check cgroup memory events immediately after a failed run before deciding whether a High-RAM Colab runtime is needed.
- **One-image smoke SIGKILL follow-up (2026-10-28):** the corrected smoke config was read successfully and the runner counted exactly one image. DoSSR logged construction of its 865.91M-parameter DiffusionWrapper and the OpenCLIP compatibility path, then the child died about five minutes later with `SIGKILL: 9`; no Python traceback was emitted. The T4 remained at 0 MiB and post-run cgroup counters were zero. This locates the termination before image processing and before confirmed GPU model use, but does not prove RAM was the cause or identify whether termination occurred during remaining model construction or checkpoint loading. Next diagnostic attempt should use Colab High-RAM (keep T4 enabled) and the same one-image config; do not start the full dataset until smoke completes.
- **Checkpoint mmap optimization (2026-10-28):** `src/dossr_inference_compat.py` wraps `torch.load`, defaults to `mmap=True`, retries with `mmap=False` only when the runtime reports mmap unsupported, and runs `gc.collect()` immediately before and after the load attempt. The OpenCLIP path shim remains; `src/run_dossr.py` continues to set `PYTHONPATH` to the DoSSR root. The Colab log at 14:14 confirms mmap loading succeeded and execution reached the sampler; this verifies checkpoint loading, not successful inference.
- **T4 GPU OOM after successful checkpoint load (2026-10-28):** the one-image smoke run loaded the checkpoint with `mmap=True`, reached `Spaced Sampler: 0%`, then failed in `ControlLDM.apply_condition_encoder` (`external/DOSSR/model/cldm.py:320-324`, called by `model/shift_sampler.py:337`) with `torch.OutOfMemoryError`. PyTorch reports the T4 has 14.56 GiB total, only 1.81 GiB free, while the operation requests 2.00 GiB; this process has 12.75 GiB in use (12.53 GiB allocated by PyTorch). This is a confirmed **GPU** OOM. The resource screenshot's 0/15 GiB GPU reading is after process exit and does not contradict the traceback. System RAM at the screenshot was 1.6/12.7 GiB, so this attempt no longer shows the earlier host-RAM SIGKILL.
- The checked-in model/inference path uses float32 inputs (`scripts/inference.py:61-67`) and the UNet/ControlNet `use_fp16` defaults are false (`model/cldm.py:64`, `ldm/modules/diffusionmodules/openaimodel.py:456`). Inference accepts `--tiled`, `--tile_size`, and `--tile_stride`, but the checked `process()` function passes no tiling options to `shift_sample_3_order`; the `--tiled` branch only changes the resize cap (`scripts/inference.py:168-172`). Do not claim tiled inference fixes this OOM; changing scale/resolution would alter the baseline input. No precision/offload change has been approved or implemented. To preserve the current float32 baseline, use a GPU with more VRAM if available. A half-precision or offload implementation on T4 is a methodological/runtime change and should be reviewed and documented before use.
- **Diagnostic output adjustment (2026-10-28):** the runner now starts the DoSSR child with Python's `-u` flag so progress prints are flushed to the timestamped log even if the child is terminated. This improves failure evidence only; it does not change inference behavior or memory use. Local syntax check passes; Colab retry remains pending.
- **Colab notebook clarification (2026-10-28):** `notebooks/colab_runner.ipynb` now labels each numbered cell, pulls `feat/baseline-runner`, separates setup from optional asset verification, prepares a real one-image smoke config through `scripts/prepare_baseline_smoke.py`, and leaves the full 200-image command disabled until the smoke output is confirmed. All saved outputs are cleared. The notebook stays a thin front end to repository scripts.
- **Fresh-runtime Jedi conflict (2026-10-28):** a new Colab runtime's `pip check` reported `ipython 7.34.0 requires jedi, which is not installed`, so `colab_setup.py` correctly stopped before dependency installation. A prior successful Colab log installed `jedi==0.20.1` and `parso==0.8.7`, then `pip check` passed. Notebook Cell 5 now installs that previously verified Jedi version and checks the environment before Cell 6 runs DoSSR setup; this repair does not change torch.
