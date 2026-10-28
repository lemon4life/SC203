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

Sizes below are publisher/repository listing sizes where available. Actual byte sizes and SHA-256 values must be filled from the Colab verifier output after download; assets are intentionally not present on this Mac.

| Asset | Source | Expected size | Colab path | Actual size / SHA-256 |
| --- | --- | ---: | --- | --- |
| DoSSR multi-step checkpoint `dossr_default.ckpt` | [DoSSR Google Drive folder](https://drive.google.com/drive/folders/1BTGRbAqXDSNWjyUnTLgDfmjd2xyu8GvP?usp=sharing) | Not published for the individual file | `/content/drive/MyDrive/lemmepass/weights/dossr_default.ckpt` | Pending Colab verification |
| `RealESRNet_x4plus.pth` | [Official Real-ESRGAN v0.1.1 release](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth) | About 67 MB | `/content/drive/MyDrive/lemmepass/weights/RealESRNet_x4plus.pth` | Pending Colab verification |
| StableSR test bundle `StableSR_testsets.zip` | [StableSR-TestSets on Hugging Face](https://huggingface.co/datasets/Iceclear/StableSR-TestSets) | 3.06 GB for the complete archive | `/content/drive/MyDrive/lemmepass/data/StableSR_testsets.zip` | Pending Colab verification |
| Extracted RealSR / DRealSR image folders | In the StableSR archive | Per-folder sizes not published | `/content/drive/MyDrive/lemmepass/data/StableSR_testsets/` (exact nested paths discovered after extraction) | Verifier reports image counts; paths pending |

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
- The setup script and notebook are **UNTESTED on Colab**. Only local compilation and notebook-structure checks are intended before handoff; no package installation or download is performed locally.
- Kernel/team decision: use Colab's managed Python kernel rather than creating a venv/Conda environment inside the runtime; the setup cell now invokes scripts with `sys.executable` so pip targets that kernel. For four teammates, use separate personal runtimes, shared Drive access for the single asset copy, and GitHub branches/commits as the code and result provenance source. This is workflow guidance, not a tested Colab run.
