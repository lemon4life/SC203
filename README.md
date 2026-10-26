# SC203 Scientific Methods — Team 1 LEMMEPASS

This repository contains the analysis and experiment setup for studying whether lightweight text-instruction conditioning reduces DoSSR's sensitivity to the random seed.

## Local analysis environment

The root `requirements.txt` lists the pinned Python packages for analysis, metric calculations, and notebooks. It is derived from the package versions in [`environment-local.yml`](environment-local.yml), which is the Conda export from the Apple Silicon Mac environment. The requirements are for analysis only; they are not the full DoSSR or InstructIR inference/training environments.

For Windows, use Python 3.11 and choose a matching Windows PyTorch/torchvision build for the machine's CPU or NVIDIA/CUDA setup before installing the remaining requirements. The pinned versions have not been validated as a Windows installation. Check OS, GPU/CUDA availability, and free disk space before installing.

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.26.0
python -m pip install -r requirements.txt
```

For a CUDA-enabled Windows setup, install the PyTorch/torchvision pair from the official PyTorch selector for the installed CUDA runtime instead of assuming the default pip build is appropriate. This analysis environment does not promise that DoSSR or InstructIR inference works on Windows.

## Project folders

- `configs/`: experiment configuration
- `src/`: project-owned analysis and experiment code
- `scripts/`: helper scripts, including the Phase 1 asset download and verification tools
- `notebooks/`: exploratory and analysis notebooks
- `prompts/`: instruction prompts and prompt sets
- `results/`: tracked summaries and figures; large outputs are ignored
- `docs/`: setup and research notes
- `external/`: local third-party clones; ignored by Git

Model weights, datasets, checkpoints, and generated outputs are not tracked. The approved Phase 1 assets are downloaded to Google Drive and verified in Colab; see [`docs/setup_notes.md`](docs/setup_notes.md).
