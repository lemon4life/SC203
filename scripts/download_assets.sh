#!/usr/bin/env bash
# Phase 1 assets only: DoSSR multi-step, RealESRNet, and StableSR test bundle.
# Intended for Colab. This script is not run automatically.
set -euo pipefail

TARGET_DIR="/content/drive/MyDrive/lemmepass/"
ALLOW_LARGE=0

usage() {
  cat <<'EOF'
Usage: bash scripts/download_assets.sh [--target-dir DIR] [--allow-large-assets]

Options:
  --target-dir DIR       Destination (default: /content/drive/MyDrive/lemmepass/)
  --allow-large-assets   Explicitly permit downloading the 3.06 GB bundle.
  -h, --help             Show this help.

This Phase 1 script does not download Stable Diffusion 2.1-base or RealLR200.
Place only dossr_default.ckpt at weights/dossr_default.ckpt in the target
folder (or use the Drive UI copy fallback described in docs/setup_notes.md).
Required commands: curl and unzip.
EOF
}

while (($#)); do
  case "$1" in
    --target-dir)
      [[ $# -ge 2 ]] || { echo "Missing value for --target-dir" >&2; exit 2; }
      TARGET_DIR="$2"
      shift 2
      ;;
    --allow-large-assets)
      ALLOW_LARGE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

for tool in curl unzip; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "Required command not found: $tool" >&2
    exit 1
  }
done

mkdir -p "$TARGET_DIR/weights" "$TARGET_DIR/data"

REESR_URL="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
REESR_OUT="$TARGET_DIR/weights/RealESRNet_x4plus.pth"
if [[ ! -s "$REESR_OUT" ]]; then
  curl --fail --location --retry 3 "$REESR_URL" --output "$REESR_OUT"
else
  echo "Already present: $REESR_OUT"
fi

DOSSR_OUT="$TARGET_DIR/weights/dossr_default.ckpt"
if [[ -s "$DOSSR_OUT" ]]; then
  echo "Already present: $DOSSR_OUT"
else
  echo "DoSSR checkpoint not found at $DOSSR_OUT. Copy only dossr_default.ckpt from the upstream Drive folder into that path."
  echo "Source: https://drive.google.com/drive/folders/1BTGRbAqXDSNWjyUnTLgDfmjd2xyu8GvP?usp=sharing"
fi

STABLESR_ZIP="$TARGET_DIR/data/StableSR_testsets.zip"
if [[ ! -s "$STABLESR_ZIP" ]]; then
  if (( ! ALLOW_LARGE )); then
    echo "StableSR test bundle is 3.06 GB. Rerun with --allow-large-assets to download." >&2
    exit 2
  fi
  echo "Downloading the approved 3.06 GB StableSR test bundle."
  curl --fail --location --retry 3 \
    "https://huggingface.co/datasets/Iceclear/StableSR-TestSets/resolve/main/StableSR_testsets.zip?download=true" \
    --output "$STABLESR_ZIP"
else
  echo "Already present: $STABLESR_ZIP"
fi

EXTRACT_DIR="$TARGET_DIR/data/StableSR_testsets"
if [[ -s "$STABLESR_ZIP" && ! -d "$EXTRACT_DIR" ]]; then
  mkdir -p "$EXTRACT_DIR"
  unzip -q "$STABLESR_ZIP" -d "$EXTRACT_DIR"
fi

echo "Phase 1 download steps finished. DoSSR checkpoint present: $([[ -s "$DOSSR_OUT" ]] && echo yes || echo no)."
