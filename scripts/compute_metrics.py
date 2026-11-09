#!/usr/bin/env python3
"""NOT TESTED: compute full-reference PSNR, SSIM, and LPIPS plus cross-seed stability metrics."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.env_info import DEFAULT_LOG_DIR, configure_logging, log_run_info  # noqa: E402

SEED_DIR_PATTERN = re.compile(r"^seed_(\d+)$")
PRED_SUFFIX = "_0"
QUARANTINE_MARKER = "README_DO_NOT_USE_AS_BASELINE.txt"


def parse_seeds(raw: str | None) -> list[int] | None:
    if raw is None or not raw.strip():
        return None
    seeds: list[int] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        seeds.append(int(token))
    if not seeds:
        raise ValueError("--seeds did not contain any valid integer seed values")
    return seeds


def discover_seed_dirs(output_root: Path, requested_seeds: list[int] | None) -> list[tuple[int, Path]]:
    if not output_root.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {output_root}")

    if (output_root / QUARANTINE_MARKER).exists():
        raise ValueError(
            f"Refusing to compute baseline metrics on quarantined directory containing {QUARANTINE_MARKER}: {output_root}"
        )

    direct_match = SEED_DIR_PATTERN.match(output_root.name)
    if direct_match:
        seed_val = int(direct_match.group(1))
        if requested_seeds is not None and seed_val not in requested_seeds:
            raise ValueError(f"Seed folder {output_root} does not match requested --seeds {requested_seeds}")
        return [(seed_val, output_root)]

    discovered: dict[int, Path] = {}
    for child in sorted(output_root.iterdir()):
        if not child.is_dir():
            continue
        match = SEED_DIR_PATTERN.match(child.name)
        if not match:
            continue
        if (child / QUARANTINE_MARKER).exists():
            raise ValueError(
                f"Refusing to compute baseline metrics on quarantined seed directory containing {QUARANTINE_MARKER}: {child}"
            )
        discovered[int(match.group(1))] = child

    if not discovered:
        raise FileNotFoundError(f"No seed_<N> directories found under {output_root}")

    if requested_seeds is not None:
        missing = [seed for seed in requested_seeds if seed not in discovered]
        if missing:
            raise FileNotFoundError(
                f"Requested seed directories missing under {output_root}: {[f'seed_{s}' for s in missing]}"
            )
        return [(seed, discovered[seed]) for seed in requested_seeds]

    return sorted(discovered.items(), key=lambda item: item[0])


def collect_predicted_images(seed_dir: Path) -> dict[str, Path]:
    pred_map: dict[str, Path] = {}
    for png_path in sorted(seed_dir.glob("*.png")):
        if png_path.name.endswith("_compare.png"):
            continue
        stem = png_path.stem
        if not stem.endswith(PRED_SUFFIX):
            continue
        base_stem = stem[: -len(PRED_SUFFIX)]
        if not base_stem:
            continue
        pred_map[base_stem] = png_path
    if not pred_map:
        raise FileNotFoundError(f"No predicted '<stem>_0.png' images found in {seed_dir}")
    return pred_map


def load_verified_rgb_uint8(path: Path, expected_size: int) -> Any:
    import numpy as np
    from PIL import Image

    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    with Image.open(path) as img:
        img.load()
        width, height = img.size
        if (width, height) != (expected_size, expected_size):
            raise ValueError(
                f"Expected image size ({expected_size}, {expected_size}) for {path}, got ({width}, {height})"
            )
        rgb = img.convert("RGB")
        arr = np.asarray(rgb, dtype=np.uint8)
    if arr.shape != (expected_size, expected_size, 3):
        raise ValueError(f"Unexpected RGB array shape {arr.shape} for {path}")
    return arr


def resolve_device(device_arg: str, torch_mod: Any) -> str:
    if device_arg == "auto":
        return "cuda" if torch_mod.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch_mod.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is False")
    return device_arg


def rgb_to_y_channel_pt(img_nchw: Any, torch_mod: Any) -> Any:
    """ITU-R BT.601 Y-channel conversion matching external/DOSSR/utils/image/common.py."""
    weight = torch_mod.tensor([[65.481], [128.553], [24.966]], dtype=img_nchw.dtype, device=img_nchw.device)
    out = torch_mod.matmul(img_nchw.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    return out / 255.0


def crop_border_pt(img_nchw: Any, crop_border: int) -> Any:
    if crop_border <= 0:
        return img_nchw
    return img_nchw[:, :, crop_border:-crop_border, crop_border:-crop_border]


def calculate_psnr_pt(img1_nchw: Any, img2_nchw: Any, crop_border: int, test_y_channel: bool, torch_mod: Any) -> float:
    """PSNR (dB) matching external/DOSSR/utils/metrics.py."""
    img1 = crop_border_pt(img1_nchw.to(torch_mod.float64), crop_border)
    img2 = crop_border_pt(img2_nchw.to(torch_mod.float64), crop_border)
    if test_y_channel:
        img1 = rgb_to_y_channel_pt(img1, torch_mod)
        img2 = rgb_to_y_channel_pt(img2, torch_mod)
    mse = torch_mod.mean((img1 - img2) ** 2, dim=[1, 2, 3])
    psnr = 10.0 * torch_mod.log10(1.0 / (mse + 1e-8))
    return float(psnr.item())


def _gaussian_window_2d(window_size: int, sigma: float, channels: int, dtype: Any, device: Any, torch_mod: Any) -> Any:
    coords = torch_mod.arange(window_size, dtype=dtype, device=device) - (window_size - 1) / 2.0
    g1d = torch_mod.exp(-(coords**2) / (2.0 * sigma**2))
    g1d = g1d / g1d.sum()
    g2d = torch_mod.outer(g1d, g1d)
    window = g2d.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)
    return window


def calculate_ssim_pt(img1_nchw: Any, img2_nchw: Any, crop_border: int, test_y_channel: bool, torch_mod: Any) -> float:
    """Standard 11x11 Gaussian-window SSIM (Wang et al. 2004 / BasicSR) in float64."""
    F = torch_mod.nn.functional
    img1 = crop_border_pt(img1_nchw.to(torch_mod.float64), crop_border)
    img2 = crop_border_pt(img2_nchw.to(torch_mod.float64), crop_border)
    if test_y_channel:
        img1 = rgb_to_y_channel_pt(img1, torch_mod)
        img2 = rgb_to_y_channel_pt(img2, torch_mod)

    channels = int(img1.shape[1])
    window = _gaussian_window_2d(11, 1.5, channels, img1.dtype, img1.device, torch_mod)

    mu1 = F.conv2d(img1, window, padding=0, groups=channels)
    mu2 = F.conv2d(img2, window, padding=0, groups=channels)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=0, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=0, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=0, groups=channels) - mu1_mu2

    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(ssim_map.mean().item())


def summarize_series(values: list[float], seeds: list[int], higher_is_better: bool) -> dict[str, float | int]:
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if len(arr) >= 2 else 0.0
    min_idx = int(np.argmin(arr))
    max_idx = int(np.argmax(arr))
    best_idx = max_idx if higher_is_better else min_idx
    worst_idx = min_idx if higher_is_better else max_idx
    return {
        "mean": mean_val,
        "std": std_val,
        "min": float(arr[min_idx]),
        "max": float(arr[max_idx]),
        "range": float(arr[max_idx] - arr[min_idx]),
        "best_seed": int(seeds[best_idx]),
        "worst_seed": int(seeds[worst_idx]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        "--study-root",
        dest="output_root",
        type=Path,
        required=True,
        help="Root directory containing seed_<N> folders (or a single seed_<N> folder)",
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        required=True,
        help="Ground-truth HR directory containing <stem>.png images",
    )
    parser.add_argument(
        "--expected-size",
        type=int,
        default=512,
        help="Expected width and height of both predicted and GT images (default: 512)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Optional comma-separated seed list to require/evaluate (e.g. 42,43,44,45,46)",
    )
    parser.add_argument(
        "--crop-border",
        type=int,
        default=0,
        help="Border pixels cropped on each edge before PSNR/SSIM (default: 0)",
    )
    parser.add_argument(
        "--test-y-channel",
        action="store_true",
        help="Log Y-channel PSNR/SSIM as primary in console summary (CSV always records both RGB and Y)",
    )
    parser.add_argument(
        "--lpips-net",
        choices=("alex", "vgg"),
        default="alex",
        help="Backbone network for LPIPS (default: alex)",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Computation device for PyTorch/LPIPS (default: auto)",
    )
    parser.add_argument(
        "--per-image-csv",
        type=Path,
        default=None,
        help="Output CSV path for per-(image, seed) rows (default: <output-root>/metrics_per_seed.csv)",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Output CSV path for per-image across-seed summary rows (default: <output-root>/metrics_summary_by_image.csv)",
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    args = parser.parse_args()

    logger, log_path = configure_logging("compute_metrics", args.log_dir)
    log_run_info(logger, args, seed=None)
    logger.info("Log file: %s", log_path)

    output_root = args.output_root.expanduser().resolve()
    gt_dir = args.gt_dir.expanduser().resolve()
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Ground-truth directory does not exist: {gt_dir}")

    requested_seeds = parse_seeds(args.seeds)
    seed_dirs = discover_seed_dirs(output_root, requested_seeds)
    seeds = [seed for seed, _ in seed_dirs]
    logger.info("Discovered %d seed folder(s): %s", len(seed_dirs), [f"seed_{s}" for s in seeds])

    per_seed_preds: dict[int, dict[str, Path]] = {}
    reference_stems: list[str] | None = None
    for seed, seed_dir in seed_dirs:
        pred_map = collect_predicted_images(seed_dir)
        stems = sorted(pred_map.keys())
        if reference_stems is None:
            reference_stems = stems
        elif stems != reference_stems:
            raise ValueError(
                f"Image stem mismatch in {seed_dir}: expected {reference_stems}, got {stems}"
            )
        per_seed_preds[seed] = pred_map

    assert reference_stems is not None
    logger.info("Verified %d common image stem(s) across all seeds: %s", len(reference_stems), reference_stems)

    gt_paths: dict[str, Path] = {}
    for stem in reference_stems:
        candidate = gt_dir / f"{stem}.png"
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Missing ground-truth HR image for stem {stem!r}: expected {candidate}"
            )
        gt_paths[stem] = candidate

    import lpips
    import numpy as np
    import torch

    device = resolve_device(args.device, torch)
    logger.info("Using device=%s | lpips_net=%s | expected_size=%d | crop_border=%d", device, args.lpips_net, args.expected_size, args.crop_border)

    lpips_fn = lpips.LPIPS(net=args.lpips_net).to(device)
    lpips_fn.eval()
    for param in lpips_fn.parameters():
        param.requires_grad_(False)

    per_image_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        total_stems = len(reference_stems)
        for idx, stem in enumerate(reference_stems, start=1):
            gt_path = gt_paths[stem]
            gt_uint8 = load_verified_rgb_uint8(gt_path, args.expected_size)
            gt_tensor = (
                torch.from_numpy(gt_uint8.astype(np.float32) / 255.0)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
            )

            seed_uint8_map: dict[int, np.ndarray] = {}
            seed_tensor_map: dict[int, Any] = {}
            stem_rows: list[dict[str, Any]] = []

            for seed in seeds:
                pred_path = per_seed_preds[seed][stem]
                pred_uint8 = load_verified_rgb_uint8(pred_path, args.expected_size)
                pred_tensor = (
                    torch.from_numpy(pred_uint8.astype(np.float32) / 255.0)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(device)
                )
                seed_uint8_map[seed] = pred_uint8
                seed_tensor_map[seed] = pred_tensor

                psnr_rgb = calculate_psnr_pt(pred_tensor, gt_tensor, args.crop_border, False, torch)
                ssim_rgb = calculate_ssim_pt(pred_tensor, gt_tensor, args.crop_border, False, torch)
                psnr_y = calculate_psnr_pt(pred_tensor, gt_tensor, args.crop_border, True, torch)
                ssim_y = calculate_ssim_pt(pred_tensor, gt_tensor, args.crop_border, True, torch)
                lpips_val = float(lpips_fn(pred_tensor, gt_tensor, normalize=True).item())

                row = {
                    "image_stem": stem,
                    "seed": seed,
                    "psnr_rgb": round(psnr_rgb, 6),
                    "ssim_rgb": round(ssim_rgb, 6),
                    "psnr_y": round(psnr_y, 6),
                    "ssim_y": round(ssim_y, 6),
                    "lpips": round(lpips_val, 6),
                    "pred_path": str(pred_path),
                    "gt_path": str(gt_path),
                }
                stem_rows.append(row)
                per_image_rows.append(row)

            pairwise_px_means: list[float] = []
            pairwise_px_maxes: list[float] = []
            pairwise_lpips_vals: list[float] = []
            for seed_a, seed_b in itertools.combinations(seeds, 2):
                diff_px = np.abs(
                    seed_uint8_map[seed_a].astype(np.float32) - seed_uint8_map[seed_b].astype(np.float32)
                )
                pairwise_px_means.append(float(diff_px.mean()))
                pairwise_px_maxes.append(float(diff_px.max()))
                pair_lp = float(lpips_fn(seed_tensor_map[seed_a], seed_tensor_map[seed_b], normalize=True).item())
                pairwise_lpips_vals.append(pair_lp)

            psnr_rgb_stats = summarize_series([r["psnr_rgb"] for r in stem_rows], seeds, higher_is_better=True)
            ssim_rgb_stats = summarize_series([r["ssim_rgb"] for r in stem_rows], seeds, higher_is_better=True)
            psnr_y_stats = summarize_series([r["psnr_y"] for r in stem_rows], seeds, higher_is_better=True)
            ssim_y_stats = summarize_series([r["ssim_y"] for r in stem_rows], seeds, higher_is_better=True)
            lpips_stats = summarize_series([r["lpips"] for r in stem_rows], seeds, higher_is_better=False)

            summary_row: dict[str, Any] = {
                "image_stem": stem,
                "seed_count": len(seeds),
                "seeds": ",".join(str(s) for s in seeds),
                "psnr_rgb_mean": round(float(psnr_rgb_stats["mean"]), 6),
                "psnr_rgb_std": round(float(psnr_rgb_stats["std"]), 6),
                "psnr_rgb_min": round(float(psnr_rgb_stats["min"]), 6),
                "psnr_rgb_max": round(float(psnr_rgb_stats["max"]), 6),
                "psnr_rgb_range": round(float(psnr_rgb_stats["range"]), 6),
                "psnr_rgb_best_seed": psnr_rgb_stats["best_seed"],
                "psnr_rgb_worst_seed": psnr_rgb_stats["worst_seed"],
                "ssim_rgb_mean": round(float(ssim_rgb_stats["mean"]), 6),
                "ssim_rgb_std": round(float(ssim_rgb_stats["std"]), 6),
                "ssim_rgb_min": round(float(ssim_rgb_stats["min"]), 6),
                "ssim_rgb_max": round(float(ssim_rgb_stats["max"]), 6),
                "ssim_rgb_range": round(float(ssim_rgb_stats["range"]), 6),
                "ssim_rgb_best_seed": ssim_rgb_stats["best_seed"],
                "ssim_rgb_worst_seed": ssim_rgb_stats["worst_seed"],
                "psnr_y_mean": round(float(psnr_y_stats["mean"]), 6),
                "psnr_y_std": round(float(psnr_y_stats["std"]), 6),
                "psnr_y_range": round(float(psnr_y_stats["range"]), 6),
                "ssim_y_mean": round(float(ssim_y_stats["mean"]), 6),
                "ssim_y_std": round(float(ssim_y_stats["std"]), 6),
                "ssim_y_range": round(float(ssim_y_stats["range"]), 6),
                "lpips_mean": round(float(lpips_stats["mean"]), 6),
                "lpips_std": round(float(lpips_stats["std"]), 6),
                "lpips_min": round(float(lpips_stats["min"]), 6),
                "lpips_max": round(float(lpips_stats["max"]), 6),
                "lpips_range": round(float(lpips_stats["range"]), 6),
                "lpips_best_seed": lpips_stats["best_seed"],
                "lpips_worst_seed": lpips_stats["worst_seed"],
                "pairwise_seed_px_mean_abs_diff": round(float(np.mean(pairwise_px_means)) if pairwise_px_means else 0.0, 6),
                "pairwise_seed_px_max_abs_diff": round(float(np.max(pairwise_px_maxes)) if pairwise_px_maxes else 0.0, 6),
                "pairwise_seed_lpips_mean": round(float(np.mean(pairwise_lpips_vals)) if pairwise_lpips_vals else 0.0, 6),
                "pairwise_seed_lpips_max": round(float(np.max(pairwise_lpips_vals)) if pairwise_lpips_vals else 0.0, 6),
            }
            summary_rows.append(summary_row)
            logger.info(
                "[%d/%d] %s | PSNR_RGB=%.4f±%.4f | SSIM_RGB=%.4f | LPIPS=%.4f | pair_px=%.4f | pair_lpips=%.4f",
                idx,
                total_stems,
                stem,
                summary_row["psnr_rgb_mean"],
                summary_row["psnr_rgb_std"],
                summary_row["ssim_rgb_mean"],
                summary_row["lpips_mean"],
                summary_row["pairwise_seed_px_mean_abs_diff"],
                summary_row["pairwise_seed_lpips_mean"],
            )

    # Append an overall dataset average row across all evaluated images
    numeric_cols = [
        k
        for k in summary_rows[0].keys()
        if k not in {"image_stem", "seed_count", "seeds"} and not k.endswith("_seed")
    ]
    overall_row: dict[str, Any] = {
        "image_stem": "ALL_IMAGES_MEAN",
        "seed_count": len(seeds),
        "seeds": ",".join(str(s) for s in seeds),
    }
    for key in summary_rows[0].keys():
        if key in {"image_stem", "seed_count", "seeds"}:
            continue
        if key.endswith("_seed"):
            overall_row[key] = ""
        elif key in numeric_cols:
            overall_row[key] = round(float(np.mean([float(r[key]) for r in summary_rows])), 6)

    per_image_csv = (
        args.per_image_csv.expanduser().resolve()
        if args.per_image_csv is not None
        else (output_root / "metrics_per_seed.csv")
    )
    summary_csv = (
        args.summary_csv.expanduser().resolve()
        if args.summary_csv is not None
        else (output_root / "metrics_summary_by_image.csv")
    )
    per_image_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    with per_image_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_image_rows)

    all_summary_rows = summary_rows + [overall_row]
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_summary_rows)

    logger.info("Wrote per-(image, seed) CSV (%d rows): %s", len(per_image_rows), per_image_csv)
    logger.info("Wrote per-image summary CSV (%d images + ALL_IMAGES_MEAN): %s", len(summary_rows), summary_csv)

    print(f"=== Per-Image Across-Seed Summary ({len(seeds)} seeds: {seeds}) ===")
    for row in all_summary_rows:
        psnr_key = "psnr_y" if args.test_y_channel else "psnr_rgb"
        ssim_key = "ssim_y" if args.test_y_channel else "ssim_rgb"
        print(
            f"{row['image_stem']:18s} | "
            f"{psnr_key.upper()}={row[f'{psnr_key}_mean']:.4f}±{row[f'{psnr_key}_std']:.4f} (range={row[f'{psnr_key}_range']:.4f}) | "
            f"{ssim_key.upper()}={row[f'{ssim_key}_mean']:.4f}±{row[f'{ssim_key}_std']:.4f} | "
            f"LPIPS={row['lpips_mean']:.4f}±{row['lpips_std']:.4f} (range={row['lpips_range']:.4f}) | "
            f"pair_px_mean={row['pairwise_seed_px_mean_abs_diff']:.4f} | "
            f"pair_lpips_mean={row['pairwise_seed_lpips_mean']:.4f}"
        )
    print(f"Wrote: {per_image_csv}")
    print(f"Wrote: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
