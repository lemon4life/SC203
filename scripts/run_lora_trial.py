#!/usr/bin/env python3
"""Controlled LoRA pilot training runner (LoRA 1 / LoRA 2 / LoRA 3) for DoSSR.

Trains LoRA adapters on paired Real-CE (LR, HR, mask) data with:
- Paired data loading bypassing RealESRGANBatchTransform._dequeue_and_enqueue
- Empty prompt conditioning ("") matching baseline finding
- Shared seed / noise-draw loop across LoRA 2 (weighted text L1) and LoRA 3 (+ cross-noise consistency)
- Two-pass exact chain-rule backward for K=2 noise samples
- Peak VRAM and s/step logging to lora_train_log.csv and lora_run_manifest.json
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import statistics
import sys
import tempfile
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.lora_text_loss import (  # noqa: E402
    boxes_to_hr_mask,
    downsample_mask_to_latent,
    get_lora_parameters,
    inject_lora_adapters,
    load_lora_state_dict,
    run_two_noise_step,
    save_lora_state_dict,
)


AUTHOR_OPENCLIP_CHECKPOINT = Path(
    "/group/ossmodelzoo/qinpecui/downloads/open_clip_pytorch_model.bin"
)


def seed_everything_deterministic(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch deterministically across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_boxes_index(boxes_csv: Path | None) -> dict[str, list[tuple[float, float, float, float]]]:
    """Load optional OCR/text bounding boxes keyed by image name and stem."""
    index: dict[str, list[tuple[float, float, float, float]]] = {}
    if boxes_csv is None or not boxes_csv.is_file():
        return index
    with boxes_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image_ref = row.get("image", "").strip()
            if not image_ref:
                continue
            try:
                x = float(row["x"])
                y = float(row["y"])
                w = float(row["w"])
                h = float(row["h"])
            except (KeyError, TypeError, ValueError):
                continue
            if w <= 0.0 or h <= 0.0:
                continue
            box = (x, y, w, h)
            name = Path(image_ref).name
            stem = Path(image_ref).stem
            index.setdefault(name, []).append(box)
            if stem != name:
                index.setdefault(stem, []).append(box)
    return index


class PairedRealCEDataset(Dataset):
    """Paired (LR, HR, mask) dataset reading a split manifest CSV directly."""

    def __init__(
        self,
        manifest_csv: Path,
        *,
        boxes_csv: Path | None = None,
        image_size: int = 512,
        sr_scale: int = 4,
        split_filter: str = "train",
        seed: int = 231,
    ) -> None:
        super().__init__()
        if not manifest_csv.is_file():
            raise FileNotFoundError(f"Training manifest CSV not found: {manifest_csv}")
        with manifest_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = [
                r for r in csv.DictReader(handle) if r.get("split", "").strip() == split_filter
            ]
        if not rows:
            raise ValueError(
                f"No rows with split=={split_filter!r} found in {manifest_csv}"
            )
        self.samples: list[dict[str, str]] = []
        for row in rows:
            lr_p = Path(row["lr_path"]).expanduser()
            hr_p = Path(row["hr_path"]).expanduser()
            if not lr_p.is_file() or not hr_p.is_file():
                raise FileNotFoundError(
                    f"Missing paired file for pair_id={row.get('pair_id')}: lr={lr_p}, hr={hr_p}"
                )
            self.samples.append(row)

        self.boxes_index = load_boxes_index(boxes_csv)
        self.image_size = int(image_size)
        self.sr_scale = int(sr_scale)
        self.seed = int(seed)
        self.bicubic_resample = getattr(getattr(Image, "Resampling", Image), "BICUBIC")

    def __len__(self) -> int:
        return len(self.samples)

    def _lookup_boxes(self, lr_path: Path, hr_path: Path, pair_id: str) -> list[tuple[float, float, float, float]]:
        for key in (
            hr_path.name,
            hr_path.stem,
            lr_path.name,
            lr_path.stem,
            pair_id,
            Path(pair_id).name,
        ):
            if key in self.boxes_index:
                return self.boxes_index[key]
        return []

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.samples[idx]
        lr_path = Path(row["lr_path"]).expanduser()
        hr_path = Path(row["hr_path"]).expanduser()
        lr_im = Image.open(lr_path).convert("RGB")
        hr_im = Image.open(hr_path).convert("RGB")

        hr_w, hr_h = hr_im.size
        lr_w, lr_h = lr_im.size
        if (hr_w, hr_h) != (lr_w * self.sr_scale, lr_h * self.sr_scale):
            raise ValueError(
                f"Scale mismatch for {row.get('pair_id')}: HR={hr_im.size}, LR={lr_im.size}, scale={self.sr_scale}"
            )
        if hr_w < self.image_size or hr_h < self.image_size:
            raise ValueError(
                f"Image {hr_path} ({hr_im.size}) smaller than target crop ({self.image_size})"
            )

        boxes = self._lookup_boxes(lr_path, hr_path, row.get("pair_id", ""))
        if hr_w != self.image_size or hr_h != self.image_size:
            crop_rng = random.Random(self.seed + idx)
            max_left_lr = (hr_w - self.image_size) // self.sr_scale
            max_top_lr = (hr_h - self.image_size) // self.sr_scale
            left_lr = crop_rng.randint(0, max_left_lr)
            top_lr = crop_rng.randint(0, max_top_lr)
            lr_patch_sz = self.image_size // self.sr_scale
            lr_im = lr_im.crop((left_lr, top_lr, left_lr + lr_patch_sz, top_lr + lr_patch_sz))
            left_hr = left_lr * self.sr_scale
            top_hr = top_lr * self.sr_scale
            hr_im = hr_im.crop((left_hr, top_hr, left_hr + self.image_size, top_hr + self.image_size))
            boxes = [(x - left_hr, y - top_hr, w, h) for (x, y, w, h) in boxes]

        lr_up = lr_im.resize((self.image_size, self.image_size), resample=self.bicubic_resample)

        hr_arr = np.asarray(hr_im, dtype=np.float32) / 127.5 - 1.0
        hint_arr = np.asarray(lr_up, dtype=np.float32) / 255.0
        mask_hr = boxes_to_hr_mask(
            boxes, height=self.image_size, width=self.image_size, dtype=torch.float32
        ).squeeze(0)

        return {
            "jpg": torch.from_numpy(hr_arr),
            "hint": torch.from_numpy(hint_arr),
            "txt": "",
            "mask_hr": mask_hr,
            "pair_id": row.get("pair_id", hr_path.stem),
        }


def load_dossr_for_lora(
    *,
    dossr_root: Path,
    model_config: Path,
    checkpoint: Path,
    device: str,
    rank: int,
    alpha: float,
) -> torch.nn.Module:
    """Instantiate ControlLDM, load strict DoSSR weights, and inject trainable LoRA."""
    dossr_str = str(dossr_root.expanduser().resolve())
    if dossr_str not in sys.path:
        sys.path.insert(0, dossr_str)

    import open_clip
    from omegaconf import OmegaConf
    from utils.common import instantiate_from_config, load_state_dict

    orig_create = open_clip.create_model_and_transforms

    def _safe_create(*call_args: Any, **call_kwargs: Any) -> Any:
        if (
            call_kwargs.get("pretrained") == str(AUTHOR_OPENCLIP_CHECKPOINT)
            and not AUTHOR_OPENCLIP_CHECKPOINT.is_file()
        ):
            call_kwargs["pretrained"] = None
        return orig_create(*call_args, **call_kwargs)

    open_clip.create_model_and_transforms = _safe_create
    try:
        cfg = OmegaConf.load(str(model_config.expanduser().resolve()))
        model = instantiate_from_config(cfg)
    finally:
        open_clip.create_model_and_transforms = orig_create

    ckpt_state = torch.load(str(checkpoint.expanduser().resolve()), map_location="cpu")
    load_state_dict(model, ckpt_state, strict=True)

    model = model.to(device)
    model.eval()
    inject_lora_adapters(model, rank=rank, alpha=alpha)
    return model


def resolve_variant_hyperparams(
    variant: str,
    alpha_text: float,
    lambda_cons: float,
    alpha_cons_text: float,
) -> tuple[float, float, float]:
    """Validate and enforce variant-specific loss hyperparameters."""
    if variant == "lora1":
        return 1.0, 0.0, 1.0
    if variant == "lora2":
        if alpha_text <= 1.0:
            raise ValueError(f"lora2 requires alpha_text > 1.0, got {alpha_text}")
        return float(alpha_text), 0.0, float(alpha_cons_text)
    if variant == "lora3":
        if alpha_text <= 1.0:
            raise ValueError(f"lora3 requires alpha_text > 1.0, got {alpha_text}")
        if lambda_cons <= 0.0:
            raise ValueError(f"lora3 requires lambda_cons > 0.0, got {lambda_cons}")
        return float(alpha_text), float(lambda_cons), float(alpha_cons_text)
    raise ValueError(f"Unsupported variant: {variant}")


def run_training_loop(
    *,
    model: Any,
    dataloader: DataLoader,
    output_dir: Path,
    variant: str,
    rank: int,
    alpha: float,
    lr: float,
    max_steps: int,
    grad_accum: int,
    num_noise_samples: int,
    alpha_text: float,
    lambda_cons: float,
    alpha_cons_text: float,
    seed: int,
    device: str,
    precision: str,
    log_every: int,
    save_every: int,
    config_meta: dict[str, Any],
) -> dict[str, Any]:
    """Run controlled gradient-accumulated LoRA training and emit logs, weights, and manifest."""
    if num_noise_samples != 2:
        raise ValueError(
            f"Controlled pilot fixes num_noise_samples == 2, got {num_noise_samples}"
        )
    alpha_text_val, lambda_cons_val, alpha_cons_val = resolve_variant_hyperparams(
        variant, alpha_text, lambda_cons, alpha_cons_text
    )

    seed_everything_deterministic(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    lora_params = get_lora_parameters(model)
    trainable_params = sum(int(p.numel()) for p in lora_params)
    if trainable_params == 0:
        raise RuntimeError("No trainable LoRA parameters found on model")

    optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=0.0)
    use_amp = device.startswith("cuda") and precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    data_iter = iter(dataloader)
    log_rows: list[dict[str, Any]] = []
    step_times: list[float] = []

    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        t_start = time.monotonic()

        accum_metrics = {
            "total_loss": 0.0,
            "rec_loss": 0.0,
            "global_l1": 0.0,
            "text_l1": 0.0,
            "cons_loss": 0.0,
        }

        for _ in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            with torch.no_grad():
                x_hr_lat, cond = model.get_input(batch, model.first_stage_key)
                preprocess_x = cond["c_preprocess"][0]
                mask_hr = batch["mask_hr"].to(device=device, dtype=torch.float32)
                mask_lat = downsample_mask_to_latent(
                    mask_hr, latent_h=x_hr_lat.shape[-2], latent_w=x_hr_lat.shape[-1]
                )
                t = torch.randint(
                    0, model.num_timesteps, (x_hr_lat.shape[0],), device=device
                ).long()
                noise_a = torch.randn_like(x_hr_lat)
                noise_b = torch.randn_like(x_hr_lat)

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_amp
                else nullcontext()
            )
            with autocast_ctx:
                step_stats = run_two_noise_step(
                    lambda eps: model.predict_x0_out_mix(
                        x_hr_lat, preprocess_x, cond, t, noise=eps
                    ),
                    x_hr_lat,
                    mask_lat,
                    noise_a,
                    noise_b,
                    alpha_text=alpha_text_val,
                    lambda_cons=lambda_cons_val,
                    alpha_cons_text=alpha_cons_val,
                    loss_scale=1.0 / float(grad_accum),
                    loss_backward_fn=(
                        (lambda loss: scaler.scale(loss).backward()) if use_amp else None
                    ),
                )
            for k in accum_metrics:
                accum_metrics[k] += step_stats[k] / float(grad_accum)

        if use_amp:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_alloc = float(torch.cuda.max_memory_allocated() / (1024.0**3))
            peak_res = float(torch.cuda.max_memory_reserved() / (1024.0**3))
        else:
            peak_alloc = 0.0
            peak_res = 0.0

        step_sec = time.monotonic() - t_start
        step_times.append(step_sec)

        row = {
            "step": step,
            "variant": variant,
            "total_loss": round(accum_metrics["total_loss"], 6),
            "rec_loss": round(accum_metrics["rec_loss"], 6),
            "global_l1": round(accum_metrics["global_l1"], 6),
            "text_l1": round(accum_metrics["text_l1"], 6),
            "cons_loss": round(accum_metrics["cons_loss"], 6),
            "lr": lr,
            "step_time_sec": round(step_sec, 4),
            "peak_vram_allocated_gib": round(peak_alloc, 4),
            "peak_vram_reserved_gib": round(peak_res, 4),
        }
        log_rows.append(row)

        if step == 1 or step % log_every == 0 or step == max_steps:
            print(
                f"[{variant}] step={step}/{max_steps} | total={row['total_loss']:.4f} "
                f"| rec={row['rec_loss']:.4f} | text_l1={row['text_l1']:.4f} "
                f"| cons={row['cons_loss']:.4f} | {row['step_time_sec']:.2f}s/step "
                f"| peak_vram={peak_alloc:.2f} GiB",
                flush=True,
            )

        if save_every > 0 and step % save_every == 0 and step < max_steps:
            ckpt_step = output_dir / f"lora_weights_step_{step:04d}.pt"
            save_lora_state_dict(
                model, ckpt_step, meta={"variant": variant, "step": step, "rank": rank, "alpha": alpha}
            )

    final_weights_path = output_dir / "lora_weights_final.pt"
    save_lora_state_dict(
        model,
        final_weights_path,
        meta={"variant": variant, "step": max_steps, "rank": rank, "alpha": alpha},
    )

    log_csv = output_dir / "lora_train_log.csv"
    with log_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)

    sorted_times = sorted(step_times)
    p95_idx = min(len(sorted_times) - 1, math.floor(0.95 * (len(sorted_times) - 1)))
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "variant": variant,
        "seed": seed,
        "prompt": "",
        "device": device,
        "precision": precision,
        "rank": rank,
        "alpha": alpha,
        "lr": lr,
        "max_steps": max_steps,
        "grad_accum": grad_accum,
        "num_noise_samples": num_noise_samples,
        "alpha_text": alpha_text_val,
        "lambda_cons": lambda_cons_val,
        "alpha_cons_text": alpha_cons_val,
        "trainable_lora_params": trainable_params,
        "mean_step_time_sec": round(statistics.mean(step_times), 4),
        "p95_step_time_sec": round(sorted_times[p95_idx], 4),
        "peak_vram_allocated_gib": round(max(r["peak_vram_allocated_gib"] for r in log_rows), 4),
        "peak_vram_reserved_gib": round(max(r["peak_vram_reserved_gib"] for r in log_rows), 4),
        "weights_path": str(final_weights_path),
        "train_log_csv": str(log_csv),
        "extra_config": config_meta,
    }
    manifest_path = output_dir / "lora_run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def run_dry_run_cpu() -> int:
    """Verify full dataset -> LoRA 2 & LoRA 3 training -> checkpoint load -> run_dossr plumbing on CPU."""
    from src.run_dossr import build_command, manifest_for

    class _MockControlLDM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first_stage_key = "jpg"
            self.num_timesteps = 1000
            self.control_model = torch.nn.ModuleDict(
                {"to_q": torch.nn.Linear(4, 4), "to_k": torch.nn.Linear(4, 4)}
            )
            self.model = torch.nn.ModuleDict(
                {
                    "diffusion_model": torch.nn.ModuleDict(
                        {
                            "output_blocks": torch.nn.ModuleList(
                                [torch.nn.ModuleDict({"to_v": torch.nn.Linear(4, 4)})]
                            )
                        }
                    )
                }
            )

        def get_input(self, batch: dict[str, Any], first_stage_key: str) -> tuple[torch.Tensor, dict[str, Any]]:
            x_hr = batch[first_stage_key].permute(0, 3, 1, 2).float()
            hint = batch["hint"].permute(0, 3, 1, 2).float()
            x_lat = torch.nn.functional.avg_pool2d(x_hr, kernel_size=4)
            x_lat = torch.cat([x_lat, x_lat[:, :1]], dim=1)
            hint_lat = torch.nn.functional.avg_pool2d(hint, kernel_size=4)
            hint_lat = torch.cat([hint_lat, hint_lat[:, :1]], dim=1)
            return x_lat, {"c_preprocess": [hint_lat], "txt": batch["txt"]}

        def predict_x0_out_mix(
            self,
            x_start: torch.Tensor,
            x_start_hat: torch.Tensor,
            cond: dict[str, Any],
            t: torch.Tensor,
            noise: torch.Tensor,
        ) -> torch.Tensor:
            noisy = (0.5 * x_start + 0.5 * x_start_hat + 0.1 * noise).permute(0, 2, 3, 1)
            ctrl = self.control_model["to_q"](noisy) + self.control_model["to_k"](noisy)
            out = self.model["diffusion_model"]["output_blocks"][0]["to_v"](noisy + ctrl)
            return out.permute(0, 3, 1, 2)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for idx in range(2):
            lr_im = Image.new("RGB", (32, 32), color=(40 + idx * 30, 80, 120))
            hr_im = Image.new("RGB", (128, 128), color=(40 + idx * 30, 80, 120))
            lr_im.save(tmp / f"scene_{idx:03d}_LR4.png")
            hr_im.save(tmp / f"scene_{idx:03d}_HR.png")

        manifest_csv = tmp / "train_manifest.csv"
        with manifest_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["split", "scene_key", "pair_id", "lr_path", "hr_path"]
            )
            writer.writeheader()
            for idx in range(2):
                writer.writerow(
                    {
                        "split": "train",
                        "scene_key": f"scene_{idx:03d}",
                        "pair_id": f"scene_{idx:03d}",
                        "lr_path": str(tmp / f"scene_{idx:03d}_LR4.png"),
                        "hr_path": str(tmp / f"scene_{idx:03d}_HR.png"),
                    }
                )

        boxes_csv = tmp / "boxes.csv"
        with boxes_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["image", "box_id", "x", "y", "w", "h", "transcript", "readable"]
            )
            writer.writeheader()
            writer.writerow(
                {
                    "image": "scene_000_HR.png",
                    "box_id": 1,
                    "x": 16,
                    "y": 16,
                    "w": 64,
                    "h": 32,
                    "transcript": "test",
                    "readable": 1,
                }
            )

        ds = PairedRealCEDataset(
            manifest_csv, boxes_csv=boxes_csv, image_size=128, sr_scale=4, seed=231
        )
        assert len(ds) == 2
        sample0 = ds[0]
        assert sample0["jpg"].shape == (128, 128, 3)
        assert sample0["hint"].shape == (128, 128, 3)
        assert sample0["txt"] == ""
        assert float(sample0["mask_hr"].sum().item()) == 64 * 32

        for variant, alpha_t, l_cons in (("lora2", 5.0, 0.0), ("lora3", 5.0, 0.5)):
            loader = DataLoader(ds, batch_size=1, shuffle=False)
            mock_model = _MockControlLDM()
            mock_model.eval()
            inject_lora_adapters(mock_model, rank=2, alpha=4.0)
            out_dir = tmp / f"out_{variant}"
            summary = run_training_loop(
                model=mock_model,
                dataloader=loader,
                output_dir=out_dir,
                variant=variant,
                rank=2,
                alpha=4.0,
                lr=1e-3,
                max_steps=2,
                grad_accum=2,
                num_noise_samples=2,
                alpha_text=alpha_t,
                lambda_cons=l_cons,
                alpha_cons_text=alpha_t,
                seed=231,
                device="cpu",
                precision="fp32",
                log_every=1,
                save_every=0,
                config_meta={"dry_run_cpu": True},
            )
            assert (out_dir / "lora_train_log.csv").is_file()
            assert (out_dir / "lora_run_manifest.json").is_file()
            weights_file = Path(summary["weights_path"])
            assert weights_file.is_file()
            fresh = _MockControlLDM()
            loaded = load_lora_state_dict(fresh, weights_file)
            assert len(loaded) == 3

        base_cfg = {
            "seed": 42,
            "input_dir": "in",
            "checkpoint": "ckpt.ckpt",
            "model_config": "cfg.yaml",
            "inference_script": "inf.py",
            "steps": 5,
            "solver_order": 3,
            "t1_fraction": 0.5,
            "device": "cuda",
            "sr_scale": 4,
            "color_fix_type": "wavelet",
            "preprocessor": "RealESRNet",
            "precision": "fp16",
            "prompt_mode": "empty",
            "output_root": "out",
        }
        assert "lora_weights" not in manifest_for(base_cfg)
        assert "--lora-weights" not in build_command(base_cfg, tmp / "b")
        lora_cfg = dict(base_cfg, lora_weights="w.pt", per_image_seed=True)
        assert manifest_for(lora_cfg)["per_image_seed"] is True
        cmd = build_command(lora_cfg, tmp / "l")
        assert "--lora-weights" in cmd and "--per-image-seed" in cmd

    print("[PASS] run_lora_trial.py --dry-run-cpu (LoRA 2, LoRA 3, dataset, checkpoints, run_dossr plumbing)")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Controlled LoRA 1/2/3 pilot training for DoSSR.")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config with 'lora:' section")
    parser.add_argument("--variant", choices=("lora1", "lora2", "lora3"), default=None)
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--boxes-csv", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--dossr-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--num-noise-samples", type=int, default=None)
    parser.add_argument("--alpha-text", type=float, default=None)
    parser.add_argument("--lambda-cons", type=float, default=None)
    parser.add_argument("--alpha-cons-text", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--sr-scale", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--dry-run-cpu", action="store_true", help="Run offline CPU smoke verification")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run_cpu:
        return run_dry_run_cpu()

    raw_cfg: dict[str, Any] = {}
    if args.config is not None:
        raw_cfg = yaml.safe_load(args.config.expanduser().read_text(encoding="utf-8")) or {}
    lora_cfg = dict(raw_cfg.get("lora", {}))

    def pick(key: str, cli_val: Any, default: Any = None) -> Any:
        if cli_val is not None:
            return cli_val
        if key in lora_cfg and lora_cfg[key] is not None:
            return lora_cfg[key]
        if key in raw_cfg and raw_cfg[key] is not None:
            return raw_cfg[key]
        return default

    variant = pick("variant", args.variant)
    train_manifest = pick("train_manifest", args.train_manifest)
    checkpoint = pick("checkpoint", args.checkpoint)
    model_config = pick("model_config", args.model_config, REPO_ROOT / "external/DOSSR/configs/model/cldm_v21.yaml")
    dossr_root = pick("dossr_root", args.dossr_root, REPO_ROOT / "external/DOSSR")
    output_dir = pick("output_dir", args.output_dir)

    missing = [
        name
        for name, val in (
            ("variant", variant),
            ("train_manifest", train_manifest),
            ("checkpoint", checkpoint),
            ("output_dir", output_dir),
        )
        if not val
    ]
    if missing:
        raise ValueError(f"Missing required arguments/config keys: {missing}")

    boxes_csv_raw = pick("boxes_csv", args.boxes_csv, None)
    boxes_csv = Path(boxes_csv_raw).expanduser() if boxes_csv_raw else None

    rank = int(pick("rank", args.rank, 8))
    alpha = float(pick("alpha", args.alpha, 16.0))
    lr = float(pick("lr", args.lr, 1e-4))
    max_steps = int(pick("max_steps", args.max_steps, 400))
    batch_size = int(pick("batch_size", args.batch_size, 1))
    grad_accum = int(pick("grad_accum", args.grad_accum, 4))
    num_noise_samples = int(pick("num_noise_samples", args.num_noise_samples, 2))
    alpha_text = float(pick("alpha_text", args.alpha_text, 1.0 if variant == "lora1" else 5.0))
    lambda_cons = float(pick("lambda_cons", args.lambda_cons, 0.5 if variant == "lora3" else 0.0))
    alpha_cons_text = float(pick("alpha_cons_text", args.alpha_cons_text, alpha_text))
    image_size = int(pick("image_size", args.image_size, 512))
    sr_scale = int(pick("sr_scale", args.sr_scale, 4))
    seed = int(pick("seed", args.seed, 231))
    precision = str(pick("precision", args.precision, "fp16"))
    device = str(pick("device", args.device, "cuda"))
    log_every = int(pick("log_every", args.log_every, 10))
    save_every = int(pick("save_every", args.save_every, 200))

    seed_everything_deterministic(seed)
    dataset = PairedRealCEDataset(
        Path(train_manifest).expanduser(),
        boxes_csv=boxes_csv,
        image_size=image_size,
        sr_scale=sr_scale,
        split_filter="train",
        seed=seed,
    )
    generator = torch.Generator().manual_seed(seed)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        generator=generator,
    )

    model = load_dossr_for_lora(
        dossr_root=Path(dossr_root),
        model_config=Path(model_config),
        checkpoint=Path(checkpoint),
        device=device,
        rank=rank,
        alpha=alpha,
    )

    run_training_loop(
        model=model,
        dataloader=dataloader,
        output_dir=Path(output_dir).expanduser(),
        variant=variant,
        rank=rank,
        alpha=alpha,
        lr=lr,
        max_steps=max_steps,
        grad_accum=grad_accum,
        num_noise_samples=num_noise_samples,
        alpha_text=alpha_text,
        lambda_cons=lambda_cons,
        alpha_cons_text=alpha_cons_text,
        seed=seed,
        device=device,
        precision=precision,
        log_every=log_every,
        save_every=save_every,
        config_meta={"config_path": str(args.config) if args.config else None},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
