#!/usr/bin/env python3
"""LoRA adapters, text-region masks, losses, and two-pass consistency step for DoSSR."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_TARGET_PREFIXES = ("control_model.", "model.diffusion_model.output_blocks.")
DEFAULT_TARGET_LEAF_SUFFIXES = (".to_q", ".to_k", ".to_v", ".to_out.0")


class LoRALinear(nn.Module):
    """Low-rank adapter wrapping a frozen nn.Linear layer (exact baseline at init)."""

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        if alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {alpha}")
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad_(False)

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        ref_weight = base_layer.weight
        self.lora_A = nn.Parameter(
            torch.empty(
                (self.rank, self.in_features),
                dtype=ref_weight.dtype,
                device=ref_weight.device,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                (self.out_features, self.rank),
                dtype=ref_weight.dtype,
                device=ref_weight.device,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        delta = F.linear(
            F.linear(self.lora_dropout(x.to(self.lora_A.dtype)), self.lora_A),
            self.lora_B,
        )
        return base_out + (delta * self.scaling).to(base_out.dtype)


def _matches_target(
    full_name: str,
    target_prefixes: Sequence[str] | None,
    target_leaf_suffixes: Sequence[str] | None,
) -> bool:
    """Return True if a named nn.Linear satisfies prefix and leaf-suffix filters."""
    if target_prefixes:
        if not any(full_name.startswith(prefix) for prefix in target_prefixes):
            return False
    if target_leaf_suffixes:
        if not any(
            full_name.endswith(sfx) or full_name == sfx.lstrip(".")
            for sfx in target_leaf_suffixes
        ):
            return False
    return True


def _set_submodule(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    """Replace a nested submodule identified by its dotted name."""
    parts = dotted_name.split(".")
    parent: nn.Module = root
    for attr in parts[:-1]:
        parent = getattr(parent, attr)
    setattr(parent, parts[-1], new_module)


def inject_lora_adapters(
    model: nn.Module,
    *,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_prefixes: Sequence[str] | None = DEFAULT_TARGET_PREFIXES,
    target_leaf_suffixes: Sequence[str] | None = DEFAULT_TARGET_LEAF_SUFFIXES,
    freeze_non_lora: bool = True,
) -> list[str]:
    """Freeze base parameters and replace matching nn.Linear modules with LoRALinear."""
    if freeze_non_lora:
        for param in model.parameters():
            param.requires_grad_(False)

    targets: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            continue
        if isinstance(module, nn.Linear) and _matches_target(
            name, target_prefixes, target_leaf_suffixes
        ):
            targets.append((name, module))

    if not targets:
        raise ValueError(
            "inject_lora_adapters found zero matching nn.Linear modules "
            f"(prefixes={target_prefixes}, suffixes={target_leaf_suffixes})"
        )

    replaced: list[str] = []
    for name, linear_mod in targets:
        wrapped = LoRALinear(linear_mod, rank=rank, alpha=alpha, dropout=dropout)
        _set_submodule(model, name, wrapped)
        replaced.append(name)
    return replaced


def get_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Return trainable LoRA parameters in deterministic named order."""
    params: list[nn.Parameter] = []
    for _, module in model.named_modules():
        if isinstance(module, LoRALinear):
            params.extend([module.lora_A, module.lora_B])
    return params


def save_lora_state_dict(
    model: nn.Module,
    path: Path,
    *,
    meta: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """Save only LoRA A/B weights (and optional metadata) to disk."""
    lora_state: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[f"{name}.lora_A"] = module.lora_A.detach().cpu().clone()
            lora_state[f"{name}.lora_B"] = module.lora_B.detach().cpu().clone()
            lora_state[f"{name}.alpha"] = torch.tensor(module.alpha, dtype=torch.float32)
            lora_state[f"{name}.rank"] = torch.tensor(module.rank, dtype=torch.int64)
    if not lora_state:
        raise ValueError("No LoRALinear modules found when saving LoRA state dict")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"lora_state_dict": lora_state}
    if meta is not None:
        payload["meta"] = meta
    torch.save(payload, path)
    return lora_state


def load_lora_state_dict(
    model: nn.Module,
    path: Path,
    *,
    default_rank: int = 8,
    default_alpha: float = 16.0,
) -> list[str]:
    """Load saved LoRA weights into a base or already-injected model."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    lora_state = payload["lora_state_dict"] if "lora_state_dict" in payload else payload
    meta = payload.get("meta", {}) if isinstance(payload, dict) else {}

    module_keys = sorted(
        {k[: -len(".lora_A")] for k in lora_state if k.endswith(".lora_A")}
    )
    if not module_keys:
        raise ValueError(f"No '*.lora_A' entries found in LoRA checkpoint: {path}")

    existing_modules = dict(model.named_modules())
    loaded: list[str] = []
    for mod_name in module_keys:
        a_tensor = lora_state[f"{mod_name}.lora_A"]
        b_tensor = lora_state[f"{mod_name}.lora_B"]
        rank = int(
            lora_state.get(f"{mod_name}.rank", meta.get("rank", a_tensor.shape[0]))
        )
        alpha = float(
            lora_state.get(f"{mod_name}.alpha", meta.get("alpha", default_alpha))
        )
        if mod_name not in existing_modules:
            raise KeyError(f"Target module '{mod_name}' not found in model")
        cur = existing_modules[mod_name]
        if not isinstance(cur, LoRALinear):
            if not isinstance(cur, nn.Linear):
                raise TypeError(f"Module '{mod_name}' is {type(cur)}, expected nn.Linear")
            cur = LoRALinear(cur, rank=rank or default_rank, alpha=alpha)
            _set_submodule(model, mod_name, cur)
        with torch.no_grad():
            cur.lora_A.copy_(a_tensor.to(device=cur.lora_A.device, dtype=cur.lora_A.dtype))
            cur.lora_B.copy_(b_tensor.to(device=cur.lora_B.device, dtype=cur.lora_B.dtype))
            cur.alpha = alpha
            cur.scaling = alpha / float(cur.rank)
        loaded.append(mod_name)
    return loaded


def boxes_to_hr_mask(
    boxes_xywh: Iterable[Sequence[float | int]],
    *,
    height: int = 512,
    width: int = 512,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Rasterize pixel bounding boxes (x, y, w, h) into [1, 1, H, W] binary mask."""
    mask = torch.zeros((1, 1, height, width), dtype=dtype, device=device)
    for box in boxes_xywh:
        if len(box) < 4:
            raise ValueError(f"Expected 4 coordinates (x, y, w, h), got {box}")
        x0 = max(0, min(width, int(math.floor(float(box[0])))))
        y0 = max(0, min(height, int(math.floor(float(box[1])))))
        x1 = max(x0, min(width, int(math.ceil(float(box[0]) + float(box[2])))))
        y1 = max(y0, min(height, int(math.ceil(float(box[1]) + float(box[3])))))
        if x1 > x0 and y1 > y0:
            mask[:, :, y0:y1, x0:x1] = 1.0
    return mask


def downsample_mask_to_latent(
    hr_mask: torch.Tensor,
    *,
    latent_h: int,
    latent_w: int,
) -> torch.Tensor:
    """Downsample [B, 1, H, W] HR mask via max pooling so touched text cells remain 1."""
    if hr_mask.ndim != 4 or hr_mask.shape[1] != 1:
        raise ValueError(f"Expected [B, 1, H, W] HR mask, got shape {tuple(hr_mask.shape)}")
    return F.adaptive_max_pool2d(hr_mask.float(), output_size=(latent_h, latent_w))


@dataclass(frozen=True)
class StepLossOutput:
    """Scalar losses computed for one step or one half of a two-pass step."""

    total: torch.Tensor
    rec_loss: float
    global_l1: float
    text_l1: float
    cons_loss: float


def _build_text_weight_map(mask_lat: torch.Tensor, alpha_text: float) -> torch.Tensor:
    """Build pixel/latent spatial weight map w = 1 + (alpha_text - 1) * mask_lat."""
    return 1.0 + (float(alpha_text) - 1.0) * mask_lat


def compute_lora_step_losses(
    pred_x0: torch.Tensor,
    target_x0: torch.Tensor,
    mask_lat: torch.Tensor,
    *,
    alpha_text: float = 1.0,
    pred_peer_detached: torch.Tensor | None = None,
    lambda_cons: float = 0.0,
    alpha_cons_text: float = 1.0,
    loss_scale: float = 1.0,
) -> StepLossOutput:
    """Compute weighted reconstruction L1 plus optional consistency L1 against peer prediction."""
    if pred_x0.shape != target_x0.shape:
        raise ValueError(f"Shape mismatch: pred_x0={pred_x0.shape} vs target_x0={target_x0.shape}")
    if mask_lat.shape[0] != pred_x0.shape[0] or mask_lat.shape[2:] != pred_x0.shape[2:]:
        raise ValueError(
            f"Latent mask shape {tuple(mask_lat.shape)} incompatible with latent {tuple(pred_x0.shape)}"
        )

    mask_f = mask_lat.to(dtype=pred_x0.dtype, device=pred_x0.device)
    err = (pred_x0 - target_x0).abs()
    w_rec = _build_text_weight_map(mask_f, alpha_text)
    rec_tensor = (w_rec * err).mean()

    global_l1 = float(err.detach().mean().item())
    mask_expanded = mask_f.expand_as(err)
    mask_denom = float(mask_expanded.sum().item())
    text_l1 = (
        float((mask_expanded * err.detach()).sum().item() / mask_denom)
        if mask_denom > 0.0
        else 0.0
    )

    cons_tensor = pred_x0.new_zeros(())
    if pred_peer_detached is not None and lambda_cons > 0.0:
        w_cons = _build_text_weight_map(mask_f, alpha_cons_text)
        cons_tensor = (w_cons * (pred_x0 - pred_peer_detached).abs()).mean()

    total = loss_scale * (rec_tensor + float(lambda_cons) * cons_tensor)
    return StepLossOutput(
        total=total,
        rec_loss=float(rec_tensor.detach().item()),
        global_l1=global_l1,
        text_l1=text_l1,
        cons_loss=float(cons_tensor.detach().item()),
    )


def run_two_noise_step(
    forward_fn: Any,
    target_x0: torch.Tensor,
    mask_lat: torch.Tensor,
    noise_a: torch.Tensor,
    noise_b: torch.Tensor,
    *,
    alpha_text: float = 1.0,
    lambda_cons: float = 0.0,
    alpha_cons_text: float = 1.0,
    loss_scale: float = 1.0,
    cons_active: bool = True,
    loss_backward_fn: Any = None,
    backward: bool = True,
) -> dict[str, float]:
    """Execute exact chain-rule two-pass gradient accumulation for two noise draws.

    Peak activation memory stays equal to one forward pass because Pass 1 frees its
    computation graph before Pass 2 builds its own, while gradients equal
    loss_scale * (0.5*L_rec(p_a) + 0.5*L_rec(p_b) + eff_lambda_cons*L_cons(p_a, p_b)).
    """
    bw = loss_backward_fn if loss_backward_fn is not None else (lambda t: t.backward())
    eff_lambda_cons = float(lambda_cons) if (cons_active and float(lambda_cons) > 0.0) else 0.0

    def _pass_objective(
        pred: torch.Tensor, peer_detached: torch.Tensor | None
    ) -> tuple[torch.Tensor, StepLossOutput]:
        rec_out = compute_lora_step_losses(
            pred,
            target_x0,
            mask_lat,
            alpha_text=alpha_text,
            pred_peer_detached=None,
            lambda_cons=0.0,
        )
        obj = 0.5 * rec_out.total
        if peer_detached is not None and eff_lambda_cons > 0.0:
            w_cons = _build_text_weight_map(mask_lat.to(pred.dtype), alpha_cons_text)
            obj = obj + eff_lambda_cons * (w_cons * (pred - peer_detached).abs()).mean()
        return obj * float(loss_scale), rec_out

    pred_b_ref: torch.Tensor | None = None
    if eff_lambda_cons > 0.0:
        with torch.no_grad():
            pred_b_ref = forward_fn(noise_b).detach()

    pred_a = forward_fn(noise_a)
    pred_a_ref = pred_a.detach()
    pass1_obj, rec_a = _pass_objective(pred_a, pred_b_ref)
    if backward:
        bw(pass1_obj)
    del pred_a, pass1_obj

    pred_b = forward_fn(noise_b)
    pass2_obj, rec_b = _pass_objective(pred_b, pred_a_ref if eff_lambda_cons > 0.0 else None)
    if backward:
        bw(pass2_obj)

    with torch.no_grad():
        cons_val = (
            float(
                (
                    _build_text_weight_map(mask_lat.to(pred_b.dtype), alpha_cons_text)
                    * (pred_a_ref - pred_b.detach()).abs()
                )
                .mean()
                .item()
            )
            if eff_lambda_cons > 0.0
            else 0.0
        )
        rec_mean = 0.5 * (rec_a.rec_loss + rec_b.rec_loss)
        global_l1 = 0.5 * (rec_a.global_l1 + rec_b.global_l1)
        text_l1 = 0.5 * (rec_a.text_l1 + rec_b.text_l1)

    return {
        "total_loss": rec_mean + eff_lambda_cons * cons_val,
        "rec_loss": rec_mean,
        "global_l1": global_l1,
        "text_l1": text_l1,
        "cons_loss": cons_val,
        "cons_active": 1.0 if eff_lambda_cons > 0.0 else 0.0,
    }


def audit_dossr_eta_and_timesteps(
    num_timesteps: int = 1000,
    num_steps: int = 5,
    linear_start: float = 0.00085,
    linear_end: float = 0.0120,
) -> dict[str, Any]:
    """Replicate DoSSR's trunc_12 eta_t and SD2.1 alpha_bar_t schedules for math auditing."""
    t1 = num_timesteps // 2
    etas = torch.ones(num_timesteps, dtype=torch.float64)
    ramp = torch.linspace(0.001, 0.999, t1, dtype=torch.float64)
    etas[:t1] = (1.0 - torch.cos(ramp * math.pi)) / 2.0
    one_minus_etas = 1.0 - etas

    betas = torch.linspace(linear_start**0.5, linear_end**0.5, num_timesteps, dtype=torch.float64) ** 2
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1.0)

    # SpacedSampler space_timesteps(t1, str(num_steps + 1))
    section_count = num_steps + 1
    frac_stride = (t1 - 1) / (section_count - 1)
    cur_idx = 0.0
    spaced_timesteps = []
    for _ in range(section_count):
        spaced_timesteps.append(round(cur_idx))
        cur_idx += frac_stride
    inference_eval_timesteps = list(reversed(spaced_timesteps[1:]))

    return {
        "num_timesteps": num_timesteps,
        "t1_trunc_12": t1,
        "active_hr_mix_range": [0, t1 - 1],
        "pure_lr_prior_range": [t1, num_timesteps - 1],
        "eta_at_t": {
            "0": float(etas[0].item()),
            "100": float(etas[100].item()),
            "200": float(etas[200].item()),
            "250": float(etas[250].item()),
            "299": float(etas[299].item()),
            "300": float(etas[300].item()),
            "399": float(etas[399].item()),
            "400": float(etas[400].item()),
            "498": float(etas[498].item()),
            "499": float(etas[499].item()),
            "500": float(etas[500].item()),
            "999": float(etas[999].item()),
        },
        "one_minus_eta_at_t": {
            "0": float(one_minus_etas[0].item()),
            "100": float(one_minus_etas[100].item()),
            "200": float(one_minus_etas[200].item()),
            "250": float(one_minus_etas[250].item()),
            "299": float(one_minus_etas[299].item()),
            "300": float(one_minus_etas[300].item()),
            "399": float(one_minus_etas[399].item()),
            "400": float(one_minus_etas[400].item()),
            "498": float(one_minus_etas[498].item()),
            "499": float(one_minus_etas[499].item()),
            "500": float(one_minus_etas[500].item()),
            "999": float(one_minus_etas[999].item()),
        },
        "sqrt_alphas_cumprod_at_t": {
            "0": float(sqrt_alphas_cumprod[0].item()),
            "499": float(sqrt_alphas_cumprod[499].item()),
            "500": float(sqrt_alphas_cumprod[500].item()),
            "999": float(sqrt_alphas_cumprod[999].item()),
        },
        "sqrt_recip_alphas_cumprod_at_t": {
            "0": float(sqrt_recip_alphas_cumprod[0].item()),
            "499": float(sqrt_recip_alphas_cumprod[499].item()),
            "500": float(sqrt_recip_alphas_cumprod[500].item()),
            "999": float(sqrt_recip_alphas_cumprod[999].item()),
        },
        "sqrt_recipm1_alphas_cumprod_at_t": {
            "0": float(sqrt_recipm1_alphas_cumprod[0].item()),
            "499": float(sqrt_recipm1_alphas_cumprod[499].item()),
            "500": float(sqrt_recipm1_alphas_cumprod[500].item()),
            "999": float(sqrt_recipm1_alphas_cumprod[999].item()),
        },
        "sqrt_one_minus_alphas_cumprod_at_t": {
            "499": float(sqrt_one_minus_alphas_cumprod[499].item()),
            "999": float(sqrt_one_minus_alphas_cumprod[999].item()),
        },
        "training_divides_by_one_minus_eta": False,
        "training_divisor": "sqrt_alphas_cumprod_t inside predict_start_from_noise (finite across 0..999)",
        "inference_divides_by_one_minus_eta": True,
        "inference_spaced_timesteps": spaced_timesteps,
        "inference_unet_eval_timesteps": inference_eval_timesteps,
        "inference_min_one_minus_eta_used": float(one_minus_etas[499].item()),
        "recommended_cons_t_range": [0, t1 - 1],
    }


def verify_fp32_twopass_vs_joint_gradients(
    *,
    forward_factory: Any = None,
    grad_accum: int = 4,
    alpha_text: float = 5.0,
    lambda_cons: float = 0.2,
    alpha_cons_text: float = 5.0,
    seed: int = 231,
) -> dict[str, Any]:
    """Compare custom two-pass chain-rule gradients against simultaneous two-branch backward in FP32."""
    torch.manual_seed(seed)

    if forward_factory is None:
        class _FP32AuditNet(nn.Module):
            def __init__(self, channels: int = 4) -> None:
                super().__init__()
                self.input_blocks = nn.ModuleList([nn.Linear(channels, channels)])
                self.middle_block = nn.Linear(channels, channels)
                self.control_model = nn.ModuleDict({"to_q": nn.Linear(channels, channels)})
                self.model = nn.ModuleDict(
                    {
                        "diffusion_model": nn.ModuleDict(
                            {
                                "output_blocks": nn.ModuleList(
                                    [nn.ModuleDict({"to_v": nn.Linear(channels, channels)})]
                                )
                            }
                        )
                    }
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                tokens = x.permute(0, 2, 3, 1)
                with torch.no_grad():
                    enc = self.middle_block(self.input_blocks[0](tokens))
                ctrl = self.control_model["to_q"](tokens)
                out = self.model["diffusion_model"]["output_blocks"][0]["to_v"](enc + ctrl)
                return out.permute(0, 3, 1, 2)

        net_joint = _FP32AuditNet(channels=4).float()
        inject_lora_adapters(net_joint, rank=4, alpha=8.0)
        for mod in net_joint.modules():
            if isinstance(mod, LoRALinear):
                nn.init.normal_(mod.lora_A, std=0.1)
                nn.init.normal_(mod.lora_B, std=0.1)
        net_twopass = copy.deepcopy(net_joint).float()

        microbatches = []
        for _ in range(grad_accum):
            x_lat = torch.randn(1, 4, 16, 16, dtype=torch.float32)
            target_lat = torch.randn(1, 4, 16, 16, dtype=torch.float32)
            mask = (torch.rand(1, 1, 16, 16) > 0.65).float()
            noise_a = torch.randn(1, 4, 16, 16, dtype=torch.float32)
            noise_b = torch.randn(1, 4, 16, 16, dtype=torch.float32)
            microbatches.append((x_lat, target_lat, mask, noise_a, noise_b))

        def _make_fwd(model_ref: nn.Module, x_in: torch.Tensor):
            return lambda eps: model_ref(x_in + eps)

        net_joint.zero_grad(set_to_none=True)
        net_twopass.zero_grad(set_to_none=True)
        joint_loss_total = 0.0
        twopass_loss_total = 0.0
        loss_scale = 1.0 / float(grad_accum)
        for x_lat, target_lat, mask, noise_a, noise_b in microbatches:
            p_a = _make_fwd(net_joint, x_lat)(noise_a)
            p_b = _make_fwd(net_joint, x_lat)(noise_b)
            w_rec = _build_text_weight_map(mask, alpha_text)
            w_cons = _build_text_weight_map(mask, alpha_cons_text)
            l_mb = loss_scale * (
                0.5 * (w_rec * (p_a - target_lat).abs()).mean()
                + 0.5 * (w_rec * (p_b - target_lat).abs()).mean()
                + float(lambda_cons) * (w_cons * (p_a - p_b).abs()).mean()
            )
            joint_loss_total += float(l_mb.detach().item())
            l_mb.backward()

            tp_stats = run_two_noise_step(
                _make_fwd(net_twopass, x_lat),
                target_lat,
                mask,
                noise_a,
                noise_b,
                alpha_text=alpha_text,
                lambda_cons=lambda_cons,
                alpha_cons_text=alpha_cons_text,
                loss_scale=loss_scale,
                cons_active=True,
            )
            twopass_loss_total += loss_scale * tp_stats["total_loss"]

        joint_params = get_lora_parameters(net_joint)
        twopass_params = get_lora_parameters(net_twopass)
    else:
        return forward_factory(grad_accum=grad_accum, alpha_text=alpha_text, lambda_cons=lambda_cons, alpha_cons_text=alpha_cons_text, seed=seed)

    abs_diffs: list[torch.Tensor] = []
    joint_sq_norm = 0.0
    diff_sq_norm = 0.0
    per_param_max: list[float] = []
    for p_j, p_t in zip(joint_params, twopass_params):
        assert p_j.grad is not None and p_t.grad is not None
        assert p_j.grad.dtype == torch.float32 and p_t.grad.dtype == torch.float32
        diff = (p_j.grad - p_t.grad).abs()
        abs_diffs.append(diff.reshape(-1))
        per_param_max.append(float(diff.max().item()))
        joint_sq_norm += float(p_j.grad.double().pow(2).sum().item())
        diff_sq_norm += float((p_j.grad.double() - p_t.grad.double()).pow(2).sum().item())

    flat_diff = torch.cat(abs_diffs)
    joint_norm = math.sqrt(joint_sq_norm)
    diff_norm = math.sqrt(diff_sq_norm)
    rel_l2 = diff_norm / joint_norm if joint_norm > 0.0 else diff_norm

    return {
        "precision": "fp32",
        "grad_accum": grad_accum,
        "num_lora_tensors": len(joint_params),
        "num_lora_elements": int(flat_diff.numel()),
        "joint_loss": joint_loss_total,
        "twopass_loss": twopass_loss_total,
        "loss_abs_diff": abs(joint_loss_total - twopass_loss_total),
        "max_abs_grad_diff": float(flat_diff.max().item()),
        "mean_abs_grad_diff": float(flat_diff.mean().item()),
        "rel_l2_grad_diff": rel_l2,
        "joint_grad_l2_norm": joint_norm,
    }


def run_cpu_tests() -> None:
    """Run all 6 CPU unit tests (a)-(f) verifying correctness without model weights."""

    class _MockControlledNet(nn.Module):
        def __init__(self, channels: int = 8) -> None:
            super().__init__()
            self.input_blocks = nn.ModuleList([nn.Linear(channels, channels)])
            self.middle_block = nn.Linear(channels, channels)
            self.control_model = nn.ModuleDict({"to_q": nn.Linear(channels, channels)})
            self.model = nn.ModuleDict(
                {
                    "diffusion_model": nn.ModuleDict(
                        {
                            "output_blocks": nn.ModuleList(
                                [nn.ModuleDict({"to_v": nn.Linear(channels, channels)})]
                            )
                        }
                    )
                }
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, c, h, w = x.shape
            tokens = x.permute(0, 2, 3, 1)
            with torch.no_grad():
                enc = self.input_blocks[0](tokens)
                enc = self.middle_block(enc)
            ctrl = self.control_model["to_q"](tokens)
            dec = self.model["diffusion_model"]["output_blocks"][0]["to_v"](enc + ctrl)
            return dec.permute(0, 3, 1, 2)

    torch.manual_seed(0)

    # (a) LoRALinear zero-init equivalence & freeze guarantees
    base = nn.Linear(8, 8)
    x_vec = torch.randn(3, 8)
    expected = base(x_vec).detach().clone()
    lora_layer = LoRALinear(base, rank=4, alpha=8.0)
    assert not lora_layer.base_layer.weight.requires_grad
    assert lora_layer.lora_A.requires_grad and lora_layer.lora_B.requires_grad
    max_diff_a = float((lora_layer(x_vec) - expected).abs().max().item())
    assert max_diff_a < 1e-6, f"Test (a) failed: init diff {max_diff_a}"
    print("[PASS] Test (a): LoRALinear zero-init preserves base output and freezes base weights")

    # (b) Injection skips no_grad input/middle blocks and injects only control + output_blocks
    net = _MockControlledNet(channels=4)
    x_lat = torch.randn(1, 4, 8, 8)
    base_out = net(x_lat).detach().clone()
    replaced = inject_lora_adapters(net, rank=2, alpha=4.0)
    assert replaced == [
        "control_model.to_q",
        "model.diffusion_model.output_blocks.0.to_v",
    ], replaced
    assert float((net(x_lat) - base_out).abs().max().item()) < 1e-6
    net(x_lat).square().mean().backward()
    lora_params = get_lora_parameters(net)
    assert len(lora_params) == 4
    assert any(p.grad is not None and p.grad.abs().max().item() > 0 for p in lora_params)
    assert net.input_blocks[0].weight.grad is None and not net.input_blocks[0].weight.requires_grad
    net.zero_grad(set_to_none=True)
    print("[PASS] Test (b): Selective LoRA injection on control_model + output_blocks skips no_grad blocks")

    # (c) HR mask rasterization & latent max-pool downsampling
    empty_mask = boxes_to_hr_mask([], height=512, width=512)
    assert float(empty_mask.sum().item()) == 0.0
    hr_mask = boxes_to_hr_mask([(320, 38, 192, 46)], height=512, width=512)
    assert float(hr_mask.sum().item()) == 192 * 46
    lat_mask = downsample_mask_to_latent(hr_mask, latent_h=64, latent_w=64)
    assert lat_mask.shape == (1, 1, 64, 64)
    assert float(lat_mask.max().item()) == 1.0 and float(lat_mask.min().item()) == 0.0
    print("[PASS] Test (c): HR box mask rasterization and 64x64 max-pool downsampling")

    # (d) Weighted L1 equivalence when alpha_text=1 / empty mask, and amplification with alpha_text>1
    pred = torch.randn(1, 4, 64, 64, requires_grad=True)
    target = torch.randn(1, 4, 64, 64)
    unweighted = (pred - target).abs().mean()
    step_lo = compute_lora_step_losses(pred, target, lat_mask, alpha_text=1.0)
    assert abs(step_lo.rec_loss - float(unweighted.item())) < 1e-6
    step_hi = compute_lora_step_losses(pred, target, lat_mask, alpha_text=5.0)
    assert step_hi.rec_loss > step_lo.rec_loss
    print("[PASS] Test (d): Text-weighted reconstruction loss matches base L1 at alpha=1 and amplifies text boxes")

    # (e) Exact gradient equivalence: two-pass chain-rule backward vs joint single-graph backward
    import copy

    net_joint = _MockControlledNet(channels=4)
    fresh_base = copy.deepcopy(net_joint)
    fresh_base_out = fresh_base(x_lat).detach().clone()
    inject_lora_adapters(net_joint, rank=2, alpha=4.0)
    for mod in net_joint.modules():
        if isinstance(mod, LoRALinear):
            nn.init.normal_(mod.lora_A, std=0.1)
            nn.init.normal_(mod.lora_B, std=0.1)

    net_twopass = copy.deepcopy(net_joint)
    noise_a = torch.randn(1, 4, 8, 8)
    noise_b = torch.randn(1, 4, 8, 8)
    target_lat = torch.randn(1, 4, 8, 8)
    small_mask = downsample_mask_to_latent(hr_mask, latent_h=8, latent_w=8)

    p_a_j = net_joint(x_lat + noise_a)
    p_b_j = net_joint(x_lat + noise_b)
    w_rec = _build_text_weight_map(small_mask, 4.0)
    w_cons = _build_text_weight_map(small_mask, 3.0)
    joint_loss = (
        0.5 * (w_rec * (p_a_j - target_lat).abs()).mean()
        + 0.5 * (w_rec * (p_b_j - target_lat).abs()).mean()
        + 0.5 * (w_cons * (p_a_j - p_b_j).abs()).mean()
    )
    joint_loss.backward()

    stats = run_two_noise_step(
        lambda eps: net_twopass(x_lat + eps),
        target_lat,
        small_mask,
        noise_a,
        noise_b,
        alpha_text=4.0,
        lambda_cons=0.5,
        alpha_cons_text=3.0,
    )
    assert abs(stats["total_loss"] - float(joint_loss.item())) < 1e-6

    for p_j, p_t in zip(get_lora_parameters(net_joint), get_lora_parameters(net_twopass)):
        assert p_j.grad is not None and p_t.grad is not None
        grad_diff = float((p_j.grad - p_t.grad).abs().max().item())
        assert grad_diff < 1e-5, f"Test (e) gradient mismatch: {grad_diff}"
    print("[PASS] Test (e): Two-pass chain-rule backward matches joint single-graph gradients (< 1e-5)")

    # (f) Checkpoint save/load roundtrip restores identical non-zero LoRA predictions
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "lora_weights_final.pt"
        save_lora_state_dict(net_twopass, ckpt_path, meta={"rank": 2, "alpha": 4.0})
        ref_out = net_twopass(x_lat).detach().clone()
        assert float((ref_out - fresh_base_out).abs().max().item()) > 1e-3

        loaded = load_lora_state_dict(fresh_base, ckpt_path)
        assert sorted(loaded) == sorted(replaced)
        roundtrip_diff = float((fresh_base(x_lat) - ref_out).abs().max().item())
        assert roundtrip_diff < 1e-6, f"Test (f) roundtrip diff: {roundtrip_diff}"
    print("[PASS] Test (f): LoRA checkpoint save/load roundtrip preserves exact model outputs")

    # (g) Timestep gating (cons_active=False vs True) & 4-minibatch FP32 gradient equivalence
    net_twopass.zero_grad(set_to_none=True)
    gated_off = run_two_noise_step(
        lambda eps: net_twopass(x_lat + eps),
        target_lat,
        small_mask,
        noise_a,
        noise_b,
        alpha_text=4.0,
        lambda_cons=0.5,
        alpha_cons_text=3.0,
        cons_active=False,
    )
    assert gated_off["cons_active"] == 0.0 and gated_off["cons_loss"] == 0.0
    assert abs(gated_off["total_loss"] - gated_off["rec_loss"]) < 1e-7

    fp32_audit = verify_fp32_twopass_vs_joint_gradients(grad_accum=4, lambda_cons=0.2, seed=231)
    assert fp32_audit["max_abs_grad_diff"] < 1e-5, f"Test (g) max_abs_grad_diff={fp32_audit['max_abs_grad_diff']}"
    assert fp32_audit["rel_l2_grad_diff"] < 1e-6, f"Test (g) rel_l2_grad_diff={fp32_audit['rel_l2_grad_diff']}"
    eta_audit = audit_dossr_eta_and_timesteps()
    assert eta_audit["t1_trunc_12"] == 500
    assert eta_audit["inference_unet_eval_timesteps"] == [499, 399, 299, 200, 100]
    assert eta_audit["one_minus_eta_at_t"]["500"] == 0.0
    assert eta_audit["one_minus_eta_at_t"]["499"] > 0.0
    print(
        "[PASS] Test (g): Timestep gating (cons_active=False), eta schedule audit, and "
        f"4-accum FP32 gradient check (max_diff={fp32_audit['max_abs_grad_diff']:.2e}, "
        f"rel_l2={fp32_audit['rel_l2_grad_diff']:.2e})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LoRA adapters and text-loss utilities for DoSSR.")
    parser.add_argument("--run-cpu-tests", action="store_true", help="Run CPU unit tests (a)-(g)")
    parser.add_argument(
        "--verify-fp32-gradients",
        action="store_true",
        help="Run explicit FP32 comparison between two-pass chain-rule backward and simultaneous joint backward",
    )
    parser.add_argument(
        "--audit-eta-schedule",
        action="store_true",
        help="Print DoSSR trunc_12 eta_t, 1-eta_t, alpha_bar_t, and SpacedSampler timestep math",
    )
    args = parser.parse_args(argv)
    if not (args.run_cpu_tests or args.verify_fp32_gradients or args.audit_eta_schedule):
        parser.print_help()
        return 0
    if args.run_cpu_tests:
        run_cpu_tests()
    if args.verify_fp32_gradients:
        print("\n=== FP32 Two-Pass vs Simultaneous Joint Two-Branch Gradient Check ===")
        for accum in (1, 4):
            res = verify_fp32_twopass_vs_joint_gradients(grad_accum=accum, lambda_cons=0.2, seed=231)
            print(json.dumps(res, indent=2))
    if args.audit_eta_schedule:
        print("\n=== DoSSR trunc_12 & SpacedSampler Schedule Audit ===")
        print(json.dumps(audit_dossr_eta_and_timesteps(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
