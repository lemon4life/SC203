#!/usr/bin/env python3
"""LoRA adapters, text-region masks, losses, and two-pass consistency step for DoSSR."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
    loss_backward_fn: Any = None,
    backward: bool = True,
) -> dict[str, float]:
    """Execute exact chain-rule two-pass gradient accumulation for two noise draws.

    Peak activation memory stays equal to one forward pass because Pass 1 frees its
    computation graph before Pass 2 builds its own, while gradients equal
    loss_scale * (0.5*L_rec(p_a) + 0.5*L_rec(p_b) + lambda_cons*L_cons(p_a, p_b)).
    """
    bw = loss_backward_fn if loss_backward_fn is not None else (lambda t: t.backward())

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
        if peer_detached is not None and lambda_cons > 0.0:
            w_cons = _build_text_weight_map(mask_lat.to(pred.dtype), alpha_cons_text)
            obj = obj + float(lambda_cons) * (w_cons * (pred - peer_detached).abs()).mean()
        return obj * float(loss_scale), rec_out

    pred_b_ref: torch.Tensor | None = None
    if lambda_cons > 0.0:
        with torch.no_grad():
            pred_b_ref = forward_fn(noise_b).detach()

    pred_a = forward_fn(noise_a)
    pred_a_ref = pred_a.detach()
    pass1_obj, rec_a = _pass_objective(pred_a, pred_b_ref)
    if backward:
        bw(pass1_obj)
    del pred_a, pass1_obj

    pred_b = forward_fn(noise_b)
    pass2_obj, rec_b = _pass_objective(pred_b, pred_a_ref if lambda_cons > 0.0 else None)
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
            if lambda_cons > 0.0
            else 0.0
        )
        rec_mean = 0.5 * (rec_a.rec_loss + rec_b.rec_loss)
        global_l1 = 0.5 * (rec_a.global_l1 + rec_b.global_l1)
        text_l1 = 0.5 * (rec_a.text_l1 + rec_b.text_l1)

    return {
        "total_loss": rec_mean + float(lambda_cons) * cons_val,
        "rec_loss": rec_mean,
        "global_l1": global_l1,
        "text_l1": text_l1,
        "cons_loss": cons_val,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LoRA adapters and text-loss utilities for DoSSR.")
    parser.add_argument("--run-cpu-tests", action="store_true", help="Run CPU unit tests (a)-(f)")
    args = parser.parse_args(argv)
    if args.run_cpu_tests:
        run_cpu_tests()
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
