#!/usr/bin/env python3
"""Compare DoSSR's einsum CrossAttention and bmm VAE AttnBlock with SDPA on tiny FP16 tensors."""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.attention_compat import (  # noqa: E402
    bmm_vae_attention_reference,
    einsum_attention_reference,
    patch_vae_attn_block,
    sdpa_attention,
    sdpa_vae_attention,
)
from src.env_info import DEFAULT_LOG_DIR, configure_logging, log_run_info  # noqa: E402


class _ReferenceAttnBlock(torch.nn.Module):
    """Minimal replica of upstream ldm.modules.diffusionmodules.model.AttnBlock."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.norm = torch.nn.GroupNorm(num_groups=8, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        h_ = bmm_vae_attention_reference(q, k, v)
        h_ = self.proj_out(h_)
        return x + h_


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    args = parser.parse_args()

    logger, log_path = configure_logging("check_attention_sdpa", args.log_dir)
    log_run_info(logger, args, seed=args.seed)
    logger.info("Log file: %s", log_path)

    try:
        torch.manual_seed(args.seed)
        device = torch.device("cpu")
        batch, heads, query_len, key_len, head_dim = 2, 4, 16, 19, 8
        q = torch.randn(batch, heads, query_len, head_dim, dtype=torch.float16, device=device)
        k = torch.randn(batch, heads, key_len, head_dim, dtype=torch.float16, device=device)
        v = torch.randn(batch, heads, key_len, head_dim, dtype=torch.float16, device=device)
        mask = torch.ones(batch, 1, 1, key_len, dtype=torch.bool, device=device)
        mask[:, :, :, -3:] = False
        scale = head_dim ** -0.5

        reference = einsum_attention_reference(q, k, v, scale, mask)
        actual = sdpa_attention(q, k, v, scale, mask)
        max_abs_error = (reference.float() - actual.float()).abs().max().item()
        torch.testing.assert_close(actual, reference, rtol=1e-2, atol=1e-2)
        logger.info(
            "SDPA FP16 comparison passed | shape=%s | max_abs_error=%.6f | rtol=0.01 | atol=0.01",
            tuple(actual.shape),
            max_abs_error,
        )

        vae_batch, channels, height, width = 2, 16, 6, 5
        vq = torch.randn(vae_batch, channels, height, width, dtype=torch.float16, device=device)
        vk = torch.randn(vae_batch, channels, height, width, dtype=torch.float16, device=device)
        vv = torch.randn(vae_batch, channels, height, width, dtype=torch.float16, device=device)
        vae_ref = bmm_vae_attention_reference(vq, vk, vv)
        vae_act = sdpa_vae_attention(vq, vk, vv)
        vae_max_abs_error = (vae_ref.float() - vae_act.float()).abs().max().item()
        torch.testing.assert_close(vae_act, vae_ref, rtol=1e-2, atol=1e-2)
        logger.info(
            "VAE AttnBlock SDPA kernel FP16 comparison passed | shape=%s | max_abs_error=%.6f | rtol=0.01 | atol=0.01",
            tuple(vae_act.shape),
            vae_max_abs_error,
        )

        x = torch.randn(vae_batch, channels, height, width, dtype=torch.float16, device=device)
        block = _ReferenceAttnBlock(channels).to(device=device, dtype=torch.float16).eval()
        with torch.no_grad():
            block_ref = block(x)
            dummy_module = types.SimpleNamespace(AttnBlock=_ReferenceAttnBlock)
            patched = patch_vae_attn_block(dummy_module)
            if not patched:
                raise RuntimeError("patch_vae_attn_block returned False")
            block_act = block(x)
        block_max_abs_error = (block_ref.float() - block_act.float()).abs().max().item()
        torch.testing.assert_close(block_act, block_ref, rtol=1e-2, atol=1e-2)
        logger.info(
            "VAE AttnBlock patched forward FP16 comparison passed | shape=%s | max_abs_error=%.6f | rtol=0.01 | atol=0.01",
            tuple(block_act.shape),
            block_max_abs_error,
        )
        return 0
    except Exception as error:
        logger.exception("SDPA comparison failed | %s", type(error).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

