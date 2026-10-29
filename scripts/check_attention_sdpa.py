#!/usr/bin/env python3
"""Compare DoSSR's einsum attention with the SDPA replacement on tiny FP16 tensors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.attention_compat import einsum_attention_reference, sdpa_attention  # noqa: E402
from src.env_info import DEFAULT_LOG_DIR, configure_logging, log_run_info  # noqa: E402


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
        return 0
    except Exception as error:
        logger.exception("SDPA comparison failed | %s", type(error).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
