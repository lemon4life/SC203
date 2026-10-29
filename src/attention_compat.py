"""Runtime PyTorch SDPA compatibility for DoSSR's vanilla CrossAttention."""

from __future__ import annotations

from typing import Any


def einsum_attention_reference(q, k, v, scale: float, mask=None):
    """Original full-score-table attention, retained for fallback and comparison."""
    import torch

    scores = torch.einsum("bhid,bhjd->bhij", q, k) * scale
    if mask is not None:
        scores = scores.masked_fill(~mask, -torch.finfo(scores.dtype).max)
    weights = scores.softmax(dim=-1)
    return torch.einsum("bhij,bhjd->bhid", weights, v)


def sdpa_attention(q, k, v, scale: float, mask=None):
    """Apply PyTorch SDPA with explicit scale and the same True-means-valid mask."""
    import torch.nn.functional as functional

    return functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )


def patch_cross_attention(attention_module: Any) -> bool:
    """Replace vanilla CrossAttention.forward; return whether SDPA is available."""
    import torch
    import torch.nn.functional as functional

    if not hasattr(functional, "scaled_dot_product_attention"):
        # Keep using the original upstream einsum implementation as fallback.
        return False

    original_forward = attention_module.CrossAttention.forward

    def forward(self, x, context=None, mask=None):
        batch, query_len, _ = x.shape
        context = x if context is None else context
        heads = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)
        head_dim = q.shape[-1] // heads
        q = q.reshape(batch, query_len, heads, head_dim).transpose(1, 2)
        k = k.reshape(batch, k.shape[1], heads, head_dim).transpose(1, 2)
        v = v.reshape(batch, v.shape[1], heads, head_dim).transpose(1, 2)

        if mask is not None:
            # Upstream treats True as a valid key and expands this over heads/queries.
            mask = mask.reshape(batch, -1)[:, None, None, :].to(dtype=torch.bool)

        try:
            out = sdpa_attention(q, k, v, self.scale, mask)
        except (NotImplementedError, RuntimeError) as error:
            message = str(error).lower()
            unsupported_kernel = isinstance(error, NotImplementedError) or any(
                marker in message
                for marker in (
                    "no available kernel",
                    "not implemented for",
                    "unsupported device",
                )
            )
            if not unsupported_kernel:
                raise
            # Fall back to DoSSR's original einsum implementation for unsupported SDPA kernels.
            return original_forward(self, x, context, mask)
        out = out.transpose(1, 2).reshape(batch, query_len, heads * head_dim)
        return self.to_out(out)

    attention_module.CrossAttention.forward = forward
    return True
