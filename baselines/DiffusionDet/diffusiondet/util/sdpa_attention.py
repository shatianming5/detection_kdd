from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


SdpaBackend = Literal["auto", "flash", "mem_efficient", "math"]


@contextmanager
def _sdpa_backend(backend: SdpaBackend):
    if backend == "auto":
        yield
        return

    # torch.backends.cuda.sdp_kernel is the legacy API; torch.nn.attention.sdpa_kernel is the newer one.
    ctx = None
    if hasattr(torch.nn, "attention") and hasattr(torch.nn.attention, "sdpa_kernel"):
        if backend == "flash":
            ctx = torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.FLASH_ATTENTION])
        elif backend == "mem_efficient":
            ctx = torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION])
        elif backend == "math":
            ctx = torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH])
    elif hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel"):
        if backend == "flash":
            ctx = torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False)
        elif backend == "mem_efficient":
            ctx = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True)
        elif backend == "math":
            ctx = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)

    if ctx is None:
        # Backend forcing is not supported; fall back to auto.
        ctx = nullcontext()

    with ctx:
        yield


class SDPAMultiheadAttention(nn.Module):
    """
    A small MultiheadAttention replacement backed by scaled_dot_product_attention (SDPA).

    - Supports additive float masks used in this repo (shape (B*H, L, S) or (L, S)).
    - Does NOT return attention weights; when need_weights=True, we raise to avoid silently slow paths.
      In this repo we fall back to torch.nn.MultiheadAttention whenever attention weights are required.
    """

    def __init__(self, embed_dim: int, num_heads: int, *, dropout: float = 0.0, backend: SdpaBackend = "auto"):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim must be divisible by num_heads, got {embed_dim} and {num_heads}")
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout = float(dropout)
        self.backend: SdpaBackend = backend

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ):
        if need_weights:
            raise NotImplementedError("SDPAMultiheadAttention does not return attention weights.")

        # torch.nn.MultiheadAttention uses (L, N, E) when batch_first=False (the default in this repo).
        if query.dim() != 3:
            raise ValueError(f"Expected query to be (L,N,E), got {tuple(query.shape)}")
        if key.shape != query.shape or value.shape != query.shape:
            raise ValueError("Only self-attention (query==key==value) is supported in this repo integration.")

        l, batch, _ = query.shape
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # (L, B, E) -> (B, H, L, D)
        q = q.permute(1, 0, 2).reshape(batch, l, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.permute(1, 0, 2).reshape(batch, l, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.permute(1, 0, 2).reshape(batch, l, self.num_heads, self.head_dim).transpose(1, 2)

        mask = None
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                mask = attn_mask.to(dtype=q.dtype)[None, None, :, :]
            elif attn_mask.dim() == 3:
                if attn_mask.shape[0] != batch * self.num_heads:
                    raise ValueError(
                        f"Expected attn_mask shape (B*H,L,S), got {tuple(attn_mask.shape)} with B={batch} H={self.num_heads}"
                    )
                mask = attn_mask.view(batch, self.num_heads, attn_mask.shape[1], attn_mask.shape[2]).to(dtype=q.dtype)
            elif attn_mask.dim() == 4:
                mask = attn_mask.to(dtype=q.dtype)
            else:
                raise ValueError(f"Unsupported attn_mask shape={tuple(attn_mask.shape)}")

        try:
            with _sdpa_backend(self.backend):
                out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=mask,
                    dropout_p=self.dropout if self.training else 0.0,
                )
        except RuntimeError as e:
            # When a backend is forced (e.g. "flash") but the kernel cannot handle the mask shape,
            # PyTorch raises "No available kernel". Fall back to auto selection to preserve correctness.
            if self.backend != "auto" and "No available kernel" in str(e):
                out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=mask,
                    dropout_p=self.dropout if self.training else 0.0,
                )
            else:
                raise

        # (B, H, L, D) -> (L, B, E)
        out = out.transpose(1, 2).reshape(batch, l, self.embed_dim).permute(1, 0, 2)
        out = self.out_proj(out)
        return out, None
