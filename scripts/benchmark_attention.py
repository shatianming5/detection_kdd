#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch


def _maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _bench(fn, *, warmup: int, iters: int, device: torch.device) -> float:
    for _ in range(max(int(warmup), 0)):
        fn()
    _maybe_sync(device)
    t0 = time.perf_counter()
    for _ in range(max(int(iters), 1)):
        fn()
    _maybe_sync(device)
    t1 = time.perf_counter()
    return (t1 - t0) / float(max(int(iters), 1))


def main() -> None:
    # Ensure `diffusiondet` is importable when running from repo root.
    repo_root = Path(__file__).resolve().parents[1]
    diffdet_dir = repo_root / "baselines" / "DiffusionDet"
    if diffdet_dir.is_dir():
        sys.path.insert(0, str(diffdet_dir))

    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", choices=["torch", "sdpa"], default="torch")
    ap.add_argument("--sdpa-backend", choices=["auto", "flash", "mem_efficient", "math"], default="auto")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seqlen", type=int, default=500)
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--with-mask", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    torch.manual_seed(0)

    l = int(args.seqlen)
    b = int(args.batch)
    e = int(args.embed_dim)
    h = int(args.heads)
    if e % h != 0:
        raise SystemExit("--embed-dim must be divisible by --heads")

    q = torch.randn((l, b, e), device=device, dtype=dtype)
    attn_mask = None
    if args.with_mask:
        # Additive bias mask (B*H, L, L) matches this repo's geometry bias format.
        attn_mask = torch.zeros((b * h, l, l), device=device, dtype=dtype)

    if args.impl == "torch":
        attn = torch.nn.MultiheadAttention(e, h, dropout=0.0).to(device=device, dtype=dtype)

        def run():
            attn(q, q, q, attn_mask=attn_mask, need_weights=False)[0]

    else:
        from diffusiondet.util.sdpa_attention import SDPAMultiheadAttention

        attn = SDPAMultiheadAttention(e, h, dropout=0.0, backend=args.sdpa_backend).to(device=device, dtype=dtype)

        def run():
            attn(q, q, q, attn_mask=attn_mask, need_weights=False)[0]

    try:
        sec = _bench(run, warmup=args.warmup, iters=args.iters, device=device)
    except RuntimeError as e:
        if "No available kernel" in str(e):
            raise SystemExit(
                "SDPA backend has no available kernel for the current settings. "
                "Try `--sdpa-backend auto`, or remove `--with-mask`."
            ) from e
        raise
    print(
        f"impl={args.impl} backend={args.sdpa_backend} device={device.type} dtype={dtype} "
        f"B={b} L={l} E={e} H={h} mask={bool(attn_mask is not None)}: {sec * 1e3:.3f} ms/iter"
    )

    if device.type == "cuda" and hasattr(torch.backends.cuda, "can_use_flash_attention"):
        # Quick capability probe (not a guarantee of actual kernel selection in all cases).
        hd = e // h
        q4 = torch.randn((b, h, l, hd), device=device, dtype=dtype)
        params = torch.backends.cuda.SDPAParams(q4, q4, q4, None, 0.0, False, False)
        can_flash = torch.backends.cuda.can_use_flash_attention(params, debug=False)
        can_eff = torch.backends.cuda.can_use_efficient_attention(params, debug=False)
        can_math = None
        if hasattr(torch.backends.cuda, "can_use_math_attention"):
            can_math = torch.backends.cuda.can_use_math_attention(params, debug=False)
        print(f"sdpa_can_use: flash={can_flash} efficient={can_eff} math={can_math}")


if __name__ == "__main__":
    main()
