#!/usr/bin/env python3
"""Dump real attention Q/K/V activations from a Hugging Face model for SLHA v2
Phase-0 offline validation.

It captures the per-token **full-width** query/key/value vectors (dim = d_model,
e.g. 768 for GPT-2) of one attention layer over some text, and writes them as
`k.bin` / `q.bin` / `v.bin` in the tiny format the Rust harness reads:

    [u32 magic = 0x534C4841 ("SLHA")][u32 rows][u32 cols][f32 rows*cols row-major, LE]

Then feed them to the harness:

    cargo run --release --example offline_validation -- --dump <OUT_DIR>

Why full width (d_model), not per-head (d_head)? SLHA compresses a key of
dim > 128 into a 128-dim latent + residual. Per-head keys (64/128 dims) are too
narrow for the residual to mean anything; the honest first experiment compresses
the full key projection. Multi-head structure is a Phase-1 follow-up.

Requirements for the real dump (on your machine — NOT needed to read this file):
    pip install torch transformers

Usage (real activations):
    python scripts/dump_activations.py --model gpt2 --layer 0 --out /tmp/act \\
        --text "Some representative text..."      # or --file corpus.txt

Usage (torch-free plumbing check — validates the dump→harness contract with NO
model, numpy or torch; use different --seed values for disjoint train/test sets):
    python scripts/dump_activations.py --synthetic --out /tmp/train --seed 1
    python scripts/dump_activations.py --synthetic --out /tmp/test  --seed 2
"""
import argparse
import struct
import sys

MAGIC = 0x534C4841  # "SLHA"


def write_bin(path, mat):
    """Write a 2-D float matrix in the harness format. `mat` may be a numpy
    ndarray (real dump) or a plain list-of-lists (synthetic, no numpy):

        [u32 magic][u32 rows][u32 cols][f32 rows*cols row-major, LE]
    """
    rows = len(mat)
    cols = len(mat[0]) if rows else 0
    with open(path, "wb") as f:
        f.write(struct.pack("<III", MAGIC, rows, cols))
        try:
            import numpy as np

            f.write(np.ascontiguousarray(np.asarray(mat, dtype="<f4")).tobytes())
        except ImportError:
            packer = struct.Struct(f"<{cols}f")
            for row in mat:
                f.write(packer.pack(*row))
    print(f"  wrote {path}  ({rows} tokens × {cols} dims)")


def synthetic_qkv(rows, dim, seed, rank=24, noise=0.1, n_outliers=4):
    """Torch-free q/k/v with a low-rank base (so keys/queries correlate) plus
    per-dim noise and a few heavy-tailed outlier dims — enough structure to
    exercise the low-rank latent + residual pipeline. NOT a source of real
    numbers: this validates the dump→harness plumbing, not model quality.

    The subspace and outlier channels come from a FIXED structural seed (they
    model one model's key geometry), so different `seed` values give disjoint
    *corpora* over the SAME subspace — the honest analog of dumping train vs
    test text from one model, where a held-out projection should generalise.
    Returns three [rows][dim] lists."""
    import math
    import random

    struct_rng = random.Random(0xB0A5E)  # fixed: shared subspace across corpora
    basis = [[struct_rng.gauss(0.0, 1.0) for _ in range(dim)] for _ in range(rank)]
    outlier_dims = [struct_rng.randrange(dim) for _ in range(n_outliers)]
    inv_sqrt_r = 1.0 / math.sqrt(rank)

    rng = random.Random(seed)  # corpus-specific: tokens + noise vary with seed

    def make(n):
        out = []
        for _ in range(n):
            code = [rng.gauss(0.0, 1.0) for _ in range(rank)]
            row = [0.0] * dim
            for c, brow in zip(code, basis):
                for j in range(dim):
                    row[j] += c * brow[j]
            for j in range(dim):
                row[j] = inv_sqrt_r * row[j] + noise * rng.gauss(0.0, 1.0)
            for od in outlier_dims:
                row[od] += rng.gauss(0.0, 3.0)
            out.append(row)
        return out

    return make(rows), make(rows), make(rows)


def main():
    ap = argparse.ArgumentParser(description="Dump Q/K/V activations for SLHA Phase 0.")
    ap.add_argument("--model", default="gpt2", help="HF model id (default: gpt2)")
    ap.add_argument("--layer", type=int, default=0, help="attention layer index")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--text", default=None, help="inline text to run through the model")
    ap.add_argument("--file", default=None, help="text file to run through the model")
    ap.add_argument("--max-tokens", type=int, default=1024, help="cap the sequence length")
    ap.add_argument(
        "--synthetic",
        action="store_true",
        help="generate torch-free synthetic q/k/v (validate the plumbing, no model)",
    )
    ap.add_argument("--rows", type=int, default=256, help="synthetic: token count")
    ap.add_argument("--dim", type=int, default=256, help="synthetic: model width (must be > 128)")
    ap.add_argument("--seed", type=int, default=1, help="synthetic: RNG seed (vary for train vs test)")
    args = ap.parse_args()

    import os

    os.makedirs(args.out, exist_ok=True)

    if args.synthetic:
        if args.dim <= 128:
            sys.exit(f"--dim must be > 128 (SLHA needs d > 128); got {args.dim}")
        print(
            f"SYNTHETIC dump (no model) — rows={args.rows} dim={args.dim} "
            f"seed={args.seed} out={args.out}"
        )
        q, k, v = synthetic_qkv(args.rows, args.dim, args.seed)
        for name, mat in (("q", q), ("k", k), ("v", v)):
            write_bin(os.path.join(args.out, f"{name}.bin"), mat)
        print(f"\n  key dim d = {args.dim}  (OK, > 128)")
        print(f"  next:  cargo run --release --example offline_validation -- --dump {args.out}")
        return

    try:
        import numpy as np  # noqa: F401
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as e:
        sys.exit(f"missing dependency: {e}\n  pip install torch transformers numpy")

    text = args.text
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    if not text:
        text = (
            "The transformer key/value cache grows with every generated token and "
            "quickly saturates memory; compressing it while preserving attention is "
            "the whole point of SLHA v2. " * 8
        )

    print(f"model={args.model} layer={args.layer} out={args.out}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model.eval()

    ids = tok(text, return_tensors="pt", truncation=True, max_length=args.max_tokens)

    # Find the attention module for the requested layer. Try common layouts.
    attn = None
    for path in (
        lambda m: m.transformer.h[args.layer].attn,          # GPT-2 family
        lambda m: m.model.layers[args.layer].self_attn,       # Llama/Mistral family
        lambda m: m.encoder.layer[args.layer].attention.self, # BERT family
        lambda m: m.h[args.layer].attn,
    ):
        try:
            attn = path(model)
            break
        except (AttributeError, IndexError):
            continue
    if attn is None:
        sys.exit("could not locate the attention module — adjust the hook for this model")

    captured = {}

    def hook(_module, inputs, output):
        # GPT-2: c_attn output is [batch, seq, 3*d_model] → split q,k,v.
        # Others: fall back to capturing the module input (hidden states) — still
        # a full-width d_model representation suitable for the offline proxy.
        out = output[0] if isinstance(output, tuple) else output
        if out.dim() == 3 and out.shape[-1] % 3 == 0 and hasattr(attn, "c_attn"):
            q, k, v = out.split(out.shape[-1] // 3, dim=2)
            captured["q"], captured["k"], captured["v"] = q, k, v
        else:
            h = inputs[0]
            captured["q"] = captured["k"] = captured["v"] = h

    # Prefer hooking c_attn (GPT-2) for a clean q/k/v split; else hook the module.
    target = getattr(attn, "c_attn", attn)
    handle = target.register_forward_hook(hook)
    with torch.no_grad():
        model(**ids)
    handle.remove()

    if "k" not in captured:
        sys.exit("hook captured nothing — adjust for this architecture")

    for name in ("q", "k", "v"):
        mat = captured[name].squeeze(0).detach().cpu().numpy()  # [seq, d_model]
        write_bin(os.path.join(args.out, f"{name}.bin"), mat)

    d = captured["k"].shape[-1]
    print(f"\n  key dim d = {d}  ({'OK' if d > 128 else 'TOO NARROW — SLHA needs d > 128'})")
    print(f"  next:  cargo run --release --example offline_validation -- --dump {args.out}")


if __name__ == "__main__":
    main()
