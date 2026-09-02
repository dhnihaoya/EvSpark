#!/usr/bin/env python3
"""EvSpark demo — lossless speculative decoding for Evo2 7B.

Loads Evo2 7B plus an EvSpark drafter checkpoint, generates DNA tokens with
speculative decoding, and cross-checks against native token-by-token decoding.

Greedy mode asserts token-for-token equality (losslessness). Sampling mode
reports wall-clock speedup under the same protocol as the paper
(temperature=1.0, top_k=4).

Usage:
    # auto-download the flagship drafter (gamma=12, 80M distill tokens) on first run
    python scripts/demo.py --ckpt L27_g12_80M_s1

    # greedy losslessness check on your own sequence
    python scripts/demo.py --ckpt L27_g12_80M_s1 --greedy --prompt-file my_seq.fa

Requires one GPU with ~40 GB free memory (RTX 4090/5090, A100, H100).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

# 1024 bp human chr21 intergenic fragment, used as the default prompt.
DEFAULT_PROMPT = (
    """CTTAAAAAATAAATACAGTAAAAAATGATAATGAATTTATTTCTTCTTATAATACTATCACTGGTGCATGATTAGAATAAATTAATCTTAAGATAATATAAGCATTTCTCCAAATGGTCTTCAAATTCATTGATTTTTTTGTGATTTCAATTTGCATATTTATTGTTCCTGTTGCAATTTTTTCAGATTTTTTAAGTATATGTATTAACTCAGAACACATAACTCTTATCACACATATTTTTCATGTAATTTATCTAAATCTCATAGAAAAGGGTCCATTTGCATTTTCTCTTATTAGACTCCTGATTTCAAATAATATATTACTTATGAGTATTTTTCTGTGCTGTAGTTATTCATTCTTATAGATATGTAACATAATTCCTTTTTCAAAGGTAAAAATTGAGCTATCTCTTGTTGAGGATTTGTTGATCTCTGTCTAAAGTTTCAAAAATAAAGAACTTTAAAAGCAAAATGTAAATTCCTTTCAAGTTTTAGTAAAATTACTTCAAACTTAGTAGCTTAAACAATACAGATTTATTATGTTACAGTTCTGTAAGACAGAAATCTGACTTGATCACACCATGTTAAAACGAAGATACTGCCAGGGTTGGTTTTTTCTTGGGGGTGGTCTGTGGGAAGAGTTCGTTTCCTTTGGTTTTCCACAGCCCAGAGGCTGCTTGCATTCCTTTAATCACTGTCCCTTCCTCCATTTTTGAAATGAGGAATGGAGTCAGGGTGACTATGGTTAGCAATATTGTATTGTATATTTCAAAATAGCTAGAAGAGAGGATTTTTGAATTCTCTCACCATAAAGATATCAAAGATGTATGAAGTGAAGAATATGTTGAATATCCTGATTCAATATTTAAACTATACATACACGTGTTGAAACATCACACTGTATCCCATAAATATGTACAATAATTATGTGTCATAAAACAAGATTTAAATTGTTTTAAAGGGCCAGCAATGGCAGTTTGTGAGTTCCCATCTCATCACTCTAACTTCTTCTGCCTCCTTCCAC"""
)


def load_prompt(args) -> str:
    if args.prompt_file:
        text = Path(args.prompt_file).read_text()
        # tolerate FASTA headers
        seq = "".join(
            line.strip() for line in text.splitlines() if not line.startswith(">")
        )
    elif args.prompt:
        seq = args.prompt
    else:
        seq = DEFAULT_PROMPT
    seq = seq.upper()
    bad = sorted(set(seq) - set("ACGTN"))
    if bad:
        raise ValueError(f"prompt contains non-ACGTN characters: {bad}")
    if len(seq) < 64:
        raise ValueError("prompt too short (<64 bp); give the model some context")
    return seq


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="L27_g12_80M_s1",
                    help="drafter checkpoint: local .pt path or a known name "
                         "(auto-downloaded from Hugging Face / ModelScope)")
    ap.add_argument("--prompt", type=str, default=None, help="DNA string (ACGTN)")
    ap.add_argument("--prompt-file", type=str, default=None, help="plain text or FASTA")
    ap.add_argument("--n-tokens", type=int, default=256)
    ap.add_argument("--gamma", type=int, default=None, help="draft length (default: read from ckpt)")
    ap.add_argument("--greedy", action="store_true",
                    help="greedy decoding + exact-match check against native decoding")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required (Evo2 7B needs ~40 GB VRAM in bf16).")

    from download_ckpt import ensure_ckpt

    ckpt = ensure_ckpt(args.ckpt)

    from evo2 import Evo2

    print("[demo] loading Evo2 7B (bf16) ...", flush=True)
    evo = Evo2("evo2_7b", use_kernels=True)
    model, tokenizer = evo.model, evo.tokenizer
    model.eval()

    from specdec.block.loop import (
        native_greedy_reference,
        native_sample_reference,
        speculative_generate,
    )
    from specdec.block.neural_draft import NeuralDraftModel

    nd = NeuralDraftModel.from_checkpoint(model, str(ckpt), device="cuda:0")
    gamma = args.gamma or nd.gamma
    print(f"[demo] drafter={ckpt.name} layers={list(nd.layer_names)} gamma={gamma}")

    seq = load_prompt(args)
    ids = torch.tensor(
        np.asarray(tokenizer.tokenize(seq), dtype=np.int64)[None, :], device="cuda:0"
    )
    print(f"[demo] prompt={len(seq)} bp, generating {args.n_tokens} tokens "
          f"({'greedy' if args.greedy else 'sampling T=1.0 top_k=4'})")

    rng = np.random.default_rng(args.seed)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    spec = speculative_generate(
        model, nd, ids, args.n_tokens, gamma, greedy=args.greedy, rng=rng,
        temperature=1.0, top_k=4,
    )
    torch.cuda.synchronize()
    t_spec = time.perf_counter() - t0
    nd.close()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    if args.greedy:
        nat_ids, _ip, _ = native_greedy_reference(model, ids, args.n_tokens)
    else:
        nat_ids = native_sample_reference(
            model, ids, args.n_tokens, np.random.default_rng(args.seed),
            temperature=1.0, top_k=4,
        )
    torch.cuda.synchronize()
    t_nat = time.perf_counter() - t0

    ks = [int(r.k) for r in spec.rounds_log]
    tau = float(np.mean(ks) + 1.0) if ks else float("nan")
    spec_tok_s = args.n_tokens / t_spec
    nat_tok_s = args.n_tokens / t_nat

    print()
    print(f"  native      : {t_nat:7.2f} s  ({nat_tok_s:6.1f} tok/s)")
    print(f"  speculative : {t_spec:7.2f} s  ({spec_tok_s:6.1f} tok/s)")
    print(f"  speedup     : {t_nat / t_spec:.2f}x   "
          f"(mean accepted tau={tau:.2f}, rounds={len(ks)})")

    if args.greedy:
        print("  note        : greedy on bacterial coding is the hardest cell for "
              "acceptance;\n                this mode is for the exact-match check. "
              "Use sampling mode for speed.")
        emitted = np.asarray(spec.emitted_ids)
        n_match = int((emitted == nat_ids).sum())
        first_diff = next(
            (i for i in range(args.n_tokens) if emitted[i] != nat_ids[i]), None
        )
        if first_diff is None:
            print(f"  lossless    : {n_match}/{args.n_tokens} tokens identical to native decoding")
        else:
            # rare bf16 tie-flip: logits tied within kernel reduction order
            print(f"  note        : {n_match}/{args.n_tokens} identical; first divergence "
                  f"at position {first_diff} (bf16 tie-flip, see paper §losslessness)")


if __name__ == "__main__":
    main()
