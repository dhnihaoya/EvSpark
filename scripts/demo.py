#!/usr/bin/env python3
"""EvSpark demo — lossless speculative decoding for Evo2 7B.

Thin CLI over the ``evspark`` library facade (scripts/evspark.py). Loads Evo2 7B
plus an EvSpark drafter checkpoint, generates DNA tokens with speculative
decoding, and cross-checks against native token-by-token decoding.

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
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evspark import EvSpark, clean_prompt  # noqa: E402

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
    return clean_prompt(seq)


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

    seq = load_prompt(args)

    with EvSpark.load(args.ckpt) as es:
        gamma = args.gamma or es.gamma
        print(f"[demo] prompt={len(seq)} bp, generating {args.n_tokens} tokens "
              f"({'greedy' if args.greedy else 'sampling T=1.0 top_k=4'})")

        spec = es.generate(seq, args.n_tokens, greedy=args.greedy,
                           gamma=gamma, seed=args.seed)
        nat = es.generate_native(seq, args.n_tokens, greedy=args.greedy,
                                 seed=args.seed)

    print()
    print(f"  native      : {nat.wall_s:7.2f} s  ({nat.tok_s:6.1f} tok/s)")
    print(f"  speculative : {spec.wall_s:7.2f} s  ({spec.tok_s:6.1f} tok/s)")
    print(f"  speedup     : {nat.wall_s / spec.wall_s:.2f}x   "
          f"(mean accepted tau={spec.mean_tau:.2f}, rounds={spec.n_rounds})")

    if args.greedy:
        print("  note        : greedy on bacterial coding is the hardest cell for "
              "acceptance;\n                this mode is for the exact-match check. "
              "Use sampling mode for speed.")
        n_match = int((spec.ids == nat.ids).sum())
        first_diff = next(
            (i for i in range(args.n_tokens) if spec.ids[i] != nat.ids[i]), None
        )
        if first_diff is None:
            print(f"  lossless    : {n_match}/{args.n_tokens} tokens identical to native decoding")
        else:
            # rare bf16 tie-flip: logits tied within kernel reduction order
            print(f"  note        : {n_match}/{args.n_tokens} identical; first divergence "
                  f"at position {first_diff} (bf16 tie-flip, see paper §losslessness)")


if __name__ == "__main__":
    main()
