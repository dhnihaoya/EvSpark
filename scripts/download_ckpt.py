#!/usr/bin/env python3
"""Download EvSpark drafter checkpoints from Hugging Face or ModelScope.

Thin CLI over ``evspark.checkpoints`` — see that module for details.
Checkpoints land in ``$EVSPARK_CKPT_DIR`` or ``~/.cache/evspark/checkpoints``.

Usage:
    python scripts/download_ckpt.py L27_g12_150M_s1                 # HF first, ModelScope fallback
    python scripts/download_ckpt.py L27_g12_150M_s1 --source modelscope
    python scripts/download_ckpt.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evspark.checkpoints import KNOWN_CKPTS, ensure_ckpt  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="checkpoint names (see --list)")
    ap.add_argument("--source", choices=["auto", "hf", "modelscope"], default="auto")
    ap.add_argument("--dest", type=Path, default=None,
                    help="default: $EVSPARK_CKPT_DIR or ~/.cache/evspark/checkpoints")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list or not args.names:
        print("known checkpoints:")
        for n in KNOWN_CKPTS:
            print(f"  {n}")
        if not args.names:
            return
    for n in args.names:
        ensure_ckpt(n, dest=args.dest, source=args.source)


if __name__ == "__main__":
    main()
