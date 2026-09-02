#!/usr/bin/env python3
"""Download EvSpark drafter checkpoints from Hugging Face or ModelScope.

Checkpoints are ~160 MB each and contain all drafter weights (frozen embedding,
Markov head, γ-parallel confidence head) plus metadata (scheme / d_model / γ);
``specdec.block.neural_draft.NeuralDraftModel.from_checkpoint`` loads them
directly.

Usage:
    python scripts/download_ckpt.py L27_g12_80M_s1                 # HF first, ModelScope fallback
    python scripts/download_ckpt.py L27_g12_80M_s1 --source modelscope
    python scripts/download_ckpt.py --list
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Published checkpoints (see README §Checkpoints for what each one is).
KNOWN_CKPTS = [
    "L27_g12_80M_s1",      # flagship: gamma=12, 80M distill tokens (3.16x suite mean)
    "L27_g12_80M_s2",      # flagship, second seed
    "L27_g12_30M_s1",      # budget cell: ~1 GPU-hour of training, 3.03x
    "L27_g12_30M_s2",
    "L27_final15_150M_s1", # gamma=7 main-table model (2.89x / 2.87x)
    "L27_final15_150M_s2",
]

HF_REPO = os.environ.get("EVSPARK_HF_REPO", "dinghhhhhhhhhhhhhhh/EvSpark")
MS_REPO = os.environ.get("EVSPARK_MS_REPO", "dinghao1120/EvSpark")

DEFAULT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


def _from_hf(name: str, dest: Path) -> Path:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(HF_REPO, f"{name}.pt", local_dir=str(dest))
    return Path(path)


def _from_modelscope(name: str, dest: Path) -> Path:
    from modelscope import snapshot_download

    snapshot_download(MS_REPO, allow_patterns=[f"{name}.pt"], local_dir=str(dest))
    return dest / f"{name}.pt"


def ensure_ckpt(name_or_path: str, dest: Path | None = None,
                source: str = "auto") -> Path:
    """Resolve a checkpoint: pass through local paths, download known names."""
    p = Path(name_or_path)
    if p.is_file():
        return p
    name = p.name.removesuffix(".pt")
    dest = dest or DEFAULT_DIR
    local = dest / f"{name}.pt"
    if local.is_file():
        return local
    if name not in KNOWN_CKPTS:
        raise SystemExit(
            f"unknown checkpoint {name!r}; known: {', '.join(KNOWN_CKPTS)} "
            f"(or pass a local .pt path)"
        )
    dest.mkdir(parents=True, exist_ok=True)
    sources = {"hf": _from_hf, "modelscope": _from_modelscope}
    order = [source] if source != "auto" else ["hf", "modelscope"]
    errors = []
    for src in order:
        try:
            print(f"[ckpt] downloading {name}.pt from {src} ({HF_REPO if src == 'hf' else MS_REPO}) ...",
                  flush=True)
            got = sources[src](name, dest)
            print(f"[ckpt] -> {got}")
            return got
        except Exception as e:  # noqa: BLE001 - fall through to next mirror
            errors.append(f"{src}: {e}")
            print(f"[ckpt] {src} failed ({type(e).__name__}), trying next source ...", flush=True)
    raise SystemExit("all download sources failed:\n" + "\n".join(errors))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", help="checkpoint names (see --list)")
    ap.add_argument("--source", choices=["auto", "hf", "modelscope"], default="auto")
    ap.add_argument("--dest", type=Path, default=None)
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
