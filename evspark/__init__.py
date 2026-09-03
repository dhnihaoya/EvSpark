"""EvSpark — lossless speculative decoding for Evo2 (StripedHyena2 DNA models).

Usage::

    from evspark import EvSpark

    with EvSpark.load("L27_g12_80M_s1") as es:   # auto-downloads the drafter ckpt
        res = es.generate("ACGTACGT...", n_tokens=1024)   # sampling (T=1.0, top_k=4)
        print(res.text, f"{res.tok_s:.1f} tok/s, tau={res.mean_tau:.2f}")

        g = es.generate(prompt, greedy=True)              # exact greedy
        nat = es.generate_native(prompt, greedy=True)     # native reference
        assert g.ids.tolist() == nat.ids.tolist()         # token-for-token identical
"""

from __future__ import annotations

from evspark.api import EvSpark, GenerateResult, clean_prompt, ids_to_text
from evspark.checkpoints import KNOWN_CKPTS, ensure_ckpt

__version__ = "0.1.0"

__all__ = [
    "EvSpark",
    "GenerateResult",
    "KNOWN_CKPTS",
    "clean_prompt",
    "ensure_ckpt",
    "ids_to_text",
    "__version__",
]
