"""EvSpark — lossless speculative decoding for Evo2 (public API).

One-line usage::

    from evspark import EvSpark

    es = EvSpark.load("L27_g12_80M_s1")          # auto-downloads the drafter ckpt
    res = es.generate("ACGTACGT...", n_tokens=1024)   # sampling (T=1.0, top_k=4)
    print(res.text, f"{res.tok_s:.1f} tok/s", f"tau={res.mean_tau:.2f}")

    g = es.generate(prompt, greedy=True)         # exact greedy decoding
    nat = es.generate_native(prompt, greedy=True)  # native reference
    assert g.ids.tolist() == nat.ids.tolist()    # token-for-token identical
    es.close()

Requires one GPU with ~40 GB VRAM (Evo2 7B bf16). The target model is never
fine-tuned; the drafter checkpoint carries everything EvSpark needs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_CKPT = "L27_g12_80M_s1"


def clean_prompt(seq: str) -> str:
    """大写化并校验 ACGTN；过短或含非法字符直接报错。"""
    seq = "".join(seq.split()).upper()
    bad = sorted(set(seq) - set("ACGTN"))
    if bad:
        raise ValueError(f"prompt contains non-ACGTN characters: {bad}")
    if len(seq) < 64:
        raise ValueError("prompt too short (<64 bp); give the model some context")
    return seq


def ids_to_text(ids) -> str:
    """token id = ASCII 字节值（CharLevelTokenizer 口径）。"""
    arr = np.asarray(ids, dtype=np.int64).ravel()
    arr = np.clip(arr, 0, 255)
    return bytes(arr.tolist()).decode("ascii", errors="replace")


@dataclass
class GenerateResult:
    """一次生成的结果。``mean_tau``/``n_rounds`` 仅投机路径有值。"""

    text: str
    ids: np.ndarray
    mode: str  # "greedy" | "sampling"
    speculative: bool
    n_tokens: int
    wall_s: float
    tok_s: float
    mean_tau: float | None = None
    n_rounds: int | None = None
    extra: dict = field(default_factory=dict)


class EvSpark:
    """Evo2 + EvSpark drafter 的封装：投机生成 / 原生参照。"""

    def __init__(self, evo, nd, device: str = "cuda:0"):
        self._evo = evo
        self._nd = nd
        self.device = device
        self.model = evo.model
        self.tokenizer = evo.tokenizer

    @classmethod
    def load(
        cls,
        ckpt: str = DEFAULT_CKPT,
        model_name: str = "evo2_7b",
        device: str = "cuda:0",
        use_kernels: bool = True,
    ) -> "EvSpark":
        """加载 Evo2 + drafter。``ckpt`` 为本地 .pt 路径或已发布 ckpt 名
        （缺则自动从 Hugging Face / ModelScope 下载，见 evspark.checkpoints）。"""
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required (Evo2 7B needs ~40 GB VRAM in bf16).")

        from evspark.checkpoints import ensure_ckpt

        ckpt_path = ensure_ckpt(ckpt)

        from evo2 import Evo2

        print(f"[evspark] loading {model_name} (bf16) ...", flush=True)
        evo = Evo2(model_name, use_kernels=use_kernels)
        evo.model.eval()

        from evspark.specdec.block.neural_draft import NeuralDraftModel

        nd = NeuralDraftModel.from_checkpoint(evo.model, str(ckpt_path), device=device)
        print(f"[evspark] drafter={Path(str(ckpt_path)).name} "
              f"layers={list(nd.layer_names)} gamma={nd.gamma}", flush=True)
        return cls(evo, nd, device)

    @property
    def gamma(self) -> int:
        return int(self._nd.gamma)

    def _tokenize(self, prompt: str):
        import torch

        seq = clean_prompt(prompt)
        ids = torch.tensor(
            np.asarray(self.tokenizer.tokenize(seq), dtype=np.int64)[None, :],
            device=self.device,
        )
        return seq, ids

    def generate(
        self,
        prompt: str,
        n_tokens: int = 256,
        *,
        greedy: bool = False,
        gamma: int | None = None,
        seed: int = 0,
        temperature: float = 1.0,
        top_k: int = 4,
    ) -> GenerateResult:
        """投机解码（无损）。greedy=True 时与原生逐 token 逐位相等
        （bf16 tie-flip 除外，见论文 losslessness 一节）。"""
        import torch

        from evspark.specdec.block.loop import speculative_generate

        seq, ids = self._tokenize(prompt)
        g = int(gamma) if gamma else self.gamma
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        spec = speculative_generate(
            self.model, self._nd, ids, int(n_tokens), g, greedy=greedy,
            rng=np.random.default_rng(seed), temperature=temperature, top_k=top_k,
            rollback="slice",
        )
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        ks = [int(r.k) for r in spec.rounds_log]
        out_ids = np.asarray(spec.emitted_ids)
        return GenerateResult(
            text=ids_to_text(out_ids),
            ids=out_ids,
            mode="greedy" if greedy else "sampling",
            speculative=True,
            n_tokens=int(n_tokens),
            wall_s=wall,
            tok_s=float(n_tokens / wall) if wall > 0 else float("nan"),
            mean_tau=float(np.mean(ks) + 1.0) if ks else None,
            n_rounds=len(ks),
            extra={"gamma": g, "seed": seed, "prompt_bp": len(seq)},
        )

    def generate_native(
        self,
        prompt: str,
        n_tokens: int = 256,
        *,
        greedy: bool = False,
        seed: int = 0,
        temperature: float = 1.0,
        top_k: int = 4,
    ) -> GenerateResult:
        """原生逐 token 参照（同 prompt 同参数；用于对拍/计时基线）。"""
        import torch

        from evspark.specdec.block.loop import native_greedy_reference, native_sample_reference

        seq, ids = self._tokenize(prompt)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if greedy:
            nat_ids, _ip, _ = native_greedy_reference(self.model, ids, int(n_tokens))
        else:
            nat_ids = native_sample_reference(
                self.model, ids, int(n_tokens), np.random.default_rng(seed),
                temperature=temperature, top_k=top_k,
            )
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        out_ids = np.asarray(nat_ids)
        return GenerateResult(
            text=ids_to_text(out_ids),
            ids=out_ids,
            mode="greedy" if greedy else "sampling",
            speculative=False,
            n_tokens=int(n_tokens),
            wall_s=wall,
            tok_s=float(n_tokens / wall) if wall > 0 else float("nan"),
            extra={"seed": seed, "prompt_bp": len(seq)},
        )

    def close(self) -> None:
        if self._nd is not None:
            self._nd.close()
            self._nd = None

    def __enter__(self) -> "EvSpark":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
