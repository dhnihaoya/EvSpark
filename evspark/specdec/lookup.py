"""prompt-lookup drafter：最近 k-mer 匹配续延 + 混合分布 q。

从当前前缀（prompt + 已生成）里 ``rfind`` 查询 k-mer，取其最近一次出现的
下一碱基作为 x*。拒绝采样要求 q 是合法分布，因此采用混合：

    q(x) = λ·1[x = x*] + (1−λ)·(1/5)    x ∈ ACGTN

默认 λ=0.9，故 q(x*)=0.92、其余 4 符号各 0.02。贪心下 argmax(q)=x*，
退化为确定性查表。无匹配或前缀短于 k_lookup 时走 fallback：
均匀 ACGTN，或可选委托 ``MarkovDraftModel``。

搜索范围是 ``s[:n-1]``（``bytes.rfind(..., end=n-1)``），天然排除末位
平凡自匹配；重叠匹配允许（同聚体续延 A 是正确行为）。

适配器实现 Step 1 的 ``NextProbModel``：prefix 为 vortex 字节 id，
输出 512 维，质量只在 {65,67,71,84,78}。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evspark.specdec.markov import (
    HOLD_OUT_FRAC,
    MASS_IDS,
    VOCAB_SIZE,
    MarkovDraftModel,
    byte_id_to_sym,
    split_contiguous,
)

DEFAULT_K_LOOKUP = 10
DEFAULT_LAMBDA = 0.9
N_SYMBOLS = 5
UNIFORM = 1.0 / N_SYMBOLS
_SYM_CHARS = b"ACGTN"
MASS = np.array(MASS_IDS, dtype=np.int64)


def prefix_to_acgtn_bytes(prefix: np.ndarray) -> bytes:
    """vortex 字节 id 前缀 → ACGTN 字节串；小写转大写，其余（含 ``|``）→ N。"""
    p = np.asarray(prefix, dtype=np.int64).ravel()
    if p.size == 0:
        return b""
    return bytes(_SYM_CHARS[byte_id_to_sym(int(b))] for b in p)


def rfind_continuation(seq: bytes, k: int) -> tuple[int, int] | None:
    """在 ``seq[:n-1]`` 中 ``rfind`` 末 k-mer，返回 ``(m, xstar_byte)`` 或 None。

    ``m`` 为最近匹配起点；``xstar_byte`` 为 ``seq[m+k]``（ASCII，属 ACGTN）。
    前缀短于 k、或 haystack 中无匹配 → None。
    """
    n = len(seq)
    if k < 1 or n < k:
        return None
    needle = seq[n - k : n]
    # end=n-1 ⇒ 搜索 seq[0:n-1]，不含末位，排除平凡自匹配
    m = seq.rfind(needle, 0, n - 1)
    if m < 0:
        return None
    return m, seq[m + k]


def _uniform_512() -> np.ndarray:
    out = np.zeros(VOCAB_SIZE, dtype=np.float64)
    out[MASS] = UNIFORM
    return out


def _mixture_512(xstar_byte: int, lam: float) -> np.ndarray:
    """q(x) = λ·1[x=x*] + (1-λ)/5，质量只写到 5 个 byte id。"""
    mix = (1.0 - lam) * UNIFORM
    out = np.zeros(VOCAB_SIZE, dtype=np.float64)
    out[MASS] = mix
    idx = int(xstar_byte)
    if 0 <= idx < VOCAB_SIZE:
        out[idx] = lam + mix
    return out


class PromptLookupModel:
    """``NextProbModel``：最近 k-mer 续延的混合分布 drafter。

    ``fallback`` 为 None 时无匹配走均匀；传入 ``MarkovDraftModel`` 则委托其 ``probs``。
    内部把 prefix 转成 ACGTN 字节串再 ``rfind``（本数据规模足够，不做索引）。
    """

    vocab_size = VOCAB_SIZE

    def __init__(
        self,
        k_lookup: int = DEFAULT_K_LOOKUP,
        lam: float = DEFAULT_LAMBDA,
        fallback: MarkovDraftModel | None = None,
    ):
        if k_lookup < 1:
            raise ValueError(f"k_lookup 须 ≥ 1，得到 {k_lookup}")
        if lam < 0.0 or lam > 1.0:
            raise ValueError(f"lam 须在 [0, 1]，得到 {lam}")
        self.k_lookup = int(k_lookup)
        self.lam = float(lam)
        self.fallback = fallback
        # 符号序列缓存：同一前缀重复查询时免重建；增长则追加，回滚则重建
        self._cache_ids: np.ndarray | None = None
        self._cache_seq: bytes = b""

    def _seq_bytes(self, prefix: np.ndarray) -> bytes:
        prefix = np.asarray(prefix, dtype=np.int64).ravel()
        cached = self._cache_ids
        if cached is not None and prefix.size >= cached.size:
            n0 = int(cached.size)
            if n0 == 0 or np.array_equal(prefix[:n0], cached):
                extra = prefix[n0:]
                if extra.size:
                    self._cache_seq = self._cache_seq + prefix_to_acgtn_bytes(extra)
                    self._cache_ids = prefix.copy()
                elif prefix.size == n0:
                    pass
                return self._cache_seq
        self._cache_seq = prefix_to_acgtn_bytes(prefix)
        self._cache_ids = prefix.copy()
        return self._cache_seq

    def lookup(self, prefix: np.ndarray) -> tuple[int, int] | None:
        """返回 ``(匹配起点 m, x* 字节 id)``；无匹配为 None。"""
        seq = self._seq_bytes(prefix)
        return rfind_continuation(seq, self.k_lookup)

    def _no_match_probs(self, prefix: np.ndarray) -> np.ndarray:
        if self.fallback is not None:
            return np.asarray(self.fallback.probs(prefix), dtype=np.float64)
        return _uniform_512()

    def probs(self, prefix: np.ndarray) -> np.ndarray:
        """下一 token 分布，shape ``[512]``。有匹配则混合分布，否则 fallback。

        draft 侧由 harness 从本分布采样（禁止在模型内做确定性提议）。
        """
        hit = self.lookup(prefix)
        if hit is None:
            return self._no_match_probs(prefix)
        _m, xstar = hit
        return _mixture_512(int(xstar), self.lam)


@dataclass(frozen=True)
class LookupHoldoutMetrics:
    k_lookup: int
    n_test: int
    n_occ: int
    n_hit: int
    occurrence_rate: float
    hit_rate_given_occ: float
    hit_rate_overall: float


def evaluate_lookup_holdout(
    seqs: list[str],
    ks: tuple[int, ...] = (8, 10, 12, 16),
    test_frac: float = HOLD_OUT_FRAC,
) -> list[LookupHoldoutMetrics]:
    """每条序列末 ``test_frac`` 连续切片作测试；只用本窗口、数据库随位置增长。

    测试位置 j（与 Step 2 相同：``split = int(n*(1-test_frac))`` 起）：
    查询 ``s[j-k:j]``，数据库 ``s[:j-1]``，x* 与真实 ``s[j]`` 比较。
    无匹配记 miss（不计入 given_occ 的分子分母之外的 overall 分母仍含该位）。
    """
    splits: list[tuple[bytes, int]] = []
    for seq in seqs:
        _train, test = split_contiguous(seq, test_frac=test_frac)
        if not test:
            continue
        splits.append((seq.encode("ascii"), len(seq) - len(test)))

    out: list[LookupHoldoutMetrics] = []
    for k in ks:
        k = int(k)
        n_test = 0
        n_occ = 0
        n_hit = 0
        for s, start in splits:
            n = len(s)
            for j in range(start, n):
                n_test += 1
                hit = rfind_continuation(s[:j], k)
                if hit is None:
                    continue
                n_occ += 1
                _m, xstar = hit
                if xstar == s[j]:
                    n_hit += 1
        occ_rate = (n_occ / n_test) if n_test else float("nan")
        given = (n_hit / n_occ) if n_occ else float("nan")
        overall = (n_hit / n_test) if n_test else float("nan")
        out.append(
            LookupHoldoutMetrics(
                k_lookup=k,
                n_test=n_test,
                n_occ=n_occ,
                n_hit=n_hit,
                occurrence_rate=float(occ_rate),
                hit_rate_given_occ=float(given),
                hit_rate_overall=float(overall),
            )
        )
    return out


def _cli() -> None:
    import argparse
    from pathlib import Path

    from evspark.specdec.genome import load_sequences

    parser = argparse.ArgumentParser(description="prompt-lookup drafter holdout 评估")
    parser.add_argument("--data", type=str, required=True, help="FASTA/CSV 路径")
    parser.add_argument(
        "--ks", type=int, nargs="+", default=[8, 10, 12, 16], help="k_lookup 列表"
    )
    args = parser.parse_args()

    path = Path(args.data)
    seqs = load_sequences(path)
    n_bp = sum(len(s) for s in seqs)
    print(f"path={path}")
    print(f"n_seqs={len(seqs)} n_bp={n_bp}")
    metrics = evaluate_lookup_holdout(seqs, ks=tuple(args.ks))
    print("k_lookup\tn_test\tn_occ\tn_hit\toccurrence_rate\thit_rate_given_occ\thit_rate_overall")
    for m in metrics:
        print(
            f"{m.k_lookup}\t{m.n_test}\t{m.n_occ}\t{m.n_hit}\t"
            f"{m.occurrence_rate:.6f}\t{m.hit_rate_given_occ:.6f}\t{m.hit_rate_overall:.6f}"
        )
        if m.k_lookup == 8:
            gate = "PASS" if m.hit_rate_given_occ >= 0.35 else "FAIL (<0.35, 视为实现 bug)"
            print(
                f"sanity_gate_k8: hit_rate_given_occ={m.hit_rate_given_occ:.6f} "
                f"vs 0.35 → {gate}"
            )


if __name__ == "__main__":
    _cli()
