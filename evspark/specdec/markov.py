"""k 阶 Markov drafter：基因组转移表 + NextProbModel 适配器。

字母表固定 5 符号 ACGTN（内部 id 0..4）。统计滑动窗口 counts[ctx, next]，
add-α 平滑后行归一化为 P(next | ctx)。未见上下文经平滑后为均匀分布，无需 backoff。

适配器 ``MarkovDraftModel`` 实现 Step 1 的 ``NextProbModel``：
prefix 为 vortex 字节 id（A=65, C=67, G=71, T=84, N=78），输出 512 维概率，
仅在这 5 个 id 上有质量且和为 1。

与 transforms 组合时以 ``log(probs)`` 作为 logits 传入
（temperature 作用于 Markov 概率的一致重整形）；本步不测该组合，属 B.3。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from evspark.specdec.genome import N_ID, N_SYMBOLS, clean_seq

DEFAULT_ALPHA = 0.1
VOCAB_SIZE = 512
HOLD_OUT_FRAC = 0.1

# vortex / 字节词表 id
BYTE_A, BYTE_C, BYTE_G, BYTE_T, BYTE_N = 65, 67, 71, 84, 78
MASS_IDS = (BYTE_A, BYTE_C, BYTE_G, BYTE_T, BYTE_N)

_SYM_FROM_UPPER_BYTE = np.full(256, N_ID, dtype=np.int64)
_SYM_FROM_UPPER_BYTE[ord("A")] = 0
_SYM_FROM_UPPER_BYTE[ord("C")] = 1
_SYM_FROM_UPPER_BYTE[ord("G")] = 2
_SYM_FROM_UPPER_BYTE[ord("T")] = 3
_SYM_FROM_UPPER_BYTE[ord("N")] = 4


def _seq_to_ids(seq: str) -> np.ndarray:
    """已清洗序列 → 内部 0..4 id。未清洗输入会先走 clean_seq。"""
    cleaned = clean_seq(seq)
    if not cleaned:
        return np.empty(0, dtype=np.int64)
    raw = np.frombuffer(cleaned.encode("ascii"), dtype=np.uint8)
    return _SYM_FROM_UPPER_BYTE[raw]


def _powers(k: int) -> np.ndarray:
    return (N_SYMBOLS ** np.arange(k - 1, -1, -1)).astype(np.int64)


def context_code(ids_k: np.ndarray, powers: np.ndarray | None = None) -> int:
    """长度 k 的内部 id 向量 → 5 进制上下文编号。"""
    ids_k = np.asarray(ids_k, dtype=np.int64).ravel()
    if powers is None:
        powers = _powers(int(ids_k.size))
    return int(ids_k @ powers)


def byte_id_to_sym(byte_id: int) -> int:
    """vortex 字节 id → 内部 0..4；小写先转大写，其余（含越界）视为 N。"""
    b = int(byte_id)
    if b < 0 or b > 255:
        return N_ID
    if 97 <= b <= 122:  # a-z
        b -= 32
    return int(_SYM_FROM_UPPER_BYTE[b])


def _count_transitions(ids: np.ndarray, k: int, powers: np.ndarray, n_ctx: int) -> np.ndarray:
    """一条序列的 (ctx, next) 计数，shape [5^k, 5]。不左填充——只统计真实 k-mer。"""
    counts = np.zeros((n_ctx, N_SYMBOLS), dtype=np.int64)
    if ids.size <= k:
        return counts
    windows = sliding_window_view(ids, k)
    ctx = windows[:-1]
    nxt = ids[k:]
    codes = ctx @ powers
    idx = codes * N_SYMBOLS + nxt
    counts.ravel()[:] = np.bincount(idx, minlength=n_ctx * N_SYMBOLS)
    return counts


@dataclass
class MarkovTable:
    k: int
    alpha: float
    counts: np.ndarray  # int64 [5^k, 5]
    probs: np.ndarray  # float64 [5^k, 5]

    @classmethod
    def build(
        cls,
        seqs: list[str],
        k: int,
        alpha: float = DEFAULT_ALPHA,
    ) -> "MarkovTable":
        if k < 1:
            raise ValueError(f"k 须 ≥ 1，得到 {k}")
        if alpha < 0.0:
            raise ValueError(f"alpha 须 ≥ 0，得到 {alpha}")
        n_ctx = int(N_SYMBOLS**k)
        powers = _powers(k)
        counts = np.zeros((n_ctx, N_SYMBOLS), dtype=np.int64)
        for seq in seqs:
            ids = _seq_to_ids(seq)
            counts += _count_transitions(ids, k, powers, n_ctx)
        # add-α 行归一化；未见上下文 → 均匀 1/5
        denom = counts.sum(axis=1, keepdims=True).astype(np.float64) + alpha * N_SYMBOLS
        probs = (counts.astype(np.float64) + alpha) / denom
        return cls(k=int(k), alpha=float(alpha), counts=counts, probs=probs)


class MarkovDraftModel:
    """``NextProbModel`` 适配器：字节 id 前缀 → 512 维下一 token 分布。"""

    vocab_size = VOCAB_SIZE

    def __init__(self, table: MarkovTable):
        self.table = table
        self.k = int(table.k)
        self._powers = _powers(self.k)
        self._mass = np.array(MASS_IDS, dtype=np.int64)

    def _prefix_to_ctx_code(self, prefix: np.ndarray) -> int:
        prefix = np.asarray(prefix, dtype=np.int64).ravel()
        ctx = np.full(self.k, N_ID, dtype=np.int64)
        if prefix.size > 0:
            take = min(self.k, int(prefix.size))
            tail = prefix[-take:]
            mapped = np.empty(take, dtype=np.int64)
            for i, b in enumerate(tail):
                mapped[i] = byte_id_to_sym(int(b))
            ctx[-take:] = mapped
        return int(ctx @ self._powers)

    def probs(self, prefix: np.ndarray) -> np.ndarray:
        """下一 token 分布，shape ``[512]``，质量只在 ACGTN 字节 id 上。"""
        code = self._prefix_to_ctx_code(prefix)
        row = self.table.probs[code]
        out = np.zeros(VOCAB_SIZE, dtype=np.float64)
        out[self._mass] = row
        return out


@dataclass(frozen=True)
class HoldoutMetrics:
    k: int
    alpha: float
    n_train_trans: int
    n_test: int
    top1: float
    ce_bits: float


def split_contiguous(seq: str, test_frac: float = HOLD_OUT_FRAC) -> tuple[str, str]:
    """连续切片：末 ``test_frac`` 作测试集。禁止 shuffle。"""
    if test_frac < 0.0 or test_frac > 1.0:
        raise ValueError(f"test_frac 须在 [0, 1]，得到 {test_frac}")
    n = len(seq)
    split = int(n * (1.0 - test_frac))
    return seq[:split], seq[split:]


def _eval_positions(
    table: MarkovTable,
    seq: str,
    start: int,
) -> tuple[int, float, int]:
    """对 ``seq[start:]`` 的每个位置做 next-token 评估；上下文取真实前缀（可跨切分点）。"""
    ids = _seq_to_ids(seq)
    L = int(ids.size)
    if start >= L:
        return 0, 0.0, 0
    k = table.k
    powers = _powers(k)
    pad = np.concatenate([np.full(k, N_ID, dtype=np.int64), ids])
    windows = sliding_window_view(pad, k)
    ctx = windows[start:L]
    nxt = ids[start:L]
    codes = ctx @ powers
    rows = table.probs[codes]
    pred = np.argmax(rows, axis=1)
    hits = int(np.sum(pred == nxt))
    p_true = rows[np.arange(nxt.size), nxt]
    ce_sum = float(-np.log2(p_true).sum())
    return hits, ce_sum, int(nxt.size)


def evaluate_holdout(
    seqs: list[str],
    ks: tuple[int, ...] = (3, 4, 5),
    alpha: float = DEFAULT_ALPHA,
    test_frac: float = HOLD_OUT_FRAC,
) -> list[HoldoutMetrics]:
    """每条序列末 ``test_frac`` 连续切片作测试；k 各评一次。

    训练集 = 各序列前缀拼接后的转移计数（不 shuffle、不跨序列拼接 k-mer）。
    测试位置用该条序列的真实前文（可落在训练前缀内），无标签泄漏。
    """
    train_seqs: list[str] = []
    splits: list[tuple[str, int]] = []
    for seq in seqs:
        train, test = split_contiguous(seq, test_frac=test_frac)
        if train:
            train_seqs.append(train)
        if test:
            splits.append((seq, len(train)))
    out: list[HoldoutMetrics] = []
    for k in ks:
        table = MarkovTable.build(train_seqs, k=int(k), alpha=alpha)
        hits = 0
        ce_sum = 0.0
        n_test = 0
        for seq, start in splits:
            h, c, n = _eval_positions(table, seq, start)
            hits += h
            ce_sum += c
            n_test += n
        top1 = (hits / n_test) if n_test else float("nan")
        ce_bits = (ce_sum / n_test) if n_test else float("nan")
        out.append(
            HoldoutMetrics(
                k=int(k),
                alpha=float(alpha),
                n_train_trans=int(table.counts.sum()),
                n_test=n_test,
                top1=float(top1),
                ce_bits=float(ce_bits),
            )
        )
    return out


def _cli() -> None:
    import argparse
    import hashlib
    from pathlib import Path

    from evspark.specdec.genome import cleaning_stats, iter_raw_sequences, load_sequences

    parser = argparse.ArgumentParser(description="Markov drafter holdout 评估")
    parser.add_argument("--data", type=str, required=True, help="FASTA/CSV 路径")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--ks", type=int, nargs="+", default=[3, 4, 5])
    args = parser.parse_args()

    path = Path(args.data)
    raw = list(iter_raw_sequences(path))
    stats = cleaning_stats(raw)
    seqs = load_sequences(path)
    n_bp = sum(len(s) for s in seqs)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"path={path}")
    print(f"sha256={digest}")
    print(f"size_bytes={path.stat().st_size}")
    print(f"n_seqs={len(seqs)} n_bp={n_bp}")
    print(f"cleaning={stats}")
    metrics = evaluate_holdout(seqs, ks=tuple(args.ks), alpha=args.alpha)
    print("k\talpha\tn_train_trans\tn_test\ttop1\tce_bits")
    for m in metrics:
        print(
            f"{m.k}\t{m.alpha}\t{m.n_train_trans}\t{m.n_test}\t"
            f"{m.top1:.6f}\t{m.ce_bits:.6f}"
        )
        gate = "PASS" if m.k != 5 or m.ce_bits < 1.99 else "FAIL (>=1.99, 视为实现 bug)"
        if m.k == 5:
            print(f"sanity_gate_k5: ce={m.ce_bits:.6f} vs random 2.0 bits/base → {gate}")


if __name__ == "__main__":
    _cli()
