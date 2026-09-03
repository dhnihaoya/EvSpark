"""4 符号 bigram mock target / draft，实现 :class:`NextProbModel` 协议。"""

from __future__ import annotations

import numpy as np

# 与 DNA 有效字母表对齐的 id：0=A, 1=C, 2=G, 3=T
TOKEN_A, TOKEN_C, TOKEN_G, TOKEN_T = 0, 1, 2, 3
DEFAULT_VOCAB_SIZE = 4


def _normalize_row(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64).ravel()
    p = np.maximum(p, 0.0)
    s = float(p.sum())
    if s <= 0.0 or not np.isfinite(s):
        return np.full(p.shape[0], 1.0 / p.shape[0], dtype=np.float64)
    return p / s


def _normalize_rows(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    out = np.empty_like(m, dtype=np.float64)
    for i in range(m.shape[0]):
        out[i] = _normalize_row(m[i])
    return out


class BigramModel:
    """一阶 Markov：``P(x_t | x_{t-1})``；空前缀用 ``start``。"""

    def __init__(self, transitions: np.ndarray, start: np.ndarray | None = None):
        transitions = np.asarray(transitions, dtype=np.float64)
        if transitions.ndim != 2 or transitions.shape[0] != transitions.shape[1]:
            raise ValueError(f"transitions 须为 [V, V]，得到 {transitions.shape}")
        self.vocab_size = int(transitions.shape[0])
        self.transitions = _normalize_rows(transitions)
        if start is None:
            self.start = np.full(self.vocab_size, 1.0 / self.vocab_size, dtype=np.float64)
        else:
            self.start = _normalize_row(start)
            if self.start.shape[0] != self.vocab_size:
                raise ValueError("start 维数与词表不一致")

    def probs(self, prefix: np.ndarray) -> np.ndarray:
        prefix = np.asarray(prefix, dtype=np.int64).ravel()
        if prefix.size == 0:
            return self.start.copy()
        prev = int(prefix[-1])
        if prev < 0 or prev >= self.vocab_size:
            raise IndexError(f"token id {prev} 超出词表 [0, {self.vocab_size})")
        return self.transitions[prev].copy()


def random_bigram(
    rng: np.random.Generator,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    alpha: float = 1.0,
) -> BigramModel:
    """每行独立 Dirichlet(α) 的随机 bigram 表。"""
    alpha_vec = np.full(vocab_size, float(alpha), dtype=np.float64)
    transitions = rng.dirichlet(alpha_vec, size=vocab_size)
    start = rng.dirichlet(alpha_vec)
    return BigramModel(transitions, start)
