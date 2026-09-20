"""等价性检验用的纯 numpy/stdlib 统计（Phase 4）。不依赖 scipy。

- 两样本 KS D
- bootstrap 百分位 CI
- TOST：90% CI 落入 [−Δ, Δ]
- 两样本比例差的 Fisher 精确检验（2×2）
- Benjamini–Hochberg
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np


def _as1d(x) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).ravel()
    return a[np.isfinite(a)]


def mean_diff(a, b) -> float:
    xa, xb = _as1d(a), _as1d(b)
    if xa.size == 0 or xb.size == 0:
        return float("nan")
    return float(xa.mean() - xb.mean())


def ks_d(a, b) -> float:
    """两样本 Kolmogorov–Smirnov 统计量 D = sup |F−G|。"""
    xa, xb = np.sort(_as1d(a)), np.sort(_as1d(b))
    if xa.size == 0 or xb.size == 0:
        return float("nan")
    grid = np.concatenate([xa, xb])
    fa = np.searchsorted(xa, grid, side="right") / xa.size
    fb = np.searchsorted(xb, grid, side="right") / xb.size
    return float(np.max(np.abs(fa - fb)))


def bootstrap_mean_diff_ci(
    a,
    b,
    *,
    n_boot: int,
    seed: int,
    alpha: float,
) -> tuple[float, float, float]:
    """返回 (point, lo, hi)。``alpha`` 是双侧尾，例如 0.10 → 90% CI。"""
    xa, xb = _as1d(a), _as1d(b)
    if xa.size == 0 or xb.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    diffs = np.empty(int(n_boot), dtype=np.float64)
    na, nb = xa.size, xb.size
    for i in range(int(n_boot)):
        sa = xa[rng.integers(0, na, size=na)]
        sb = xb[rng.integers(0, nb, size=nb)]
        diffs[i] = float(sa.mean() - sb.mean())
    point = float(xa.mean() - xb.mean())
    lo = float(np.quantile(diffs, 0.5 * alpha))
    hi = float(np.quantile(diffs, 1.0 - 0.5 * alpha))
    return point, lo, hi


def tost_mean_diff(
    a,
    b,
    margin: float,
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> dict:
    """α=0.05 TOST ⇔ 90% CI ⊂ [−margin, margin]。"""
    m = float(margin)
    if m <= 0:
        raise ValueError("margin 须为正")
    point, lo, hi = bootstrap_mean_diff_ci(
        a, b, n_boot=n_boot, seed=seed, alpha=2.0 * alpha
    )
    inside = bool(np.isfinite(lo) and np.isfinite(hi) and lo >= -m and hi <= m)
    return {
        "point": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "margin": m,
        "equivalent": inside,
        "ci_level": 1.0 - 2.0 * alpha,
    }


def bootstrap_stat_ci(
    compute: Callable[[np.ndarray, np.ndarray], float],
    a,
    b,
    *,
    n_boot: int,
    seed: int,
    alpha: float,
) -> tuple[float, float, float]:
    """对两条样本（行=观测）重采样后算标量统计量的百分位 CI。

    ``a``/``b`` 可以是 1d 数组，或 object 数组（例如每条序列一个 k-mer Counter）。
    ``compute(a_sample, b_sample)`` 必须返回 float。
    """
    xa = np.asarray(a, dtype=object)
    xb = np.asarray(b, dtype=object)
    if xa.size == 0 or xb.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(compute(xa, xb))
    rng = np.random.default_rng(int(seed))
    stats = np.empty(int(n_boot), dtype=np.float64)
    na, nb = xa.size, xb.size
    for i in range(int(n_boot)):
        ia = rng.integers(0, na, size=na)
        ib = rng.integers(0, nb, size=nb)
        stats[i] = float(compute(xa[ia], xb[ib]))
    lo = float(np.quantile(stats, 0.5 * alpha))
    hi = float(np.quantile(stats, 1.0 - 0.5 * alpha))
    return point, lo, hi


def tvd_upper_ok(point: float, hi: float, margin: float) -> bool:
    return bool(np.isfinite(point) and np.isfinite(hi) and hi < float(margin))


def _hypergeom_pmf(k: int, n: int, K: int, N: int) -> float:
    """P(X=k), X~Hypergeom(N, K, n)。"""
    if k < max(0, n - (N - K)) or k > min(n, K):
        return 0.0
    return math.comb(K, k) * math.comb(N - K, n - k) / math.comb(N, n)


def fisher_exact_two_sided(table: Sequence[Sequence[int]]) -> dict:
    """2×2 Fisher 精确检验（两尾）。table = [[a,b],[c,d]]。"""
    a, b = int(table[0][0]), int(table[0][1])
    c, d = int(table[1][0]), int(table[1][1])
    if min(a, b, c, d) < 0:
        raise ValueError("2×2 计数不能为负")
    n1 = a + b
    n2 = c + d
    k = a + c
    n = n1 + n2
    if n == 0 or n1 == 0 or n2 == 0:
        return {"odds_ratio": float("nan"), "pvalue": 1.0, "table": [[a, b], [c, d]]}
    lo = max(0, k - n2)
    hi = min(k, n1)
    p_obs = _hypergeom_pmf(a, n1, k, n)
    p = 0.0
    for x in range(lo, hi + 1):
        px = _hypergeom_pmf(x, n1, k, n)
        if px <= p_obs + 1e-15:
            p += px
    p = min(1.0, max(0.0, p))
    if b == 0 or c == 0:
        oratio = float("inf") if a > 0 and d > 0 else float("nan")
    else:
        oratio = (a / b) / (c / d) if (c / d) != 0 else float("inf")
    return {"odds_ratio": oratio, "pvalue": p, "table": [[a, b], [c, d]]}


def benjamini_hochberg(pvalues: Sequence[float], q: float = 0.05) -> list[bool]:
    """返回与输入等长的拒绝标记（FDR ≤ q）。NaN p 视为不拒绝。"""
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: (not np.isfinite(pvalues[i]), pvalues[i]))
    rejected = [False] * m
    thresh_idx = -1
    n_valid = sum(1 for p in pvalues if np.isfinite(p))
    if n_valid == 0:
        return rejected
    rank = 0
    for i in order:
        p = pvalues[i]
        if not np.isfinite(p):
            continue
        rank += 1
        if p <= (rank / n_valid) * q:
            thresh_idx = rank
    rank = 0
    for i in order:
        p = pvalues[i]
        if not np.isfinite(p):
            continue
        rank += 1
        if rank <= thresh_idx:
            rejected[i] = True
    return rejected
