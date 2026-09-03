"""temperature / top_k 采样变换。p、q 两侧必须使用同一实现、同一参数。

项目默认 ``temperature=1.0, top_k=4``（在 512 维词表上是真截断）。

注意：与 vortex 原生采样对拍前，须核对 ``vortex/model/sample.py`` 的变换顺序
（先 temperature 再 top_k，或相反）以及 top_k 是在 logit 空间掩码还是在概率
空间置零。不一致则以 vortex 为准调整本实现。本步测试自包含，不受该核对影响。
"""

from __future__ import annotations

import numpy as np

DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_K = 4


def apply_transform(
    logits: np.ndarray,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k: int | None = DEFAULT_TOP_K,
) -> np.ndarray:
    """将 logits 变为采样分布。

    顺序：先 ``logits / T`` 再 softmax，再 top_k（非 top_k 位置置 0 后重新归一化）。

    ``temperature → 0`` 时退化为 one-hot(argmax)，与 :func:`verify_round_greedy`
    的「最小 index」tie-break 一致（``np.argmax`` 语义）。
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 1:
        raise ValueError(f"logits 须为一维 [V]，得到 shape={logits.shape}")
    vocab_size = int(logits.shape[0])
    if vocab_size == 0:
        raise ValueError("空词表")

    if temperature <= 0.0:
        # 贪心极限：one-hot，并列最大值取最小 index（np.argmax）
        probs = np.zeros(vocab_size, dtype=np.float64)
        probs[int(np.argmax(logits))] = 1.0
        return probs

    scaled = logits / float(temperature)
    scaled = scaled - np.max(scaled)
    exp = np.exp(scaled)
    z = float(exp.sum())
    if z <= 0.0 or not np.isfinite(z):
        probs = np.full(vocab_size, 1.0 / vocab_size, dtype=np.float64)
    else:
        probs = exp / z

    if top_k is None or top_k <= 0 or top_k >= vocab_size:
        return probs

    # 并列时按 (-prob, index) 排序，较小 index 优先进入 top_k，确定性可复现
    order = np.lexsort((np.arange(vocab_size), -probs))
    keep = order[: int(top_k)]
    masked = np.zeros(vocab_size, dtype=np.float64)
    masked[keep] = probs[keep]
    s = float(masked.sum())
    if s <= 0.0 or not np.isfinite(s):
        masked[keep] = 1.0 / float(top_k)
        return masked
    return masked / s
