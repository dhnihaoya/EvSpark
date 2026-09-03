"""投机解码 drafter 实现。

本步交付恒等 drafter（A.6）：抄上一 token 的 one-hot，用于验证循环框架
而非接受率。CPU-only、numpy；实现 Step 1 ``NextProbModel`` 协议。
"""

from __future__ import annotations

import numpy as np

VOCAB_SIZE = 512
BYTE_A, BYTE_C, BYTE_G, BYTE_T, BYTE_N = 65, 67, 71, 84, 78
MASS_IDS = (BYTE_A, BYTE_C, BYTE_G, BYTE_T, BYTE_N)


class IdentityDraftModel:
    """恒等 drafter：``q(·|prefix)`` = 上一 token 的 one-hot。

    空前缀退化为 ACGTN 均匀（各 1/5）。贪心下草稿 = 逐位复制上一 token；
    采样下 q 为 one-hot，接受率自然低——A.6 测的是框架正确性，不是 τ。
    """

    vocab_size = VOCAB_SIZE

    def probs(self, prefix: np.ndarray) -> np.ndarray:
        prefix = np.asarray(prefix, dtype=np.int64).ravel()
        out = np.zeros(VOCAB_SIZE, dtype=np.float64)
        if prefix.size == 0:
            out[np.array(MASS_IDS, dtype=np.int64)] = 1.0 / len(MASS_IDS)
            return out
        last = int(prefix[-1])
        if 0 <= last < VOCAB_SIZE:
            out[last] = 1.0
            return out
        out[np.array(MASS_IDS, dtype=np.int64)] = 1.0 / len(MASS_IDS)
        return out
