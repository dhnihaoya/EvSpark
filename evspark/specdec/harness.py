"""mock 模型协议下的原生解码 / 投机解码循环。

target 侧调用写成「收集全部 p_i 后一次性交给 verify_round」，
接口形状与将来一次块前向返回 ``[γ+1, V]`` 对齐。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from evspark.specdec.verifier import VerifyResult, sample_categorical, verify_round, verify_round_greedy


class NextProbModel(Protocol):
    def probs(self, prefix: np.ndarray) -> np.ndarray:
        """下一 token 分布，shape ``[V]``。"""
        ...


def run_native(
    model: NextProbModel,
    prompt: np.ndarray,
    n_tokens: int,
    rng: np.random.Generator,
    greedy: bool = False,
) -> np.ndarray:
    """逐 token 原生解码。``greedy=True`` 时用 ``np.argmax``（最小 index 并列约定）。

    ``greedy`` 为可选参数（默认 False），位置参数签名与契约一致。
    """
    prompt = np.asarray(prompt, dtype=np.int64).ravel()
    if n_tokens < 0:
        raise ValueError("n_tokens 不能为负")
    if n_tokens == 0:
        return np.empty(0, dtype=np.int64)

    buf = np.empty(prompt.size + n_tokens, dtype=np.int64)
    buf[: prompt.size] = prompt
    length = int(prompt.size)
    out = np.empty(n_tokens, dtype=np.int64)
    for t in range(n_tokens):
        dist = model.probs(buf[:length])
        tok = int(np.argmax(dist)) if greedy else sample_categorical(dist, rng)
        out[t] = tok
        buf[length] = tok
        length += 1
    return out


def run_speculative(
    target: NextProbModel,
    draft: NextProbModel,
    prompt: np.ndarray,
    n_tokens: int,
    gamma: int,
    rng: np.random.Generator,
    greedy: bool = False,
) -> tuple[np.ndarray, list[VerifyResult]]:
    """投机解码。draft 自回归提出 γ 个 token；target 先收集 γ+1 个分布再验证。"""
    prompt = np.asarray(prompt, dtype=np.int64).ravel()
    if n_tokens < 0:
        raise ValueError("n_tokens 不能为负")
    if gamma < 1:
        raise ValueError("gamma 须 ≥ 1")
    if n_tokens == 0:
        return np.empty(0, dtype=np.int64), []

    # 缓冲：prompt + 已生成 + 本轮最多 γ 个 draft（验证时拼到 prefix 上）
    buf = np.empty(prompt.size + n_tokens + gamma, dtype=np.int64)
    buf[: prompt.size] = prompt
    length = int(prompt.size)
    out = np.empty(n_tokens, dtype=np.int64)
    filled = 0
    results: list[VerifyResult] = []

    while filled < n_tokens:
        # draft 自回归提出 γ 个，逐位置记录 q_i
        draft_tokens = np.empty(gamma, dtype=np.int64)
        q_rows: list[np.ndarray] = []
        dlen = length
        for i in range(gamma):
            q_i = np.asarray(draft.probs(buf[:dlen]), dtype=np.float64)
            q_rows.append(q_i)
            tok = int(np.argmax(q_i)) if greedy else sample_categorical(q_i, rng)
            draft_tokens[i] = tok
            buf[dlen] = tok
            dlen += 1
        draft_probs = np.stack(q_rows, axis=0)

        # target：收集 p_1..p_{γ+1}（含 bonus 位），再一次性交给 verify_round
        # 注意：这里按「draft 提议的前缀」取 p_i，与块前向喂 [prefix | draft] 同形
        p_rows: list[np.ndarray] = []
        plen = length
        for i in range(gamma + 1):
            p_i = np.asarray(target.probs(buf[:plen]), dtype=np.float64)
            p_rows.append(p_i)
            if i < gamma:
                buf[plen] = draft_tokens[i]
                plen += 1
        target_probs = np.stack(p_rows, axis=0)

        if greedy:
            result = verify_round_greedy(draft_tokens, target_probs)
        else:
            result = verify_round(draft_tokens, draft_probs, target_probs, rng)
        results.append(result)

        take = min(int(result.tokens.shape[0]), n_tokens - filled)
        out[filled : filled + take] = result.tokens[:take]
        buf[length : length + take] = result.tokens[:take]
        length += take
        filled += take

    return out, results
