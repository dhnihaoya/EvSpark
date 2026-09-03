"""拒绝采样验证器：Leviathan et al. 2023 标准算法（采样）+ 贪心模式。

一轮中 draft 提出 γ 个 token ``x_1..x_γ``；
``q_i = q(·|prefix, x_<i)``（i=1..γ）；
``p_i = p(·|prefix, x_<i)``（i=1..γ），``p_{γ+1}`` 为 bonus 位。

数值：概率比与残差一律 fp64。``q_i(x_i)`` 在正常 draft 下不为 0；
``p_i(x_i)=0`` 必须安全（必然拒绝，且不产生 NaN）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VerifyResult:
    tokens: np.ndarray  # 本轮产出的 token id，长度 = accepted_len + 1
    accepted_len: int  # 0..γ
    accepted_mask: np.ndarray  # bool[γ]，逐位置接受标记（供位置接受率曲线）
    from_residual: bool  # 末位是否来自残差重采样（False 则为 bonus）
    fallback: bool  # 残差下溢 fallback 是否触发


def _as_prob_row(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64).ravel()
    p = np.maximum(p, 0.0)
    s = float(p.sum())
    if s <= 0.0 or not np.isfinite(s):
        return np.full(p.shape[0], 1.0 / p.shape[0], dtype=np.float64)
    return p / s


def sample_categorical(probs: np.ndarray, rng: np.random.Generator) -> int:
    """从离散分布采样；输入不必已归一化。"""
    p = _as_prob_row(probs)
    return int(rng.choice(p.shape[0], p=p))


def _accept_prob(p_x: float, q_x: float) -> float:
    """min(1, p(x)/q(x))；q=0 时不除零。p=0 且 q>0 → 0（必然拒绝）。"""
    if q_x > 0.0:
        ratio = p_x / q_x
        if not np.isfinite(ratio):
            return 0.0
        return 1.0 if ratio >= 1.0 else float(ratio)
    # draft 不应提出支撑集外 token；q=0 时仅当 p>0 才视为可接受（比值 +∞）
    return 1.0 if p_x > 0.0 else 0.0


def verify_round(
    draft_tokens: np.ndarray,
    draft_probs: np.ndarray,
    target_probs: np.ndarray,
    rng: np.random.Generator,
) -> VerifyResult:
    """采样模式拒绝采样（Leviathan et al. 2023 §3）。

    对 i = 1..γ 依次：抽 r ~ U(0,1)，若 ``r < min(1, p_i(x_i)/q_i(x_i))`` 则接受；
    首次拒绝于 j：从残差 ``norm(max(0, p_j − q_j))`` 采样后本轮结束。
    全部接受：从 ``p_{γ+1}`` 采 bonus token。
    每轮产出 token 数 = 接受数 + 1（恒 ≥ 1）。

    ``temperature → 0`` 时本算法退化为 :func:`verify_round_greedy`
    （p、q 均为 one-hot：同 argmax 则必接受，否则残差即为 target 的 one-hot）。
    """
    draft_tokens = np.asarray(draft_tokens, dtype=np.int64).ravel()
    q = np.asarray(draft_probs, dtype=np.float64)
    p = np.asarray(target_probs, dtype=np.float64)
    gamma = int(draft_tokens.shape[0])
    if q.ndim != 2 or p.ndim != 2:
        raise ValueError(f"draft_probs/target_probs 须为二维，得到 {q.shape} / {p.shape}")
    if q.shape[0] != gamma:
        raise ValueError(f"draft_probs 首维须为 γ={gamma}，得到 {q.shape[0]}")
    if p.shape[0] != gamma + 1:
        raise ValueError(f"target_probs 首维须为 γ+1={gamma + 1}，得到 {p.shape[0]}")
    if q.shape[1] != p.shape[1]:
        raise ValueError(f"词表不一致：q V={q.shape[1]} vs p V={p.shape[1]}")

    accepted_mask = np.zeros(gamma, dtype=bool)
    accepted: list[int] = []

    for i in range(gamma):
        x_i = int(draft_tokens[i])
        p_x = float(p[i, x_i])
        q_x = float(q[i, x_i])
        # p(x)=0 → accept_prob=0，必然拒绝，且不会 NaN
        r = float(rng.random())
        if r < _accept_prob(p_x, q_x):
            accepted_mask[i] = True
            accepted.append(x_i)
            continue

        # 残差：数学上拒绝发生时 Σ max(0, p_j − q_j) > 0；
        # 若 fp64 下溢为 0，fallback 为直接从 p_j 采样。
        residual = np.maximum(p[i] - q[i], 0.0)
        mass = float(residual.sum())
        fallback = mass <= 0.0 or not np.isfinite(mass)
        if fallback:
            y = sample_categorical(p[i], rng)
        else:
            y = sample_categorical(residual, rng)
        tokens = np.asarray(accepted + [y], dtype=np.int64)
        return VerifyResult(
            tokens=tokens,
            accepted_len=len(accepted),
            accepted_mask=accepted_mask,
            from_residual=True,
            fallback=fallback,
        )

    bonus = sample_categorical(p[gamma], rng)
    tokens = np.asarray(accepted + [bonus], dtype=np.int64)
    return VerifyResult(
        tokens=tokens,
        accepted_len=gamma,
        accepted_mask=accepted_mask,
        from_residual=False,
        fallback=False,
    )


def verify_round_greedy(
    draft_tokens: np.ndarray,
    target_probs: np.ndarray,
) -> VerifyResult:
    """贪心模式（top_k=1 / argmax）。

    接受 ⇔ ``x_i == argmax(p_i)``；首次不匹配处输出 ``argmax(p_j)`` 结束；
    全部匹配则输出 ``argmax(p_{γ+1})`` 作 bonus。

    tie-break 必须确定性且与比较基准一致：统一「最小 index」约定
    （``np.argmax`` 语义）。后续与 vortex 原生 decode 对拍时，原生侧也须套
    同一约定（本步不涉及 vortex）。

    贪心拒绝位没有真正的残差分布，``from_residual=True`` 表示末位是纠正 token
    而非 bonus，与采样模式字段语义对齐（False ⇔ bonus）。
    """
    draft_tokens = np.asarray(draft_tokens, dtype=np.int64).ravel()
    p = np.asarray(target_probs, dtype=np.float64)
    gamma = int(draft_tokens.shape[0])
    if p.ndim != 2:
        raise ValueError(f"target_probs 须为二维，得到 {p.shape}")
    if p.shape[0] != gamma + 1:
        raise ValueError(f"target_probs 首维须为 γ+1={gamma + 1}，得到 {p.shape[0]}")

    accepted_mask = np.zeros(gamma, dtype=bool)
    accepted: list[int] = []

    for i in range(gamma):
        greedy_tok = int(np.argmax(p[i]))
        if int(draft_tokens[i]) == greedy_tok:
            accepted_mask[i] = True
            accepted.append(greedy_tok)
            continue
        tokens = np.asarray(accepted + [greedy_tok], dtype=np.int64)
        return VerifyResult(
            tokens=tokens,
            accepted_len=len(accepted),
            accepted_mask=accepted_mask,
            from_residual=True,
            fallback=False,
        )

    bonus = int(np.argmax(p[gamma]))
    tokens = np.asarray(accepted + [bonus], dtype=np.int64)
    return VerifyResult(
        tokens=tokens,
        accepted_len=gamma,
        accepted_mask=accepted_mask,
        from_residual=False,
        fallback=False,
    )
