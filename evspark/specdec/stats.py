"""τ（每轮接受长度）与位置条件接受率统计。"""

from __future__ import annotations

import numpy as np

from evspark.specdec.verifier import VerifyResult


def tau_from_mask(accepted_mask: np.ndarray) -> int:
    """由 ``accepted_mask`` 重算 τ：从左起连续 True 的个数。"""
    mask = np.asarray(accepted_mask, dtype=bool).ravel()
    tau = 0
    for flag in mask:
        if not flag:
            break
        tau += 1
    return tau


def mean_tau(results: list[VerifyResult]) -> float:
    if not results:
        return 0.0
    return float(np.mean([r.accepted_len for r in results]))


def position_accept_rates(results: list[VerifyResult]) -> np.ndarray:
    """各 draft 位置的无条件接受率（拒绝后的后缀 mask 为 False）。

    返回 shape ``[γ]``。首位接受率在该定义下必然 ≥ 任一后缀位置。
    """
    if not results:
        return np.empty(0, dtype=np.float64)
    masks = np.stack([np.asarray(r.accepted_mask, dtype=bool) for r in results], axis=0)
    return masks.mean(axis=0).astype(np.float64)


def position_conditional_accept_rates(results: list[VerifyResult]) -> np.ndarray:
    """位置 k 的条件接受率：在前 k 个 draft 均已接受的轮次中，第 k 位被接受的比例。"""
    if not results:
        return np.empty(0, dtype=np.float64)
    masks = np.stack([np.asarray(r.accepted_mask, dtype=bool) for r in results], axis=0)
    gamma = masks.shape[1]
    rates = np.zeros(gamma, dtype=np.float64)
    reached = np.ones(masks.shape[0], dtype=bool)
    for k in range(gamma):
        n = int(reached.sum())
        if n == 0:
            rates[k] = np.nan
        else:
            rates[k] = float(masks[reached, k].mean())
        reached = reached & masks[:, k]
    return rates
