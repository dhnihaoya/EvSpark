"""训练混合终版配比（final15）与加权窗分配。

**唯一权威定义在此**；dump 侧（``dump_c1_dataset.py`` 的配额公式）与训练侧
（``train_drafter.py --mix final15``）从这里取数，避免第三处重复表。

终版有效权重：钉住四源 gtdb 0.20 / mrna 0.315 /
imgvr 0.10 / euk 0.10，其余源保持 Step 14 有效值
（ncbi 0.135 / ncrna 0.072 / organelle 0.036 / promoters 0.027 / hg38 0.135，
合计 0.405）按 ``0.285/0.405`` 缩放凑满 1.0。

已知限制（如实记录，勿静默过采样）：hg38 目标 0.095 但落盘池仅 4.5M 位置，
训练侧按 ``cap_epochs``（默认 2.0 遍）封顶——150M 监督预算下实际占比 ≈2.6%，
与 Step 14 均匀盘面口径（1.7%）同量级，报告须写明 shortfall。

``allocate_epoch_windows``：给定各源可访问窗数与目标权重，求每 epoch 各源
访问窗数，使相对占比贴合权重、每源不超过 ``cap_epochs`` 遍自有池。解
``Σ_s min(cap·A_s, w_s·E) = Σ_s A_s`` 的 E（关于 E 单调，二分），被上限
截断的源把预算让给其余源（E 变大），未截断源间严格按权重比例。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Step 14 有效权重（MIX_SEVEN×0.9 + euk 0.10）中"其余源"的部分
_REST_STEP14 = {
    "ncbi": 0.135,
    "ncrna": 0.072,
    "organelle": 0.036,
    "promoters": 0.027,
    "hg38": 0.135,
}
_PINNED = {
    "gtdb": 0.20,
    "mrna": 0.315,
    "imgvr": 0.10,
    "euk": 0.10,
}
_REST_SCALE = (1.0 - sum(_PINNED.values())) / sum(_REST_STEP14.values())

FINAL_MIX = {**_PINNED, **{k: v * _REST_SCALE for k, v in _REST_STEP14.items()}}
assert abs(sum(FINAL_MIX.values()) - 1.0) < 1e-12

FINAL_MIX_NOTE = (
    "plans/17 §1.5 初值：钉住 gtdb 0.20 / mrna 0.315 / imgvr 0.10 / euk 0.10，"
    "其余源按 Step 14 有效值 ×0.7037 归一；hg38 受池上限（4.5M）由 cap_epochs 封顶"
)


def load_mix(spec: str) -> dict[str, float] | None:
    """--mix 参数 → 权重表。``uniform``/空 → None（旧均匀盘面行为）；``final15``
    → FINAL_MIX；否则按 JSON 字符串或文件路径解析。"""
    spec = (spec or "uniform").strip()
    if spec in ("uniform", ""):
        return None
    if spec == "final15":
        return dict(FINAL_MIX)
    if spec.startswith("{"):
        obj = json.loads(spec)
    else:
        obj = json.loads(Path(spec).read_text())
    if not isinstance(obj, dict) or not obj:
        raise ValueError(f"配比 JSON 须为非空对象: {spec}")
    out = {str(k): float(v) for k, v in obj.items()}
    if any(v < 0 for v in out.values()) or sum(out.values()) <= 0:
        raise ValueError(f"配比权重须非负且不全为零: {out}")
    return out


def distribute_int(total: int, sizes: dict[str, int]) -> dict[str, int]:
    """把 total 按 sizes 比例分成整数（最大余数法）；键集与 sizes 一致。"""
    keys = list(sizes)
    if not keys:
        return {}
    tot_size = sum(sizes.values())
    if tot_size <= 0:
        eq, r = divmod(total, len(keys))
        return {k: eq + (1 if i < r else 0) for i, k in enumerate(keys)}
    raw = {k: total * sizes[k] / tot_size for k in keys}
    floors = {k: int(np.floor(raw[k])) for k in keys}
    rem = total - sum(floors.values())
    for k in sorted(keys, key=lambda x: -(raw[x] - floors[x]))[: max(rem, 0)]:
        floors[k] += 1
    return floors


def allocate_epoch_windows(
    available: dict[str, int],
    weights: dict[str, float] | None,
    cap_epochs: float = 2.0,
    bisections: int = 200,
) -> dict[str, int]:
    """每 epoch 各源访问窗数（见模块 docstring）。权重为 None → 各源全量一遍。"""
    avail = {s: int(v) for s, v in available.items() if int(v) > 0}
    if not avail:
        return {}
    names = sorted(avail)
    total = sum(avail.values())
    if cap_epochs < 1.0:
        raise ValueError(f"cap_epochs ≥ 1 才可能凑满一个 epoch（得到 {cap_epochs}）")
    if weights is None:
        return dict(avail)
    wsum = sum(max(float(weights.get(s, 0.0)), 0.0) for s in names)
    if wsum <= 0:
        raise ValueError(f"权重表在可用源 {names} 上全为零——这些 shard 将永远不会被访问")
    w = {s: max(float(weights.get(s, 0.0)), 0.0) / wsum for s in names}
    caps = {s: cap_epochs * avail[s] for s in names}

    def f(e: float) -> float:
        return sum(min(caps[s], w[s] * e) for s in names)

    lo, hi = 0.0, 1.0
    while f(hi) < total:
        hi *= 2.0
        if hi > 1e18:
            raise RuntimeError("allocate_epoch_windows: E 发散（cap_epochs 与权重不可行）")
    for _ in range(bisections):
        mid = 0.5 * (lo + hi)
        if f(mid) < total:
            lo = mid
        else:
            hi = mid
    e_star = 0.5 * (lo + hi)
    raw = {s: min(caps[s], w[s] * e_star) for s in names}
    alloc = distribute_int(total, {s: max(raw[s], 1e-9) for s in names})
    capped = [s for s in names if raw[s] >= caps[s] - 0.5]
    return _with_meta(alloc, capped, raw)


def _with_meta(alloc: dict[str, int], capped: list[str], raw: dict[str, float]) -> dict[str, int]:
    # 把截断源挂到键 "__capped__" 供调用方记录（非 int 值，调用方自行 pop）
    out = dict(alloc)
    out["__capped__"] = capped  # type: ignore[assignment]
    out["__raw__"] = {k: float(v) for k, v in raw.items()}  # type: ignore[assignment]
    return out


def build_epoch_plan(
    metas_mine: list[dict],
    weights: dict[str, float] | None,
    window: int,
    rng: np.random.Generator,
    cap_epochs: float = 2.0,
) -> tuple[list[tuple[int, list[int]]], dict]:
    """rank 私有 shard 集 → 一个 epoch 的访问计划。

    返回 (plan, info)：plan 为 ``[(shard 全局下标, [窗下标…]), …]``（shard 粒度
    洗牌，加载顺序友好）；info 含各源配额/截断标记，供落盘记录。
    """
    by_source: dict[str, list[tuple[int, int]]] = {}
    for gi, m in enumerate(metas_mine):
        n_win = int(m["n_pos"]) // window
        if n_win > 0:
            by_source.setdefault(str(m["source"]), []).append((gi, n_win))
    avail = {s: sum(n for _, n in v) for s, v in by_source.items()}
    alloc_meta = allocate_epoch_windows(avail, weights, cap_epochs)
    capped = list(alloc_meta.pop("__capped__", []))
    raw = alloc_meta.pop("__raw__", {})
    alloc = {s: int(v) for s, v in alloc_meta.items()}

    plan: list[tuple[int, list[int]]] = []
    for s, shards in by_source.items():
        k = int(alloc.get(s, 0))
        if k <= 0:
            continue
        per = distribute_int(k, {gi: n for gi, n in shards})
        for gi, n in shards:
            take = int(per.get(gi, 0))
            if take <= 0:
                continue
            if take <= n:
                wins = rng.choice(n, size=take, replace=False)
            else:  # 超过自有窗数 → 整遍 + 无放回余量（每窗访问次数均衡：⌈take/n⌉ 或 ⌊⌋）
                full, rem = divmod(take, n)
                parts = [np.arange(n)] * full
                if rem:
                    parts.append(rng.choice(n, size=rem, replace=False))
                wins = np.concatenate(parts)
            plan.append((gi, [int(x) for x in wins]))
    rng.shuffle(plan)  # type: ignore[arg-type]
    info = {
        "available": avail,
        "alloc": alloc,
        "alloc_raw": raw,
        "capped": capped,
        "n_windows_epoch": int(sum(len(w) for _, w in plan)),
    }
    return plan, info
