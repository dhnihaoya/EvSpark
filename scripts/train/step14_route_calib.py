"""Step 14 路由阈值标定（plans/16 后处理）：神经 conf vs Markov 查表置信。

在留出评估窗（Step 8–13 同 28 窗协议）上以 **teacher-forced** 口径逐锚点计算：

- 神经臂：trunk + ``serial_sample(forced_tokens=真值续延)`` → conf（路由信号）与
  q′ 行；用 pack 内 p_t（top_k=4 口径）算逐位解析 ᾱ^neural。
- Markov 臂：k=5 查表沿真值续延逐位行 maxprob（路由信号）与 q′ 行 → ᾱ^markov。

路由规则 ``markov_score > neural_score + θ``（两臂得分均为逐位均值）；扫 θ 网格，
以池化解析 τ̂（1+Σ_j Π_{i≤j} ᾱ_i）最大者定 θ*。同时给出纯神经/纯 Markov/逐锚点
oracle 三根参照线与分窗明细。teacher-forced 与 decode 自草稿存在口径差（Step 10
已记录一位滞后效应），θ* 的最终效力以 C.5 端到端实测为准。

Markov 表语料 = Step 14 部署混合（gtdb 4 chunk + euk 0.10，留出纪律同训练），
落盘缓存 ``benchmarks/step14_markov_table_k5.npz`` 供 C.5 复用。

用法（48G 卡）::

    CUDA_VISIBLE_DEVICES=0 python -u scripts/train/step14_route_calib.py \
      --ckpt benchmarks/phasec_ckpts/L27_d1024_g7_offline_step14.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / "hf_home"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import numpy as np
import torch

from specdec.block.loop import _q_prime_from_raw
from specdec.block.neural_draft import serial_sample, trunk_forward
from specdec.markov import MarkovDraftModel, MarkovTable
from train.distill import (
    build_eval_packs,
    holdout_anchors,
    load_evo2,
    log,
    pick_eval_records,
    save_json,
    seed_all,
)
from train.drafter import SCHEME_LAYERS, Drafter, analytic_accept
from train.step11_retrain import build_holdout_eval_records
from train.step12_data import gtdb_holdout_eval_records, load_gtdb
from train.step12_retrain import build_pools, load_pool_pin, parse_holdout_pin

REPO = Path(__file__).resolve().parents[2]
PIN_PATH = REPO / "benchmarks" / "step13_pool_pin.json"
TABLE_CACHE = REPO / "benchmarks" / "step14_markov_table_k5.npz"
RESULTS = REPO / "benchmarks" / "step14_routing_calib.json"


def build_or_load_markov(seed: int = 20260821) -> MarkovDraftModel:
    """部署混合语料的 k=5 表（带缓存）。留出纪律经 build_pools 同训练口径。"""
    if TABLE_CACHE.exists():
        data = np.load(TABLE_CACHE)
        table = MarkovTable(k=int(data["k"]), alpha=float(data["alpha"]),
                            counts=data["counts"], probs=data["probs"])
        log(f"Markov 表命中缓存 {TABLE_CACHE}")
        return MarkovDraftModel(table)
    pin = load_pool_pin(PIN_PATH)
    try:
        pools, built = build_pools(
            seed,
            4,
            pool_pin=None,
            with_euk=True,
            gtdb_holdout_pin=parse_holdout_pin(",".join(pin["gtdb_holdout_pin"])),
        )
    except RuntimeError as exc:
        # euk 无完整 chunk 时按七源配比回退（与 Step 14 管线的 euk 回退同口径）
        log(f"部署混合建池失败（{exc}）——回退 --no-euk 七源")
        pools, built = build_pools(
            seed,
            4,
            pool_pin=None,
            with_euk=False,
            gtdb_holdout_pin=parse_holdout_pin(",".join(pin["gtdb_holdout_pin"])),
        )
    all_train = [r for pool in pools.values() for r in pool]
    t0 = time.perf_counter()
    table = MarkovTable.build([r.seq for r in all_train], k=5, alpha=0.1)
    log(f"Markov 表构建 {time.perf_counter() - t0:.1f}s（{len(all_train)} 条）")
    np.savez(TABLE_CACHE, k=table.k, alpha=table.alpha, counts=table.counts, probs=table.probs)
    return MarkovDraftModel(table)


@torch.no_grad()
def anchor_rows(pack, drafter, markov, layers: tuple[str, ...], gamma: int, a: int, device) -> dict:
    """单锚点两臂：teacher-forced 的 (ᾱ[γ], score) × 2。"""
    ids_np = pack.ids
    true_cont = ids_np[a + 1 : a + 1 + gamma]
    pt_rows = torch.as_tensor(pack.p_t[a : a + gamma], dtype=torch.float32, device=device)

    # 神经臂：trunk（训练口径 ctx_lens=a+1）+ forced 串行
    h_raw = torch.cat(
        [torch.as_tensor(pack.hidden[ln], dtype=torch.float32, device=device) for ln in layers],
        dim=-1,
    )
    anchor = torch.tensor([int(ids_np[a])], dtype=torch.long, device=device)
    ctx_lens = torch.tensor([a + 1], dtype=torch.long, device=device)
    U, h = trunk_forward(drafter, anchor, h_raw, ctx_lens)
    _rng = np.random.default_rng(0)  # forced 不消耗
    _t, q_n, confs = serial_sample(
        drafter, U, h, int(ids_np[a]), False, _rng,
        temperature=1.0, top_k=4, forced_tokens=true_cont,
    )
    alpha_n = analytic_accept(torch.as_tensor(q_n, dtype=torch.float32, device=device), pt_rows)

    # Markov 臂：沿真值续延逐位查表
    prefix = ids_np[: a + 1].astype(np.int64)
    q_m = np.empty((gamma, q_n.shape[1]), dtype=np.float64)
    tops = np.empty(gamma, dtype=np.float64)
    for k in range(gamma):
        row = np.asarray(markov.probs(prefix), dtype=np.float64)
        tops[k] = float(row.max())
        q_m[k] = _q_prime_from_raw(row, temperature=1.0, top_k=4)
        prefix = np.append(prefix, int(true_cont[k]))
    alpha_m = analytic_accept(torch.as_tensor(q_m, dtype=torch.float32, device=device), pt_rows)

    return {
        "alpha_neural": alpha_n.detach().cpu().numpy(),
        "alpha_markov": alpha_m.detach().cpu().numpy(),
        "neural_score": float(np.mean(confs)),
        "markov_score": float(np.mean(tops)),
    }


def tau_of(alpha: np.ndarray) -> float:
    prod = 1.0
    acc = 1.0
    for a in alpha:
        prod *= float(a)
        acc += prod
    return acc


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA")
    device = torch.device("cuda:0")
    ck = torch.load(args.ckpt, map_location="cpu")
    scheme = ck["scheme"]
    layers = SCHEME_LAYERS[scheme]
    drafter = Drafter.from_scheme(scheme, d_model=ck["d_model"], gamma=int(ck["gamma"]))
    drafter.load_state_dict(ck["state_dict"])
    drafter = drafter.to(device).eval()
    for p in drafter.parameters():
        p.requires_grad_(False)
    gamma = int(ck["gamma"])
    log(f"ckpt={args.ckpt} scheme={scheme} γ={gamma} n_pos={ck.get('n_pos')}")

    seed_all(args.seed)
    evo = load_evo2()
    pin = load_pool_pin(PIN_PATH)
    gtdb_meta = load_gtdb(
        seed=args.seed,
        only=set(pin["chunks"]["gtdb_v220_imgpr"]),
        holdout_pin=parse_holdout_pin(",".join(pin["gtdb_holdout_pin"])),
    )[1]
    eval_recs = (
        list(pick_eval_records(n_hg=16))
        + list(build_holdout_eval_records())
        + list(gtdb_holdout_eval_records(gtdb_meta))
    )
    packs = build_eval_packs(evo, eval_recs, 2048, device, layers)
    del evo
    torch.cuda.empty_cache()
    markov = build_or_load_markov(args.seed)

    # 逐锚点收集
    rows: list[dict] = []
    t0 = time.perf_counter()
    for pack in packs:
        anchors = holdout_anchors(int(pack.ids.size), pack.holdout_start, gamma, args.cap_anchors)
        for a in anchors:
            r = anchor_rows(pack, drafter, markov, layers, gamma, int(a), device)
            r["name"] = pack.name
            rows.append(r)
    log(f"锚点收集完成 {len(rows)} 个，{time.perf_counter() - t0:.0f}s")

    thetas = np.linspace(args.theta_lo, args.theta_hi, args.n_theta)
    # 预计算每锚点两臂 τ
    for r in rows:
        r["tau_neural"] = tau_of(r["alpha_neural"])
        r["tau_markov"] = tau_of(r["alpha_markov"])
    sweep = []
    for th in thetas:
        taus = [
            r["tau_markov"] if r["markov_score"] > r["neural_score"] + th else r["tau_neural"]
            for r in rows
        ]
        sweep.append({"theta": float(th), "tau_pooled": float(np.mean(taus))})
    best = max(sweep, key=lambda r: r["tau_pooled"])
    theta_star = best["theta"]

    def region_breakdown(theta: float | None) -> dict:
        by: dict[str, list[float]] = {}
        for r in rows:
            if theta is None:  # oracle
                t = max(r["tau_neural"], r["tau_markov"])
            else:
                t = r["tau_markov"] if r["markov_score"] > r["neural_score"] + theta else r["tau_neural"]
            by.setdefault(r["name"], []).append(t)
        return {k: float(np.mean(v)) for k, v in sorted(by.items())}

    n_markov = sum(1 for r in rows if r["markov_score"] > r["neural_score"] + theta_star)
    out = {
        "ckpt": args.ckpt,
        "scheme": scheme,
        "gamma": gamma,
        "n_anchors": len(rows),
        "theta_star": theta_star,
        "tau_at_star": best["tau_pooled"],
        "tau_pure_neural": float(np.mean([r["tau_neural"] for r in rows])),
        "tau_pure_markov": float(np.mean([r["tau_markov"] for r in rows])),
        "tau_oracle": float(np.mean([max(r["tau_neural"], r["tau_markov"]) for r in rows])),
        "markov_share_at_star": n_markov / max(len(rows), 1),
        "sweep": sweep,
        "per_window_at_star": region_breakdown(theta_star),
        "per_window_pure_neural": region_breakdown(float("inf")),
        "per_window_oracle": region_breakdown(None),
        "protocol": (
            "teacher-forced 解析 ᾱ/τ̂（top_k=4 对齐）；路由信号=conf 均值 vs 查表 maxprob 均值；"
            "Markov 表=部署混合（gtdb4+euk0.10，留出同训练）；decode 自草稿口径差以 C.5 为准"
        ),
        "status": "done",
    }
    save_json(Path(args.results), out)
    log(
        f"θ*={theta_star:.3f} 池化 τ̂ {out['tau_pure_neural']:.4f}(纯神经) → "
        f"{best['tau_pooled']:.4f}(路由) [oracle {out['tau_oracle']:.4f}，纯 Markov "
        f"{out['tau_pure_markov']:.4f}] Markov 占比 {out['markov_share_at_star']:.1%} → {args.results}"
    )
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--cap-anchors", type=int, default=256)
    p.add_argument("--theta-lo", type=float, default=-0.4)
    p.add_argument("--theta-hi", type=float, default=0.4)
    p.add_argument("--n-theta", type=int, default=81)
    p.add_argument("--seed", type=int, default=20260821)
    p.add_argument("--results", type=str, default=str(RESULTS))
    return p.parse_args(argv)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
