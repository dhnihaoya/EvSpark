"""Step 11：多物种编码区语料加训（plans/13_step11_multispecies_retrain.md §2）。

复用 ``train.distill`` 在线蒸馏管线与 Step 8/9 评估协议（18 个 hold-out 窗、
解析 ᾱ/τ̂、top_k=4 截断对齐、冻结纪律、同种子 20260821），改动只在数据侧：

- 训练语料 = 三源加权混合（``train.data.make_mixed_sampler``）：
  NCBI 多物种编码区（``data/ncbi_coding/train``，≥50%）
  + hg38 窗口 + dna_samples（Step 8/9 同一切分，~30%）
  + OG2 完整 jsonl chunk 子抽样拼接（ncrna train chunk4 / promoters train chunk1，~20%）
- 评估 = 18 标准窗 + NCBI 留物种窗（salmonella / bacillus / bacteroides 各 2 chunk）
  + lacZ 单点（含于 18 窗）
- 格子：γ=12（Step 9 峰值）20M 位置为主；γ=7 15M 位置附格保持与 Step 10 网格直接可比。
  同种子全新训练（数据分布变了，cosine 一次到位，不从 5M ckpt 续训）。

用法（必须可见设备为物理 GPU2）::

    CUDA_VISIBLE_DEVICES=2 HF_HOME=./hf_home \\
      /home/dh/miniconda3/envs/evo2/bin/python -u scripts/train/step11_retrain.py

冒烟：``--cells S3_d1024_g12_step11 --max-positions-override 100000 \\
  --results /tmp/step11_smoke.json --ckpt-dir /tmp/step11_ckpts``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / "hf_home"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import torch

from specdec.markov import DEFAULT_ALPHA, MarkovDraftModel, MarkovTable
from train.data import (
    LabeledSeq,
    load_default_corpus,
    load_ncbi_chunks,
    load_og2_pool,
    make_mixed_sampler,
    split_corpus,
)
from train.drafter import MASK_ID, TARGET_HIDDEN, VOCAB_SIZE
from train.distill import (
    ALL_INJECT_LAYERS,
    STEP6_TAU_LACZ,
    assert_layers_exist,
    build_eval_packs,
    eval_markov,
    load_evo2,
    log,
    pick_eval_records,
    save_json,
    seed_all,
    train_one_cell,
)

REPO = Path(__file__).resolve().parents[2]
RESULTS_PATH = REPO / "benchmarks" / "step11_retrain.json"
CKPT_DIR = REPO / "benchmarks" / "phasec_ckpts"
NCBI_DIR = REPO / "data" / "ncbi_coding"

# 混合比例（plans/13 §2）：NCBI 编码区 ≥50% + hg38 ~30% + OG2 ~20%
MIX_WEIGHTS = {"ncbi": 0.50, "hg38": 0.30, "og2": 0.20}

CELLS: list[dict] = [
    {"tag": "S3_d1024_g12_step11", "scheme": "S3", "d_model": 1024, "gamma": 12,
     "max_positions": 20_000_000, "purpose": "主格：γ=12（Step 9 τ̂ 峰值），多物种混合语料"},
    {"tag": "S3_d1024_g7_step11", "scheme": "S3", "d_model": 1024, "gamma": 7,
     "max_positions": 15_000_000, "purpose": "附格：γ=7，与 Step 10 网格（γ=7）直接可比"},
]

# 留物种泛化窗：每个 holdout 物种取前 2 个 chunk 进 eval（整体未参与训练）
HOLDOUT_EVAL_CHUNKS_PER_SPECIES = 2


def build_pools(seed: int) -> dict[str, list[LabeledSeq]]:
    """三源训练池。hg38 池沿用 Step 8/9 同一切分（90% 前缀，18 窗协议要求）；
    NCBI/OG2 池不切分——其评估只走按物种整体留出的 holdout 窗，无泄漏面。"""
    hg_train, _ = split_corpus(load_default_corpus())
    ncbi = load_ncbi_chunks("train")
    og2 = load_og2_pool(seed=seed)
    pools = {"ncbi": ncbi, "hg38": hg_train, "og2": og2}
    for name, pool in pools.items():
        if not pool:
            raise RuntimeError(f"训练池 {name} 为空")
        log(f"池 {name}: {len(pool)} 条记录，{sum(len(r.seq) for r in pool):,} bp")
    return pools


def build_holdout_eval_records() -> list[LabeledSeq]:
    by_species: dict[str, list[LabeledSeq]] = {}
    for rec in load_ncbi_chunks("holdout"):
        by_species.setdefault(rec.source.rsplit("/", 1)[-1], []).append(rec)
    out: list[LabeledSeq] = []
    for species in sorted(by_species):
        for rec in by_species[species][:HOLDOUT_EVAL_CHUNKS_PER_SPECIES]:
            out.append(LabeledSeq(name=f"ncbi_holdout_{rec.name}", seq=rec.seq, source=rec.source))
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 11 多物种编码区加训")
    p.add_argument("--cells", type=str, default=None, help="逗号分隔格子 tag 子集；默认 2 格")
    p.add_argument("--max-positions-override", type=int, default=None, help="冒烟用")
    p.add_argument("--eval-every", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=20260821)
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--n-anchors", type=int, default=64)
    p.add_argument("--min-ctx", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ctx-window", type=int, default=64)
    p.add_argument("--n-eval-hg", type=int, default=16)
    p.add_argument("--cap-eval-anchors", type=int, default=256)
    p.add_argument("--no-kernels", action="store_true")
    p.add_argument("--results", type=str, default=str(RESULTS_PATH))
    p.add_argument("--ckpt-dir", type=str, default=str(CKPT_DIR))
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA（plans/13 约定 GPU2）")
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    log(
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} "
        f"name={props.name} mem={props.total_memory / 1e9:.1f}GB"
    )
    if props.total_memory < 30e9:
        log("警告：可见卡 <30GB，可能不是 GPU2（48GB）")

    cells = list(CELLS)
    if args.cells:
        want = {t.strip() for t in args.cells.split(",") if t.strip()}
        cells = [c for c in cells if c["tag"] in want]
        missing = want - {c["tag"] for c in cells}
        if missing:
            raise RuntimeError(f"未知格子 tag: {sorted(missing)}")
    if not cells:
        raise RuntimeError("没有选中任何格子")
    if args.max_positions_override is not None:
        for c in cells:
            c["max_positions"] = int(args.max_positions_override)
    log(f"待跑格子 {[c['tag'] for c in cells]}")

    seed_all(args.seed)
    pools = build_pools(args.seed)
    counts: Counter[str] = Counter()
    sampler = make_mixed_sampler(pools, MIX_WEIGHTS, counts)
    manifest_summary = None
    manifest_path = NCBI_DIR / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open() as fh:
            manifest_summary = json.load(fh).get("summary")

    evo = load_evo2(use_kernels=not args.no_kernels)
    assert_layers_exist(evo, ALL_INJECT_LAYERS)

    eval_recs = pick_eval_records(n_hg=args.n_eval_hg)
    holdout_recs = build_holdout_eval_records()
    eval_recs = list(eval_recs) + holdout_recs
    log(f"预计算 eval teacher（{len(eval_recs)} 窗 = 18 标准 + {len(holdout_recs)} 留物种）")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    eval_packs = build_eval_packs(evo, eval_recs, args.window, device, ALL_INJECT_LAYERS)

    log("构建 Markov k=5 地板（混合训练语料全量）")
    all_train = [r for pool in pools.values() for r in pool]
    t0 = time.perf_counter()
    table = MarkovTable.build([r.seq for r in all_train], k=5, alpha=DEFAULT_ALPHA)
    log(f"Markov 表构建 {time.perf_counter() - t0:.1f}s（{len(all_train)} 条）")
    markov_model = MarkovDraftModel(table)
    gammas = sorted({c["gamma"] for c in cells})
    floors: dict[int, dict] = {}
    for g in gammas:
        floors[g] = eval_markov(markov_model, eval_packs, g, args.cap_eval_anchors)
        log(f"Markov 地板 γ={g} τ̂={floors[g].get('tau_hat')}")

    results = {
        "env": {
            "torch": torch.__version__,
            "gpu": props.name,
            "gpu_mem_gb": round(props.total_memory / 1e9, 1),
            "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "hf_home": os.environ.get("HF_HOME"),
            "mask_id": MASK_ID,
            "target_hidden": TARGET_HIDDEN,
            "vocab": VOCAB_SIZE,
            "seed": args.seed,
            "window": args.window,
            "n_anchors": args.n_anchors,
            "ctx_window": args.ctx_window,
            "lr": args.lr,
            "eval_every": args.eval_every,
            "pt_transform": {"temperature": 1.0, "top_k": 4},
            "step6_lacz_markov_tau": STEP6_TAU_LACZ,
            "step9_anchor": "S3/d1024/γ7/5M τ̂=4.250（hg38 单域语料，benchmarks/step9_layer_gamma_sweep.json）",
            "mix_weights": MIX_WEIGHTS,
            "pool_stats": {
                name: {"n_records": len(pool), "bp": sum(len(r.seq) for r in pool)}
                for name, pool in pools.items()
            },
            "ncbi_manifest_summary": manifest_summary,
            "holdout_eval_species": sorted({r.source for r in holdout_recs}),
        },
        "eval_names": [p.name for p in eval_packs],
        "markov_floor_by_gamma": {str(g): floors[g] for g in gammas},
        "cells": [],
        "planned_cells": [c["tag"] for c in cells],
        "status": "running",
    }
    results_path = Path(args.results)
    save_json(results_path, results)

    t_all = time.perf_counter()
    for cell in cells:
        counts_before = dict(counts)
        out = train_one_cell(
            evo=evo,
            train_recs=pools["hg38"],  # 仅占位：window_sampler 已接管
            eval_packs=eval_packs,
            markov_metrics=floors[cell["gamma"]],
            scheme=cell["scheme"],
            d_model=cell["d_model"],
            gamma=cell["gamma"],
            window=args.window,
            n_anchors=args.n_anchors,
            min_ctx=args.min_ctx,
            max_positions=cell["max_positions"],
            lr=args.lr,
            eval_every=args.eval_every,
            seed=args.seed,
            device=device,
            ckpt_dir=Path(args.ckpt_dir),
            cap_eval_anchors=args.cap_eval_anchors,
            ctx_window=args.ctx_window,
            ckpt_name=f"{cell['tag']}.pt",
            window_sampler=sampler,
        )
        out["tag"] = cell["tag"]
        out["purpose"] = cell["purpose"]
        delta = {k: counts.get(k, 0) - counts_before.get(k, 0) for k in pools}
        tot = sum(delta.values()) or 1
        out["sampler_actual_window_frac"] = {k: v / tot for k, v in sorted(delta.items())}
        results["cells"].append(out)
        save_json(results_path, results)
        log(
            f"格子 {cell['tag']} 落盘（{len(results['cells'])}/{len(cells)}）"
            f" 实际采样窗占比 {out['sampler_actual_window_frac']}"
        )

    results["status"] = "done"
    results["elapsed_s"] = time.perf_counter() - t_all
    save_json(results_path, results)
    log(f"全部格子完成，已写入 {results_path}")
    return results


if __name__ == "__main__":
    run(parse_args())
