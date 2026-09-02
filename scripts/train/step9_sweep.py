"""Step 9：单层 HCL 扫描 + γ 扫描 + 优势配置加训（plans/11_step9_layer_gamma_sweep.md）。

严格复用 ``train.distill`` 管线与评估协议（18 个 hold-out 窗、解析 ᾱ/τ̂、
top_k=4 截断对齐、冻结纪律、同种子 20260821），一次加载 7B 顺序跑 6 格：

- ``L6/L16/L27`` d=1024 γ=7 0.5M：S3 三个 HCL 各自单独注入（C.1 落盘层数决策）
- ``S3`` d=1024 γ∈{12,16} 0.5M：γ 扫描（γ=7 已有 Step 8 对照；C.3 选 γ）
- ``S3`` d=1024 γ=7 5M：优势配置加训（checkpoint 留给 Step 10 当默认 drafter）

用法（必须可见设备为物理 GPU2）::

    CUDA_VISIBLE_DEVICES=2 HF_HOME=./hf_home \\
      /home/dh/miniconda3/envs/evo2/bin/python -u scripts/train/step9_sweep.py

冒烟：``--cells L6_d1024_g7 --max-positions-override 10000 --results /tmp/...``
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / "hf_home"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import torch

from specdec.markov import DEFAULT_ALPHA, MarkovDraftModel, MarkovTable
from train.data import load_default_corpus, split_corpus
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
RESULTS_PATH = REPO / "benchmarks" / "step9_layer_gamma_sweep.json"
CKPT_DIR = REPO / "benchmarks" / "phasec_ckpts"

# 格子定义：tag = checkpoint 文件名 / json 格子代号
CELLS: list[dict] = [
    {"tag": "L6_d1024_g7", "scheme": "L6", "d_model": 1024, "gamma": 7,
     "max_positions": 500_000, "purpose": "单层 HCL blocks.6（C.1 落盘层数）"},
    {"tag": "L16_d1024_g7", "scheme": "L16", "d_model": 1024, "gamma": 7,
     "max_positions": 500_000, "purpose": "单层 HCL blocks.16（C.1 落盘层数）"},
    {"tag": "L27_d1024_g7", "scheme": "L27", "d_model": 1024, "gamma": 7,
     "max_positions": 500_000, "purpose": "单层 HCL blocks.27（C.1 落盘层数）"},
    {"tag": "S3_d1024_g12", "scheme": "S3", "d_model": 1024, "gamma": 12,
     "max_positions": 500_000, "purpose": "γ 扫描（C.3 选 γ）"},
    {"tag": "S3_d1024_g16", "scheme": "S3", "d_model": 1024, "gamma": 16,
     "max_positions": 500_000, "purpose": "γ 扫描（C.3 选 γ）"},
    {"tag": "S3_d1024_g7_5M", "scheme": "S3", "d_model": 1024, "gamma": 7,
     "max_positions": 5_000_000, "purpose": "优势配置加训（Step 10 默认 drafter 权重）"},
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 9 单层 HCL 扫描 + γ 扫描 + 加训")
    p.add_argument("--cells", type=str, default=None,
                   help="逗号分隔的格子 tag 子集；默认全部 6 格")
    p.add_argument("--max-positions-override", type=int, default=None,
                   help="覆盖所有格子的位置预算（冒烟用）")
    p.add_argument("--eval-every", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=20260821)
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--n-anchors", type=int, default=64)
    p.add_argument("--min-ctx", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ctx-window", type=int, default=64)
    p.add_argument("--n-eval-hg", type=int, default=16)
    p.add_argument("--cap-eval-anchors", type=int, default=256)
    p.add_argument("--fasta-dir", type=str, default=None)
    p.add_argument("--no-kernels", action="store_true")
    p.add_argument("--results", type=str, default=str(RESULTS_PATH))
    p.add_argument("--ckpt-dir", type=str, default=str(CKPT_DIR))
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA（plans/11 约定 GPU2）")
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
    corpus = load_default_corpus(fasta_dir=args.fasta_dir)
    train_recs, _ = split_corpus(corpus)
    log(f"训练序列 {len(train_recs)} 条，bp={sum(len(r.seq) for r in train_recs):,}")

    evo = load_evo2(use_kernels=not args.no_kernels)
    assert_layers_exist(evo, ALL_INJECT_LAYERS)

    eval_recs = pick_eval_records(n_hg=args.n_eval_hg)
    log(f"预计算 eval teacher（{len(eval_recs)} 窗，层 {ALL_INJECT_LAYERS}）")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    eval_packs = build_eval_packs(evo, eval_recs, args.window, device, ALL_INJECT_LAYERS)

    log("构建 Markov k=5 地板（训练前缀）")
    table = MarkovTable.build([r.seq for r in train_recs], k=5, alpha=DEFAULT_ALPHA)
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
            "step8_anchor": "S3/d1024/γ7/0.5M τ̂=3.749（benchmarks/phasec_smoke.json）",
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
        out = train_one_cell(
            evo=evo,
            train_recs=train_recs,
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
        )
        out["tag"] = cell["tag"]
        out["purpose"] = cell["purpose"]
        results["cells"].append(out)
        save_json(results_path, results)
        log(f"格子 {cell['tag']} 落盘（{len(results['cells'])}/{len(cells)}）")

    results["status"] = "done"
    results["elapsed_s"] = time.perf_counter() - t_all
    save_json(results_path, results)
    log(f"全部格子完成，已写入 {results_path}")
    return results


if __name__ == "__main__":
    run(parse_args())
