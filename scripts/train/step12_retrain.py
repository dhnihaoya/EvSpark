"""Step 12：部分 OG2 正式档加训（plans/14_step12_partial_og2_retrain.md §2）。

复用 ``train.distill`` 在线蒸馏管线与 Step 8–11 评估协议（18 标准 hold-out 窗、
解析 ᾱ/τ̂、top_k=4 截断对齐、冻结纪律、同种子 20260821），改动只在数据侧与规模：

- 训练语料 = 七源加权混合（``train.data.make_mixed_sampler`` + ``step12_data``）：
  NCBI 多物种编码区 15% + gtdb_v220_imgpr 细菌基因组 20%（合细菌侧 35%）
  + mrna_splice（真核基因区）35% + ncrna/organelle/promoters 合 15% + hg38 15%
- 评估 = 18 标准窗 + NCBI 留物种窗 + gtdb 留基因组窗（新，每家族最长 2 contig）
- 格子：单格 S3={blocks.6,16,27} + d=1024 + ctx_window=64 + **γ=7**（Step 11
  真实 τ 口径胜者），50M 位置，同种子全新训练，cosine 一次到位
- 数据构成与完整性校验（chunk 清单 / keep_prob / gtdb 留出基因组 / ORF prompt）
  全部落盘 ``benchmarks/step12_retrain.json``，ORF prompt 供复测脚本读取

用法（Step 12 起走 DDP 双卡启动器，每 rank 独占一张 48G 卡，默认 GPU0+GPU2；
evo2/vortex 会把模型铺到所有可见卡，**禁止** torchrun 共享可见设备直起）::

    bash scripts/train/step12_launch_ddp.sh            # 50M 长训
    GPUS="2" WORLD_SIZE=1 bash scripts/train/step12_launch_ddp.sh   # 单卡回退

单卡直跑（旧行为，默认可见 GPU2）::

    CUDA_VISIBLE_DEVICES=2 HF_HOME=./hf_home \\
      /home/dh/miniconda3/envs/evo2/bin/python -u scripts/train/step12_retrain.py

冒烟：``--max-positions-override 100000 --results /tmp/step12_smoke.json \\
  --ckpt-dir /tmp/step12_ckpts``（gtdb chunk 未就绪时可加 ``--gtdb-chunks 0``）
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
from train.data import LabeledSeq, load_default_corpus, load_ncbi_chunks, make_mixed_sampler, split_corpus
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
from train.drafter import MASK_ID, TARGET_HIDDEN, VOCAB_SIZE
from train.step11_retrain import build_holdout_eval_records
from train.step12_data import (
    TARGET_BP,
    build_gtdb_prompt_meta,
    gtdb_holdout_eval_records,
    load_gtdb,
    load_og2_source,
    scan_inventory,
)

REPO = Path(__file__).resolve().parents[2]
RESULTS_PATH = REPO / "benchmarks" / "step12_retrain.json"
CKPT_DIR = REPO / "benchmarks" / "phasec_ckpts"

# 混合配比（plans/14 §2）：细菌编码（NCBI+gtdb）~35% / mrna_splice ~35% /
# ncrna+organelle+promoters ~15% / hg38 ~15%；--gtdb-chunks 0 时 gtdb 权重回摊
MIX_WEIGHTS = {
    "ncbi": 0.15,
    "gtdb": 0.20,
    "mrna": 0.35,
    "ncrna": 0.08,
    "organelle": 0.04,
    "promoters": 0.03,
    "hg38": 0.15,
}

CELLS: list[dict] = [
    {"tag": "S3_d1024_g7_step12", "scheme": "S3", "d_model": 1024, "gamma": 7,
     "max_positions": 50_000_000,
     "purpose": "单格：γ=7（Step 11 真实 τ 胜者），OG2 正式档多域混合，50M 位置"},
]

# Step 13 取层复扫格（plans/15 §3）：C.1 落盘前的多种子证据加固。
# 仅经 --cells 显式选中（不进默认集合）；per-cell ``seed`` 覆盖全局 --seed；
# 单卡运行（plans/15 原约束 GPU2 单卡；Step 12 暂停期间 GPU0 并行跑另一半，
# 每格仍是单卡协议）。10M 位置/格，判定规则见 plans/15 §3。
STEP13_CELLS: list[dict] = [
    {"tag": "L27_d1024_g7_10M_s1", "scheme": "L27", "d_model": 1024, "gamma": 7,
     "max_positions": 10_000_000, "seed": 20260821,
     "purpose": "Step 13 复扫：单层 blocks.27，种子 1"},
    {"tag": "L27_d1024_g7_10M_s2", "scheme": "L27", "d_model": 1024, "gamma": 7,
     "max_positions": 10_000_000, "seed": 20260822,
     "purpose": "Step 13 复扫：单层 blocks.27，种子 2"},
    {"tag": "S3_d1024_g7_10M_s1", "scheme": "S3", "d_model": 1024, "gamma": 7,
     "max_positions": 10_000_000, "seed": 20260821,
     "purpose": "Step 13 复扫：三层 {6,16,27}，种子 1"},
    {"tag": "S3_d1024_g7_10M_s2", "scheme": "S3", "d_model": 1024, "gamma": 7,
     "max_positions": 10_000_000, "seed": 20260822,
     "purpose": "Step 13 复扫：三层 {6,16,27}，种子 2"},
]

# Step 13b 层组合复扫（plans/16 §2）：9 scheme × 2 种子 × 3M，短预算。
# 仅经 --cells 显式选中。池必须 --pool-chunks-json 锁定，才能与 Step 13 锚点可比。
_STEP13B_SCHEMES = ("L6", "L16", "L20", "L23", "L27", "L30", "D16_27", "D27_30", "S3")
_STEP13B_SEEDS = ((1, 20260821), (2, 20260822))
STEP13B_CELLS: list[dict] = [
    {
        "tag": f"{sch}_d1024_g7_3M_s{si}",
        "scheme": sch,
        "d_model": 1024,
        "gamma": 7,
        "max_positions": 3_000_000,
        "seed": seed,
        "purpose": f"Step 13b 层组合复扫：{sch} 种子 {si}（3M 短预算）",
    }
    for sch in _STEP13B_SCHEMES
    for si, seed in _STEP13B_SEEDS
]

# Step 14 长训（plans/16 §4）：100M，只跑复扫赢家那一格（默认 L27）。
STEP14_CELLS: list[dict] = [
    {"tag": "L27_d1024_g7_100M_step14", "scheme": "L27", "d_model": 1024, "gamma": 7,
     "max_positions": 100_000_000, "seed": 20260821,
     "purpose": "Step 14 长训：单层 L27（复扫 incumbent / 默认赢家）"},
    {"tag": "S3_d1024_g7_100M_step14", "scheme": "S3", "d_model": 1024, "gamma": 7,
     "max_positions": 100_000_000, "seed": 20260821,
     "purpose": "Step 14 长训：三层 S3（复扫翻盘备用）"},
    {"tag": "D16_27_d1024_g7_100M_step14", "scheme": "D16_27", "d_model": 1024, "gamma": 7,
     "max_positions": 100_000_000, "seed": 20260821,
     "purpose": "Step 14 长训：双层 {16,27}（复扫翻盘备用）"},
]

# 复扫单层/双层赢家可能落在其余 6 个 scheme（merge 的 step14_tag 按
# f"{scheme}_d1024_g7_100M_step14" 生成）——缺格会让 Step 14 以「未知格子 tag」
# 安全失败、接力断在半路；补齐保证任何判定结果都能全自动接 Step 14。
# 注意必须在 ALL_CELL_BANKS 之前 extend。
STEP14_CELLS += [
    {"tag": f"{sch}_d1024_g7_100M_step14", "scheme": sch, "d_model": 1024, "gamma": 7,
     "max_positions": 100_000_000, "seed": 20260821,
     "purpose": f"Step 14 长训：{sch}（复扫翻盘备用）"}
    for sch in ("L6", "L16", "L20", "L23", "L30", "D27_30")
]

EUK_WEIGHT = 0.10
ALL_CELL_BANKS = CELLS + STEP13_CELLS + STEP13B_CELLS + STEP14_CELLS


def load_pool_pin(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or "chunks" not in data:
        raise RuntimeError(f"pool pin 缺少 chunks 字段: {path}")
    return data


def parse_holdout_pin(s: str | None) -> tuple[str, ...] | None:
    if not s:
        return None
    pin = tuple(x.strip() for x in str(s).split(",") if x.strip())
    return pin or None


def build_pools(
    seed: int,
    gtdb_chunks: int,
    *,
    pool_pin: dict | None = None,
    with_euk: bool = False,
    gtdb_holdout_pin: tuple[str, ...] | None = None,
) -> tuple[dict[str, list[LabeledSeq]], dict]:
    """七源（+可选 euk）训练池 + 数据构成元数据。hg38 池沿用 Step 8/9/11 同一切分。"""
    hg_train, _ = split_corpus(load_default_corpus())
    ncbi = load_ncbi_chunks("train")
    pools: dict[str, list[LabeledSeq]] = {"hg38": hg_train, "ncbi": ncbi}
    inv_keys = [k for k in TARGET_BP if k != "eukaryotic_genic_windows" or with_euk]
    meta: dict = {"inventory": scan_inventory(include=inv_keys)}
    pin_chunks = (pool_pin or {}).get("chunks") if pool_pin else None

    def only_for(sub: str) -> set[str] | None:
        if not pin_chunks or sub not in pin_chunks:
            return None
        return set(pin_chunks[sub])

    gtdb_meta: dict = {"skipped": "gtdb-chunks=0（冒烟逃生门），权重已回摊"}
    if gtdb_chunks > 0:
        gtdb, gtdb_meta = load_gtdb(
            seed=seed,
            n_chunks=gtdb_chunks,
            only=only_for("gtdb_v220_imgpr"),
            holdout_pin=gtdb_holdout_pin,
        )
        pools["gtdb"] = gtdb
    for sub, key, seed_off in (
        ("mrna_splice_promoter", "mrna", 101),
        ("ncrna", "ncrna", 202),
        ("organelle", "organelle", 303),
        ("promoters", "promoters", 404),
    ):
        pool, m = load_og2_source(sub, seed=seed + seed_off, only=only_for(sub))
        if pool:
            pools[key] = pool
            meta[key] = m
        else:
            log(f"警告：OG2 源 {sub} 无完整 train chunk，池缺席（权重回摊）")
            meta[key] = m

    if with_euk:
        euk, euk_meta = load_og2_source(
            "eukaryotic_genic_windows",
            seed=seed + 505,
            only=only_for("eukaryotic_genic_windows"),
        )
        meta["euk"] = euk_meta
        if not euk:
            raise RuntimeError(
                "euk 已请求但池为空（无完整 train chunk）。"
                "去掉 --with-euk 按原七源配比跑，或等下载完成。"
            )
        pools["euk"] = euk

    weights = {k: v for k, v in MIX_WEIGHTS.items() if pools.get(k)}
    if with_euk:
        weights = {k: v * (1.0 - EUK_WEIGHT) for k, v in weights.items()}
        weights["euk"] = EUK_WEIGHT
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}
    meta["mix_weights_effective"] = weights
    meta["gtdb"] = {k: v for k, v in gtdb_meta.items() if k != "holdout_contigs"}
    meta["gtdb_holdout"] = gtdb_meta.get("holdout", {})
    meta["gtdb_holdout_contig_stats"] = {
        fam: {"n_contigs": len(v), "bp": sum(len(s) for _, s in v)}
        for fam, v in gtdb_meta.get("holdout_contigs", {}).items()
    }
    if pool_pin:
        meta["pool_pin"] = {
            "chunks": pin_chunks,
            "gtdb_holdout_pin": list(gtdb_holdout_pin) if gtdb_holdout_pin else None,
        }

    for name, pool in pools.items():
        if not pool:
            raise RuntimeError(f"训练池 {name} 为空")
        log(f"池 {name}: {len(pool)} 条记录，{sum(len(r.seq) for r in pool):,} bp")
    return pools, {"weights": weights, "gtdb_meta": gtdb_meta, "pools_meta": meta}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 12 部分 OG2 正式档加训")
    p.add_argument("--cells", type=str, default=None, help="逗号分隔格子 tag 子集；默认单格")
    p.add_argument("--max-positions-override", type=int, default=None, help="冒烟用")
    p.add_argument("--gtdb-chunks", type=int, default=2,
                   help="gtdb_v220_imgpr 用前几个完整 chunk（用户定：2 个足够；0=跳过，权重回摊）")
    p.add_argument("--pool-chunks-json", type=str, default=None,
                   help="池锁定 JSON（Step 13b：benchmarks/step13_pool_pin.json）")
    p.add_argument("--with-euk", action="store_true",
                   help="接入 eukaryotic_genic_windows，权重 0.10，其余七源 ×0.9 再归一")
    p.add_argument("--gtdb-holdout-pin", type=str, default=None,
                   help="逗号分隔基因组家族前缀；提供时跳过 choose_holdout_prefixes")
    p.add_argument("--eval-every", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=20260821)
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--n-anchors", type=int, default=64)
    p.add_argument("--min-ctx", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ctx-window", type=int, default=64)
    p.add_argument("--compile", action="store_true",
                   help="torch.compile 包装 drafter 训练前向（评估仍 eager；数值不逐位等）")
    p.add_argument("--amp", action="store_true",
                   help="drafter 训练前向 bf16 autocast（损失保持 fp32；参数/优化器 fp32）")
    p.add_argument("--n-eval-hg", type=int, default=16)
    p.add_argument("--cap-eval-anchors", type=int, default=256)
    p.add_argument("--no-kernels", action="store_true")
    p.add_argument("--results", type=str, default=str(RESULTS_PATH))
    p.add_argument("--ckpt-dir", type=str, default=str(CKPT_DIR))
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    # torchrun 注入 RANK/WORLD_SIZE/LOCAL_RANK；直跑（单卡）时全取默认 0/1/0，
    # 行为与无 DDP 版逐行等价。两张 48G 卡（GPU0+GPU2）时由外层设
    # CUDA_VISIBLE_DEVICES=0,2 后 torchrun --nproc_per_node=2 启动
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA（plans/14 约定 GPU2；DDP 时 GPU0+GPU2）")
    # evo2/vortex 会把模型平铺到**所有可见 GPU**（Evo2 docstring 明示），因此
    # DDP 必须每 rank 只见一张卡（launcher 各绑一张）；此环境下 local_rank→0
    visible = torch.cuda.device_count()
    if visible < 1:
        raise RuntimeError("没有可见 GPU")
    dev_idx = local_rank if visible > local_rank else 0

    if args.cells:
        want = {t.strip() for t in args.cells.split(",") if t.strip()}
        cells = [c for c in ALL_CELL_BANKS if c["tag"] in want]
        missing = want - {c["tag"] for c in cells}
        if missing:
            raise RuntimeError(f"未知格子 tag: {sorted(missing)}")
    else:
        cells = list(CELLS)
    if not cells:
        raise RuntimeError("没有选中任何格子")
    if args.max_positions_override is not None:
        for c in cells:
            c["max_positions"] = int(args.max_positions_override)

    pool_pin = load_pool_pin(args.pool_chunks_json) if args.pool_chunks_json else None
    holdout_pin = parse_holdout_pin(args.gtdb_holdout_pin)
    if holdout_pin is None and pool_pin is not None:
        raw_pin = pool_pin.get("gtdb_holdout_pin")
        if raw_pin:
            holdout_pin = tuple(str(x) for x in raw_pin)

    # CPU 侧重活先行：进程池 fork 必须发生在任何 CUDA/NCCL 初始化之前
    pools, built = build_pools(
        args.seed,
        args.gtdb_chunks,
        pool_pin=pool_pin,
        with_euk=args.with_euk,
        gtdb_holdout_pin=holdout_pin,
    )
    weights = built["weights"]
    gtdb_meta = built["gtdb_meta"]
    counts: Counter[str] = Counter()
    sampler = make_mixed_sampler(pools, weights, counts)

    if world > 1:
        import datetime

        import torch.distributed as dist

        # 默认 NCCL watchdog 10min：rank0 独占预计算（eval packs/Markov/prompts）
        # 期间 rank1 卡在 DDP 构造的 broadcast 上，留足余量防误杀
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30))
    device = torch.device(f"cuda:{dev_idx}")
    torch.cuda.set_device(device)
    seed_all(args.seed)
    props = torch.cuda.get_device_properties(device)
    log(
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} "
        f"rank={rank}/{world} name={props.name} mem={props.total_memory / 1e9:.1f}GB"
    )
    if props.total_memory < 30e9:
        log("警告：可见卡 <30GB，可能不是 48GB 卡")
    log(f"待跑格子 {[c['tag'] for c in cells]}")
    log(f"混合权重（生效）: { {k: round(v, 4) for k, v in weights.items()} }")
    if pool_pin:
        log(f"池锁定: {args.pool_chunks_json} holdout_pin={holdout_pin}")
    if args.with_euk:
        log("euk=0.10 已接入")

    evo = load_evo2(use_kernels=not args.no_kernels)
    assert_layers_exist(evo, ALL_INJECT_LAYERS)

    # 评估窗 / Markov 地板 / gtdb prompt 只在 rank0 预计算（teacher 同权重，
    # 结果确定性；其余 rank 训练循环内不消费 eval_packs）
    eval_packs: list = []
    floors: dict[int, dict] = {}
    gtdb_prompts: list[dict] = []
    holdout_recs: list = []
    if rank == 0:
        eval_recs = pick_eval_records(n_hg=args.n_eval_hg)
        holdout_recs = list(build_holdout_eval_records())  # NCBI 留物种窗（Step 11 同协议）
        gtdb_hold_recs = []
        if args.gtdb_chunks > 0:
            gtdb_hold_recs = gtdb_holdout_eval_records(gtdb_meta)
        eval_recs = list(eval_recs) + holdout_recs + gtdb_hold_recs
        log(
            f"预计算 eval teacher（{len(eval_recs)} 窗 = 18 标准 + "
            f"{len(holdout_recs)} NCBI 留物种 + {len(gtdb_hold_recs)} gtdb 留基因组）"
        )
        torch.cuda.reset_peak_memory_stats()
        eval_packs = build_eval_packs(evo, eval_recs, args.window, device, ALL_INJECT_LAYERS)

        log("构建 Markov k=5 地板（混合训练语料全量）")
        all_train = [r for pool in pools.values() for r in pool]
        t0 = time.perf_counter()
        table = MarkovTable.build([r.seq for r in all_train], k=5, alpha=DEFAULT_ALPHA)
        log(f"Markov 表构建 {time.perf_counter() - t0:.1f}s（{len(all_train)} 条）")
        markov_model = MarkovDraftModel(table)
        for g in sorted({c["gamma"] for c in cells}):
            floors[g] = eval_markov(markov_model, eval_packs, g, args.cap_eval_anchors)
            log(f"Markov 地板 γ={g} τ̂={floors[g].get('tau_hat')}")

        gtdb_prompts = build_gtdb_prompt_meta(gtdb_meta) if args.gtdb_chunks > 0 else []
        for gp in gtdb_prompts:
            log(
                f"gtdb 留基因组 prompt {gp['name']}: 最长 ORF {gp['orf_span_bp']}bp"
                f"@contig {gp['contig_len']}bp 窗起点 {gp['window_start']}"
            )

    results = None
    results_path = Path(args.results)
    if rank == 0:
        results = {
            "env": {
                "torch": torch.__version__,
                "gpu": props.name,
                "gpu_ranks": world,
                "ddp_note": (
                    f"world_size={world}：单步有效 batch = {args.n_anchors}锚×{world}卡，"
                    "优化器步数相应减半；lr 与总位置数不变（相对 Step 11 单卡协议的已知偏离）"
                ) if world > 1 else "单卡（与 Step 8–11 同路径）",
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
                "amp": args.amp,
                "compile": args.compile,
                "eval_every": args.eval_every,
                "pt_transform": {"temperature": 1.0, "top_k": 4},
                "step6_lacz_markov_tau": STEP6_TAU_LACZ,
                "step11_anchor": (
                    "γ=7 真实 τ：lacZ 1.62 / chr21 3.88 / #0 6.05 / #235 5.71 / 随机 5.47"
                    "（15M 位置，benchmarks/step11_retrain.json）"
                ),
                "mix_weights": weights,
                "mix_weights_planned": MIX_WEIGHTS,
                "with_euk": bool(args.with_euk),
                "euk_weight": EUK_WEIGHT if args.with_euk else 0.0,
                "pool_chunks_json": args.pool_chunks_json,
                "gtdb_holdout_pin": list(holdout_pin) if holdout_pin else None,
                "pool_stats": {
                    name: {"n_records": len(pool), "bp": sum(len(r.seq) for r in pool)}
                    for name, pool in pools.items()
                },
                "pools_meta": built["pools_meta"],
                "holdout_eval_species": sorted({r.source for r in holdout_recs}),
                "gtdb_holdout": built["pools_meta"].get("gtdb_holdout", {}),
                "gtdb_holdout_contig_stats": built["pools_meta"].get("gtdb_holdout_contig_stats", {}),
            },
            "gtdb_holdout_prompts": gtdb_prompts,
            "eval_names": [p.name for p in eval_packs],
            "markov_floor_by_gamma": {str(g): floors[g] for g in floors},
            "cells": [],
            "planned_cells": [c["tag"] for c in cells],
            "status": "running",
        }
        save_json(results_path, results)

    t_all = time.perf_counter()
    for cell in cells:
        counts_before = dict(counts)
        out = train_one_cell(
            evo=evo,
            train_recs=pools["hg38"],  # 仅占位：window_sampler 已接管
            eval_packs=eval_packs,
            markov_metrics=floors.get(cell["gamma"]),
            scheme=cell["scheme"],
            d_model=cell["d_model"],
            gamma=cell["gamma"],
            window=args.window,
            n_anchors=args.n_anchors,
            min_ctx=args.min_ctx,
            max_positions=cell["max_positions"],
            lr=args.lr,
            eval_every=args.eval_every,
            seed=int(cell.get("seed", args.seed)),  # Step 13 复扫格带 per-cell 种子
            device=device,
            ckpt_dir=Path(args.ckpt_dir),
            cap_eval_anchors=args.cap_eval_anchors,
            ctx_window=args.ctx_window,
            ckpt_name=f"{cell['tag']}.pt",
            window_sampler=sampler,
            ddp_world_size=world,
            compile_drafter=args.compile,
            amp=args.amp,
        )
        if rank == 0:
            out["tag"] = cell["tag"]
            out["purpose"] = cell["purpose"]
            out["seed"] = int(cell.get("seed", args.seed))
            delta = {k: counts.get(k, 0) - counts_before.get(k, 0) for k in pools}
            tot = sum(delta.values()) or 1
            out["sampler_actual_window_frac"] = {k: v / tot for k, v in sorted(delta.items())}
            results["cells"].append(out)
            save_json(results_path, results)
            log(
                f"格子 {cell['tag']} 落盘（{len(results['cells'])}/{len(cells)}）"
                f" 实际采样窗占比 {out['sampler_actual_window_frac']}"
            )

    if world > 1:
        import torch.distributed as dist

        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        results["status"] = "done"
        results["elapsed_s"] = time.perf_counter() - t_all
        save_json(results_path, results)
        log(f"全部格子完成，已写入 {results_path}")
    return results


if __name__ == "__main__":
    run(parse_args())
