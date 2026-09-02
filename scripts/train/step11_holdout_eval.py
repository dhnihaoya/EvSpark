"""Step 11 补遗：NCBI 留物种窗事后评估（plans/13 §2 评估口径的补齐）。

背景：长训启动时 holdout 仅 salmonella 落盘（训练内 eval 覆盖其 2 窗）；
bacillus/bacteroides 补抓较晚，未进训练 eval packs。本脚本对训练完成的
checkpoint 按**完全同协议**（holdout_eval_window 2048 末对齐窗、cap=256 锚点、
解析 ᾱ/τ̂、top_k=4 截断对齐）补评全部 3 个留物种，并给 Markov k=5 地板对照；
salmonella 数值应与训练 json 末次 eval 的 per_name 完全一致（一致性校验）。

结果合并进 ``benchmarks/step11_retrain.json`` 的 ``holdout_posthoc_eval`` 节。

用法（训练完成后，GPU 空闲卡）::

    CUDA_VISIBLE_DEVICES=2 HF_HOME=./hf_home \\
      /home/dh/miniconda3/envs/evo2/bin/python -u scripts/train/step11_holdout_eval.py \\
      --ckpt benchmarks/phasec_ckpts/S3_d1024_g12_step11.pt \\
      --ckpt benchmarks/phasec_ckpts/S3_d1024_g7_step11.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / "hf_home"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import torch

from specdec.markov import DEFAULT_ALPHA, MarkovDraftModel, MarkovTable
from train.data import load_default_corpus, load_ncbi_chunks, load_og2_pool, split_corpus
from train.drafter import SCHEME_LAYERS, Drafter
from train.distill import (
    build_eval_packs,
    eval_drafter,
    eval_markov,
    load_evo2,
    log,
    seed_all,
)
from train.step11_retrain import HOLDOUT_EVAL_CHUNKS_PER_SPECIES, build_holdout_eval_records

REPO = Path(__file__).resolve().parents[2]
TRAIN_JSON = REPO / "benchmarks" / "step11_retrain.json"
SEED = 20260821
CAP_ANCHORS = 256


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 11 留物种窗事后评估")
    p.add_argument("--ckpt", type=str, action="append", required=True,
                   help="可重复；相对仓库根或绝对路径")
    p.add_argument("--results", type=str, default=str(TRAIN_JSON))
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    seed_all(SEED)
    device = torch.device("cuda:0")

    holdout_recs = build_holdout_eval_records()
    n_species = len({r.source for r in holdout_recs})
    log(f"留物种评估窗：{len(holdout_recs)} 窗 / {n_species} 物种"
        f"（每物种 {HOLDOUT_EVAL_CHUNKS_PER_SPECIES} chunk）")
    if n_species < 3:
        log(f"警告：留物种数 {n_species} < 3（plans/13 要求 2–3 个）")

    evo = load_evo2(use_kernels=True)
    log("构建 Markov k=5 地板（与训练同混合语料）")
    hg_train, _ = split_corpus(load_default_corpus())
    ncbi = load_ncbi_chunks("train")
    og2 = load_og2_pool(seed=SEED)
    table = MarkovTable.build(
        [r.seq for r in hg_train + ncbi + og2], k=5, alpha=DEFAULT_ALPHA
    )
    markov_model = MarkovDraftModel(table)

    with open(args.results) as fh:
        results = json.load(fh)
    posthoc = results.setdefault("holdout_posthoc_eval", {})

    for ckpt_arg in args.ckpt:
        ckpt_path = ckpt_arg if os.path.isabs(ckpt_arg) else str(REPO / ckpt_arg)
        tag = Path(ckpt_path).stem
        ck = torch.load(ckpt_path, map_location="cpu")
        scheme, gamma = ck["scheme"], int(ck["gamma"])
        layers = SCHEME_LAYERS[scheme]
        log(f"=== {tag}: scheme={scheme} γ={gamma} 注入层={layers} ===")
        packs = build_eval_packs(evo, holdout_recs, 2048, device, tuple(layers))

        drafter = Drafter.from_scheme(scheme, d_model=ck["d_model"], gamma=gamma)
        drafter.load_state_dict(ck["state_dict"])
        drafter = drafter.to(device)
        drafter.eval()
        for p in drafter.parameters():
            p.requires_grad_(False)
        del ck

        ev = eval_drafter(drafter, packs, scheme, gamma, device, CAP_ANCHORS)
        floor = eval_markov(markov_model, packs, gamma, CAP_ANCHORS)
        log(f"{tag} 留物种 τ̂={ev['tau_hat']:.4f}（Markov 地板 {floor['tau_hat']:.4f}）")
        for name, per in sorted(ev["per_name"].items()):
            log(f"  {name}: τ̂={per['tau_hat']:.4f} ᾱ1={per['alpha_bar'][0]:.3f}")

        # 一致性校验：salmonella 窗应与训练 json 末次 eval 的 per_name 一致
        consistency = {}
        for cell in results.get("cells", []):
            if cell.get("tag") != tag:
                continue
            per_name = (cell.get("eval") or {}).get("per_name") or {}
            for name, per in per_name.items():
                if not name.startswith("ncbi_holdout_"):
                    continue
                mine = (ev["per_name"].get(name) or {}).get("tau_hat")
                consistency[name] = {
                    "train_eval_tau_hat": per.get("tau_hat"),
                    "posthoc_tau_hat": mine,
                    "match": (mine is not None and abs(mine - per["tau_hat"]) < 1e-9),
                }
        if consistency:
            log(f"一致性（训练 eval vs 事后）: {consistency}")

        posthoc[tag] = {
            "ckpt": ckpt_path,
            "scheme": scheme,
            "gamma": gamma,
            "cap_anchors": CAP_ANCHORS,
            "n_windows": len(packs),
            "species": sorted({p.name.rsplit(":", 1)[0] for p in packs}),
            "drafter_eval": ev,
            "markov_floor": floor,
            "consistency_vs_train_eval": consistency,
        }
        tmp = Path(args.results).with_suffix(".json.tmp")
        with tmp.open("w") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        tmp.replace(args.results)
        log(f"{tag} 已合并进 {args.results}")
        del drafter
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
