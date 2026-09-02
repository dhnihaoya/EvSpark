"""Step 15 主表评估集构造（plans/17 §3.1）：每区域 ≥3 条序列。

区域构成（prompt 名 → 区域）：

- coding 细菌：lacZ 3 窗（3079bp 切 1024 步进）+ gtdb 留出 2（C.5 套件自带）
- coding 噬菌体：imgvr valid 3 条（``step15_imgvr_eval.py`` 产物存在时并入）
- intergenic 人源：chr21 基因稀疏区 3 窗（4097bp）
- repeat：hg38 重复池 3 窗（w0/w235 为 C.5 自带，此处补第 3 条）
- random：3 种子
- 域覆盖列（不计入五大区域均值）：OG2 mrna/ncrna/euk/organelle/promoters 各 1

产出 ``benchmarks/step15_main_suite_extra.json`` 供
``step15_c5_suite.py --extra-prompts-json`` 消费；C.5 内置 5 prompt 由套件
自身提供，不重复。种子纪律：主表 mean±std = 区域内序列 × 训练种子（s1/s2）。

用法::

    /home/dh/miniconda3/envs/evo2/bin/python scripts/train/step15_main_suite.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "benchmarks" / "step15_main_suite_extra.json"
IMGVR_PROMPTS = REPO / "benchmarks" / "step15_imgvr_prompts.json"


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 15 主表扩展 prompt 构造")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()

    import importlib.util

    from train.data import load_dna_samples

    spec = importlib.util.spec_from_file_location(
        "bench_acceptance_mod", str(_SCRIPTS / "bench_acceptance.py")
    )
    ba = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ba)

    specs: list[dict] = []
    samples = {r.name: r.seq for r in load_dna_samples()}
    # lacZ 3 窗（w0 = C.5 自带 lacz_coding，这里补 w1/w2）
    lacZ = samples["coding_ecoli_lacZ"]
    for i, off in enumerate((1024, 2048)):
        specs.append({"name": f"lacz_w{i + 1}", "region": "coding", "seq": lacZ[off : off + args.ctx]})
    # chr21 3 窗（w0 = C.5 自带 chr21_intergenic）
    chr21 = samples["intergenic_human_chr21"]
    for i, off in enumerate((1024, 2048)):
        specs.append({"name": f"chr21_w{i + 1}", "region": "intergenic",
                      "seq": chr21[off : off + args.ctx]})
    # repeat 第 3 窗：w0（端粒强制）/次低复杂度窗由 C.5 自带；从 ranked_lowest10
    # 顺位再取一条（避开前两者），seq 从 windows 映射取
    hg38 = ba.load_hg38_windows(ba.HG38_CSV)
    sel = ba.select_repeat_windows(hg38, args.ctx)
    taken = {s["index"] for s in sel["selected"]}
    third = next((r for r in sel["ranked_lowest10"] if r["index"] not in taken), None)
    if third is not None:
        by_i = {w["index"]: w for w in hg38}
        specs.append(
            {
                "name": f"hg38_w{third['index']}_repeat3",
                "region": "repeat",
                "seq": by_i[third["index"]]["seq"],
            }
        )
    # random 2 个额外种子（第一个 = C.5 自带 random_acgt）
    for seed in (99, 2026):
        specs.append({"name": f"random_s{seed}", "region": "random",
                      "seq": ba.synthetic_random_acgt(args.ctx, seed)})
    # ncbi CDS 2 条（独立基因座，补细菌编码区伪重复：lacZ 三窗同基因）
    from train.data import load_ncbi_chunks

    for r in load_ncbi_chunks("train")[:2]:
        specs.append({"name": f"ncbi_{r.name}", "region": "coding",
                      "seq": r.seq[:args.ctx]})
    # OG2 域覆盖列
    og2 = importlib.util.spec_from_file_location(
        "bench_step15_lossless_mod", str(_SCRIPTS / "bench_step15_lossless.py")
    )
    lm = importlib.util.module_from_spec(og2)
    og2.loader.exec_module(lm)
    for key, sub in lm.OG2_PROMPT_SUBDIR.items():
        specs.append({"name": f"og2_{key}_first", "region": f"domain_{key}",
                      "seq": lm._og2_first_seq(sub)})
    # imgvr valid 3 条（存在则并入）
    if IMGVR_PROMPTS.exists():
        imgvr = json.loads(IMGVR_PROMPTS.read_text())
        for s in imgvr.get("prompts", []):
            specs.append({"name": s["name"], "region": "coding_imgvr", "seq": s["seq"]})
    else:
        print("[main-suite] imgvr prompts 未就绪（跳过，落地后重跑本脚本）")

    out = Path(args.out)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(
            {
                "note": "Step 15 主表扩展 prompt（C.5 内置 5 条之外）；区域标签与报告主表对齐",
                "prompts": specs,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )
    tmp.replace(out)
    by_region: dict[str, int] = {}
    for s in specs:
        by_region[s["region"]] = by_region.get(s["region"], 0) + 1
    print(f"[main-suite] {len(specs)} 条 → {out}；区域分布 {by_region}")


if __name__ == "__main__":
    main()
