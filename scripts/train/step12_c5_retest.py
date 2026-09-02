"""Step 12 C.5 复测（plans/14 §3）：新 checkpoint 走 Step 10/11 全套路，不改审定文件。

与 ``step11_c5_retest`` 同机制（importlib 加载 ``bench_neural_c5`` 后覆盖模块常量
再调 ``main()``），两处差异：

1. 结果合并目标改为 ``benchmarks/step12_retrain.json``；
2. ``run_grid`` 的 5 prompt 之外追加 1–2 个 gtdb 留基因组编码区 prompt——通过
   包装 ``mod.load_ba`` 返回打了补丁的 ``bench_acceptance`` 模块实现（prompt
   记录结构与 ``bench_acceptance.build_prompts`` 输出同构，tokenize/复杂度打分
   复用其自身函数），``bench_neural_c5.py`` / ``bench_acceptance.py`` 零改动。
   prompt 序列来自训练时落盘的 ``gtdb_holdout_prompts``（ORF 密集窗前 1024bp，
   所在基因组家族整体未参与训练）。贪心对拍 4 prompt 集不扩（plans/14 未要求）。

用法（必须可见设备为物理 GPU0）::

    CUDA_VISIBLE_DEVICES=0 HF_HOME=./hf_home \\
      /home/dh/miniconda3/envs/evo2/bin/python -u scripts/train/step12_c5_retest.py \\
      --ckpt benchmarks/phasec_ckpts/S3_d1024_g7_step12.pt --tag S3_d1024_g7_step12 --gamma 7
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # plans/14 硬约束：复测只用 GPU0
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / "hf_home"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

REPO = Path(__file__).resolve().parents[2]
BENCH = _SCRIPTS / "bench_neural_c5.py"
TRAIN_JSON = REPO / "benchmarks" / "step12_retrain.json"


def log(msg: str) -> None:
    print(f"[step12_c5] {msg}", flush=True)


def load_bench():
    spec = importlib.util.spec_from_file_location("bench_neural_c5_step12", BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_gtdb_prompt_specs(train_json: Path | None = None) -> list[dict]:
    """训练时落盘的 gtdb 留基因组 prompt 描述 → build_prompts 的 spec 条目。"""
    path = Path(train_json) if train_json else TRAIN_JSON
    if not path.exists():
        raise RuntimeError(f"{path} 不存在，无法读取 gtdb_holdout_prompts")
    with path.open() as fh:
        prompts = json.load(fh).get("gtdb_holdout_prompts") or []
    if not prompts:
        raise RuntimeError("step12_retrain.json 里没有 gtdb_holdout_prompts（训练未用 gtdb？）")
    out = []
    for spec in prompts[:2]:
        out.append(
            {
                "region": "coding",
                "name": spec["name"],
                "source": spec.get("source", "opengenome2:gtdb_v220_imgpr 留出基因组"),
                "seq": spec["prompt_seq"],
                "window": None,
                "note": (
                    f"Step 12 新增：gtdb 留基因组 ORF 密集窗（最长 ORF {spec['orf_span_bp']}bp"
                    f"@contig {spec['contig_len']}bp，家族未参与训练）"
                ),
            }
        )
    return out


def patch_load_ba(mod, train_json: Path | None = None) -> None:
    """给 mod.load_ba 的返回值打 build_prompts 补丁（每次调用都重新打）。"""
    orig_load_ba = mod.load_ba

    def load_ba_patched():
        ba = orig_load_ba()
        orig_build = ba.build_prompts
        gtdb_specs = build_gtdb_prompt_specs(train_json)
        log(f"追加 gtdb 留基因组 prompt: {[s['name'] for s in gtdb_specs]}")

        def build_prompts_patched(tokenizer, windows, ctx, random_seed):
            prompts = orig_build(tokenizer, windows, ctx, random_seed)
            for s in gtdb_specs:
                ids, used = ba.tokenize_text(tokenizer, s["seq"], ctx)
                rec = {k: v for k, v in s.items() if k != "seq"}
                rec["ctx"] = ctx
                rec["offset"] = 0
                rec["used_prefix"] = used[:80]
                rec["ids"] = ids
                rec["prompt_complexity"] = ba.score_prompt_span(used, ctx)
                prompts.append(rec)
            return prompts

        ba.build_prompts = build_prompts_patched
        return ba

    mod.load_ba = load_ba_patched


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 12 C.5 复测包装（含 gtdb 留基因组 prompt）")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--tag", type=str, required=True, help="格子 tag（命名结果文件与 drafter 种子名）")
    p.add_argument("--gamma", type=int, required=True, help="与训练一致的 γ")
    p.add_argument("--n-tokens", type=int, default=256)
    p.add_argument("--results", type=str, default=None,
                   help="默认 benchmarks/step12_c5_<tag>.json")
    p.add_argument("--modes", type=str, default="smoke,greedy,grid",
                   help="逗号分隔子集：smoke,greedy,grid（默认全跑）")
    p.add_argument("--no-merge", action="store_true", help="不合并进 step12_retrain.json")
    p.add_argument("--merge-into", type=str, default=None,
                   help="合并目标 json（默认 benchmarks/step12_retrain.json）")
    p.add_argument("--no-gtdb-prompts", action="store_true",
                   help="不追加 gtdb prompt（对照/排障用）")
    p.add_argument("--train-json", type=str, default=None,
                   help="gtdb_holdout_prompts 读取来源（默认 benchmarks/step12_retrain.json；冒烟可指向 /tmp 的训练 json）")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    results_path = Path(
        args.results or (REPO / "benchmarks" / f"step12_c5_{args.tag}.json")
    )
    ckpt = args.ckpt if os.path.isabs(args.ckpt) else str(REPO / args.ckpt)
    if not os.path.isfile(ckpt):
        raise RuntimeError(f"checkpoint 不存在: {ckpt}")

    mod = load_bench()
    mod.CKPT = ckpt
    mod.GAMMA = int(args.gamma)
    mod.DRAFTER_NAME = f"neural_{args.tag}"
    mod.RESULTS = str(results_path)
    if not args.no_gtdb_prompts:
        patch_load_ba(mod, Path(args.train_json) if args.train_json else None)
    log(f"CKPT={ckpt}")
    log(f"GAMMA={mod.GAMMA} DRAFTER_NAME={mod.DRAFTER_NAME} RESULTS={mod.RESULTS}")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    argv = ["--ckpt", ckpt, "--n-tokens", str(args.n_tokens)]
    for m in modes:
        if m not in ("smoke", "greedy", "grid"):
            raise RuntimeError(f"未知 mode {m}")
        argv.append(f"--{m}")
    sys.argv = ["bench_neural_c5.py"] + argv
    mod.main()

    with results_path.open() as fh:
        raw = json.load(fh)
    if args.no_merge:
        return
    merge_path = Path(args.merge_into) if args.merge_into else TRAIN_JSON
    if not merge_path.exists():
        log(f"警告：{merge_path} 不存在，跳过合并")
        return
    with merge_path.open() as fh:
        train = json.load(fh)
    grid = raw.get("grid") or {}
    train.setdefault("c5_retest", {})[args.tag] = {
        "ckpt": ckpt,
        "gamma": int(args.gamma),
        "n_tokens": args.n_tokens,
        "results_file": str(results_path),
        "greedy_summary": raw.get("greedy_summary"),
        "grid": {
            "drafter": grid.get("drafter"),
            "gamma": grid.get("gamma"),
            "runs": [
                {
                    "prompt": r["prompt"],
                    "region": r["region"],
                    "mean_tau": r["mean_tau"],
                    "tok_s": r["tok_s"],
                    "native_tok_s": r["native_tok_s"],
                    "speedup_vs_native": r["speedup_vs_native"],
                    "pos_cond_accept": r["pos_cond_accept"],
                    "n_rounds": r["n_rounds"],
                }
                for r in grid.get("runs", [])
            ],
        },
    }
    tmp = merge_path.with_suffix(merge_path.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(train, fh, indent=2, ensure_ascii=False)
    tmp.replace(merge_path)
    log(f"已合并 {args.tag} 复测关要进 {merge_path}")


if __name__ == "__main__":
    main()
