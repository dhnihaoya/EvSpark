"""Step 11 C.5 复测（plans/13 §3）：新 checkpoint 走 Step 10 全套路，不改 Step 10 文件。

``bench_neural_c5.py`` 的 CKPT/GAMMA/DRAFTER_NAME/RESULTS 是模块级常量；
本脚本以 importlib 加载后**覆盖模块常量**再调 ``main()``，复用其全部函数与协议
（5 prompt × 256 token 采样 T=1.0/top_k=4、4 prompt × 256 贪心无损对拍、
同种子体系），原始输出落 ``benchmarks/step11_c5_<tag>.json``，并把关要合并进
``benchmarks/step11_retrain.json`` 的 ``c5_retest`` 节。Step 10 的
``benchmarks/step10_c5.json`` 不受影响。

用法（必须可见设备为物理 GPU0）::

    CUDA_VISIBLE_DEVICES=0 HF_HOME=./hf_home \\
      /home/dh/miniconda3/envs/evo2/bin/python -u scripts/train/step11_c5_retest.py \\
      --ckpt benchmarks/phasec_ckpts/S3_d1024_g12_step11.pt --tag S3_d1024_g12_step11 --gamma 12
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # plans/13 硬约束：复测只用 GPU0
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / "hf_home"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

REPO = Path(__file__).resolve().parents[2]
BENCH = _SCRIPTS / "bench_neural_c5.py"
TRAIN_JSON = REPO / "benchmarks" / "step11_retrain.json"


def log(msg: str) -> None:
    print(f"[step11_c5] {msg}", flush=True)


def load_bench():
    spec = importlib.util.spec_from_file_location("bench_neural_c5_step11", BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 11 C.5 复测包装（覆盖 bench_neural_c5 模块常量）")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--tag", type=str, required=True, help="格子 tag（命名结果文件与 drafter 种子名）")
    p.add_argument("--gamma", type=int, required=True, help="与训练一致的 γ")
    p.add_argument("--n-tokens", type=int, default=256)
    p.add_argument("--results", type=str, default=None,
                   help="默认 benchmarks/step11_c5_<tag>.json")
    p.add_argument("--modes", type=str, default="smoke,greedy,grid",
                   help="逗号分隔子集：smoke,greedy,grid（默认全跑；align-check 属 Step 10 机制结论，不复跑）")
    p.add_argument("--no-merge", action="store_true", help="不合并进 step11_retrain.json")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    results_path = Path(
        args.results or (REPO / "benchmarks" / f"step11_c5_{args.tag}.json")
    )
    ckpt = args.ckpt if os.path.isabs(args.ckpt) else str(REPO / args.ckpt)
    if not os.path.isfile(ckpt):
        raise RuntimeError(f"checkpoint 不存在: {ckpt}")

    mod = load_bench()
    mod.CKPT = ckpt
    mod.GAMMA = int(args.gamma)
    mod.DRAFTER_NAME = f"neural_{args.tag}"
    mod.RESULTS = str(results_path)
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
    if not TRAIN_JSON.exists():
        log(f"警告：{TRAIN_JSON} 不存在，跳过合并")
        return
    with TRAIN_JSON.open() as fh:
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
    tmp = TRAIN_JSON.with_suffix(".json.tmp")
    with tmp.open("w") as fh:
        json.dump(train, fh, indent=2, ensure_ascii=False)
    tmp.replace(TRAIN_JSON)
    log(f"已合并 {args.tag} 复测关要进 {TRAIN_JSON}")


if __name__ == "__main__":
    main()
