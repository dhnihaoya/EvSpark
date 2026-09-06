"""论文评估套件：神经 drafter ckpt × 自包含 24-prompt 包（1024-token 协议）。

机制：importlib 加载 ``bench_neural_c5`` 后覆盖模块常量（CKPT / GAMMA /
DRAFTER_NAME / RESULTS / PROMPT_PACK）再调 ``main()``。prompt 包默认
``evspark/data/eval_prompts_24.json``（论文 24-prompt 套件，序列自包含，
区域构成见文件内 meta），可用 ``--prompts`` 换成自定义包（同 schema）。

用法（ckpt 可为已发布名字或本地 .pt 路径）::

    CUDA_VISIBLE_DEVICES=0 python -m evspark.train.eval_suite \
      --ckpt L27_g12_150M_s1 --tag repro_s1 --n-tokens 1024 --modes smoke,grid
    # 结果落盘 benchmarks/eval_suite_<tag>.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / "hf_home"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

REPO = Path(__file__).resolve().parents[2]
BENCH = REPO / "scripts" / "bench_neural_c5.py"

import evspark  # noqa: E402

DEFAULT_PROMPT_PACK = Path(evspark.__file__).resolve().parent / "data" / "eval_prompts_24.json"


def log(msg: str) -> None:
    print(f"[eval-suite] {msg}", flush=True)


def load_bench():
    spec = importlib.util.spec_from_file_location("bench_neural_c5_mod", BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser(description="论文评估套件（神经 ckpt × 自包含 prompt 包）")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--gamma", type=int, default=None,
                    help="decode γ′，默认读 ckpt 元数据（训练 γ）；须 ≤ 训练 γ"
                         "（decode-γ 解耦：γ′<训练γ 为精确前缀）")
    ap.add_argument("--n-tokens", type=int, default=1024)
    ap.add_argument("--modes", type=str, default="grid", help="smoke,greedy,grid 子集")
    ap.add_argument("--results", type=str, default=None)
    ap.add_argument("--prompts", type=str, default=str(DEFAULT_PROMPT_PACK),
                    help="prompt 包 JSON（默认论文 24-prompt 套件）")
    ap.add_argument("--label", type=str, default=None,
                    help="登记备注（如 before/after），写入结果 json")
    ap.add_argument("--no-markov-head", action="store_true",
                    help="消融探针：建 drafter 后将全矩阵 Markov 头清零（trunk 不动），"
                         "测 Markov 头对端到端加速的贡献")
    args = ap.parse_args()

    from evspark.checkpoints import ensure_ckpt

    ckpt = str(ensure_ckpt(args.ckpt))
    if args.gamma is None:
        ck = torch_load_meta(ckpt)
        args.gamma = int(ck["gamma"])
        log(f"γ 取自 ckpt: {args.gamma}")

    mod = load_bench()
    mod.CKPT = ckpt
    mod.GAMMA = int(args.gamma)
    mod.DRAFTER_NAME = f"neural_{args.tag}"
    mod.PROMPT_PACK = args.prompts
    results_path = Path(args.results or (REPO / "benchmarks" / f"eval_suite_{args.tag}.json"))
    mod.RESULTS = str(results_path)
    log(f"CKPT={ckpt} γ={mod.GAMMA} PROMPTS={mod.PROMPT_PACK} RESULTS={mod.RESULTS}")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if args.no_markov_head:
        import torch

        from evspark.specdec.block.neural_draft import NeuralDraftModel

        orig_from_ckpt = NeuralDraftModel.from_checkpoint.__func__

        def _patched(cls, model, ckpt_path, **kw):
            nd = orig_from_ckpt(cls, model, ckpt_path, **kw)
            with torch.no_grad():
                nd.drafter.markov.zero_()
            log("Markov 头已清零（消融探针）")
            return nd

        NeuralDraftModel.from_checkpoint = classmethod(_patched)
    argv = ["--ckpt", ckpt, "--n-tokens", str(args.n_tokens)]
    for m in modes:
        if m not in ("smoke", "greedy", "grid"):
            raise RuntimeError(f"未知 mode {m}")
        argv.append(f"--{m}")
    sys.argv = ["bench_neural_c5.py"] + argv
    mod.main()

    # 登记套件上下文（label/γ 来源等）——不动 bench 自身结构，外层补写
    with results_path.open() as fh:
        raw = json.load(fh)
    raw.setdefault("eval_suite", {}).update(
        {
            "tag": args.tag,
            "label": args.label,
            "prompts": args.prompts,
        }
    )
    tmp = results_path.with_suffix(results_path.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(raw, fh, indent=2, ensure_ascii=False)
    tmp.replace(results_path)
    log(f"完成，结果 {results_path}")


def torch_load_meta(ckpt: str) -> dict:
    import torch

    ck = torch.load(ckpt, map_location="cpu")
    return {k: v for k, v in ck.items() if k != "state_dict"}


if __name__ == "__main__":
    main()
