"""Step 15 C.5 套件包装（plans/17 §1.4/§3.1/§3.3）：神经 ckpt × 任意扩展 prompt。

与 ``step12_c5_retest`` 同机制（importlib 加载 ``bench_neural_c5`` 后覆盖模块常量
再调 ``main()``），差异：

1. 支持 ``--extra-prompts-json``（``step15_imgvr_eval.py`` 产出的 prompt spec，
   imgvr valid 留出），与 gtdb 留出 prompt 一并追加进 grid；
2. 结果独立落 ``benchmarks/step15_c5_<tag>.json``（不回写 step12_retrain.json），
   由 Step 15 报告侧汇总。

用法::

    CUDA_VISIBLE_DEVICES=0 HF_HOME=... python -u scripts/train/step15_c5_suite.py \\
      --ckpt benchmarks/phasec_ckpts/L27_d1024_g7_offline_step14.pt --tag L27_step14_before \\
      --extra-prompts-json benchmarks/step15_imgvr_prompts.json --modes grid
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

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

REPO = Path(__file__).resolve().parents[2]
BENCH = _SCRIPTS / "bench_neural_c5.py"


def log(msg: str) -> None:
    print(f"[step15-c5] {msg}", flush=True)


def load_bench():
    spec = importlib.util.spec_from_file_location("bench_neural_c5_step15", BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def append_specs_patch(mod, specs: list[dict], label: str) -> None:
    """给 mod.load_ba 返回的 bench_acceptance 打 build_prompts 补丁。"""
    orig_load_ba = mod.load_ba

    def load_ba_patched():
        ba = orig_load_ba()
        orig_build = ba.build_prompts

        def build_prompts_patched(tokenizer, windows, ctx, random_seed):
            prompts = orig_build(tokenizer, windows, ctx, random_seed)
            for s in specs:
                ids, used = ba.tokenize_text(tokenizer, s["seq"], ctx)
                rec = {k: v for k, v in s.items() if k not in ("seq", "meta")}
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
    log(f"追加 {label}: {[s['name'] for s in specs]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 15 C.5 套件（支持扩展 prompt）")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--tag", type=str, required=True)
    ap.add_argument("--gamma", type=int, default=None, help="默认读 ckpt")
    ap.add_argument("--n-tokens", type=int, default=256)
    ap.add_argument("--modes", type=str, default="grid", help="smoke,greedy,grid 子集")
    ap.add_argument("--results", type=str, default=None)
    ap.add_argument("--extra-prompts-json", type=str, default=None,
                    help="扩展 prompt spec（step15_imgvr_eval 产物）")
    ap.add_argument("--no-gtdb-prompts", action="store_true")
    ap.add_argument("--train-json", type=str, default=None,
                    help="gtdb_holdout_prompts 来源（默认 benchmarks/step12_retrain.json）")
    ap.add_argument("--label", type=str, default=None,
                    help="登记备注（如 before/after），写入结果 json 的 step15 字段")
    ap.add_argument("--no-markov-head", action="store_true",
                    help="消融探针：建 drafter 后将全矩阵 Markov 头清零（trunk 不动），"
                         "测 Markov 头对端到端加速的贡献（Step 16 消融）")
    args = ap.parse_args()

    ckpt = str(REPO / args.ckpt) if not os.path.isabs(args.ckpt) else args.ckpt
    if not os.path.isfile(ckpt):
        raise RuntimeError(f"checkpoint 不存在: {ckpt}")
    if args.gamma is None:
        ck = torch_load_meta(ckpt)
        args.gamma = int(ck["gamma"])
        log(f"γ 取自 ckpt: {args.gamma}")

    mod = load_bench()
    mod.CKPT = ckpt
    mod.GAMMA = int(args.gamma)
    mod.DRAFTER_NAME = f"neural_{args.tag}"
    results_path = Path(args.results or (REPO / "benchmarks" / f"step15_c5_{args.tag}.json"))
    mod.RESULTS = str(results_path)
    log(f"CKPT={ckpt} γ={mod.GAMMA} RESULTS={mod.RESULTS}")

    if not args.no_gtdb_prompts:
        c5 = importlib.util.spec_from_file_location(
            "step12_c5_retest_mod", str(_SCRIPTS / "train" / "step12_c5_retest.py")
        )
        c5m = importlib.util.module_from_spec(c5)
        c5.loader.exec_module(c5m)
        gtdb_specs = c5m.build_gtdb_prompt_specs(
            Path(args.train_json) if args.train_json else None
        )
        append_specs_patch(mod, gtdb_specs, "gtdb 留出 prompt")
    if args.extra_prompts_json:
        extra = json.loads(Path(args.extra_prompts_json).read_text())
        specs = extra.get("prompts") if isinstance(extra, dict) else extra
        append_specs_patch(mod, specs, "extra prompt")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if args.no_markov_head:
        import torch

        from specdec.block.neural_draft import NeuralDraftModel

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

    # 登记 step15 上下文（label/γ 来源等）——不动 bench 自身结构，外层补写
    with results_path.open() as fh:
        raw = json.load(fh)
    raw.setdefault("step15", {}).update(
        {
            "tag": args.tag,
            "label": args.label,
            "extra_prompts_json": args.extra_prompts_json,
            "gtdb_prompts": not args.no_gtdb_prompts,
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
