"""Step 14 C.5 验收驱动：离线最终模型 × 路由 drafter（plans/16 后处理收尾）。

复用 ``step12_c5_retest`` 的全部机制（importlib 加载 ``bench_neural_c5`` + 覆盖常量 +
gtdb 留基因组 prompt 补丁），唯一改动是把 drafter 工厂替换为
``RoutedDraftModel``（神经 conf 头 + Markov 查表置信的 per-anchor 路由，
θ 来自 ``benchmarks/step14_routing_calib.json`` 的标定结果）。

默认跑两臂：**neural 对照**（不路由，同 ckpt）+ **routed**（θ*），无损门相同
（贪心 4 prompt 非 tie 分歧 == 0）。结果合并进 ``benchmarks/step14_offline.json``
的 ``c5_retest`` 下（键分别为 ``<tag>_neural`` / ``<tag>_routedθ<θ>``）。

用法（48G 卡）::

    CUDA_VISIBLE_DEVICES=0 python -u scripts/train/step14_c5_accept.py \
      --ckpt benchmarks/phasec_ckpts/L27_d1024_g7_offline_step14.pt --gamma 7
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / "hf_home"))

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

REPO = Path(__file__).resolve().parents[2]
CALIB_JSON = REPO / "benchmarks" / "step14_routing_calib.json"
OFFLINE_JSON = REPO / "benchmarks" / "step14_offline.json"


def log(msg: str) -> None:
    print(f"[step14_c5] {msg}", flush=True)


def run_one(args: argparse.Namespace, routed: bool, theta: float | None, tag: str) -> None:
    from train import step12_c5_retest as r12

    if routed:
        import specdec.block.neural_draft as nd_mod
        from specdec.block.routed_draft import RoutedDraftModel
        from train.step14_route_calib import build_or_load_markov

        markov = build_or_load_markov()
        # 先存原始工厂再打补丁：RoutedDraftModel.from_checkpoint 内部也调
        # NeuralDraftModel.from_checkpoint——若工厂内部走它会自递归（实测踩过）
        orig_factory = nd_mod.NeuralDraftModel.from_checkpoint

        def routed_factory(model, ckpt_path, *, device="cuda:0"):
            neural = orig_factory(model, ckpt_path, device=device)
            return RoutedDraftModel(neural, markov, float(theta))

        nd_mod.NeuralDraftModel.from_checkpoint = staticmethod(routed_factory)
        log(f"路由臂：θ={theta:.4f}（Markov 表已就绪）")
    else:
        log("对照臂：纯神经（不路由）")

    argv = [
        "step12_c5_retest.py",
        "--ckpt", args.ckpt,
        "--tag", tag,
        "--gamma", str(args.gamma),
        "--n-tokens", str(args.n_tokens),
        "--modes", args.modes,
        "--merge-into", args.merge_into,
        "--train-json", args.train_json,
    ]
    sys.argv = argv
    r12.main()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--gamma", type=int, required=True)
    p.add_argument("--tag", type=str, default=None,
                   help="结果键前缀（默认取 ckpt 文件名去 .pt）")
    p.add_argument("--theta", type=float, default=None,
                   help="路由阈值；默认读 step14_routing_calib.json 的 theta_star")
    p.add_argument("--arm", choices=["both", "neural", "routed"], default="both")
    p.add_argument("--modes", type=str, default="smoke,greedy,grid")
    p.add_argument("--n-tokens", type=int, default=256)
    p.add_argument("--merge-into", type=str, default=str(OFFLINE_JSON))
    p.add_argument("--train-json", type=str,
                   default=str(REPO / "benchmarks" / "step12_retrain.json"))
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    tag_base = args.tag or Path(args.ckpt).name.removesuffix(".pt")
    theta = args.theta
    if args.arm in ("both", "routed") and theta is None:
        if not CALIB_JSON.exists():
            raise RuntimeError(f"缺标定文件 {CALIB_JSON} 且未显式给 --theta")
        theta = float(json.loads(CALIB_JSON.read_text())["theta_star"])
        log(f"标定 θ*={theta:.4f}（{CALIB_JSON}）")

    if args.arm in ("both", "neural"):
        run_one(args, routed=False, theta=None, tag=f"{tag_base}_neural")
    if args.arm in ("both", "routed"):
        run_one(args, routed=True, theta=theta, tag=f"{tag_base}_routedθ{theta:.3f}")


if __name__ == "__main__":
    main()
