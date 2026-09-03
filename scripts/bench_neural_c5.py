# 端到端集成测量引擎：神经 drafter 接入切片投机循环。
#
# prompt 来自自包含评估包（evspark/data/eval_prompts_24.json（随包发布），经 evalpack 加载），
# 不依赖外部语料。通常经 evspark.train.eval_suite 包装调用（覆盖 CKPT/GAMMA/
# RESULTS 等模块常量），也可单独直跑。
#
# 模式（默认全跑；落盘 RESULTS）：
#   --align-check   先手：量化「滞后一位注入」对 ᾱ/τ̂ 的影响
#                   （需完整训练语料构建 eval packs，开发用，公开仓库默认缺料会报错）
#   --smoke         1 格 32 token 通路冒烟（lacZ 采样）
#   --greedy        无损性：base 5 prompt 贪心 vs vortex 原生 top_k=1
#   --grid          核心：24 prompt 采样（T=1.0/top_k=4），真实 τ、位置条件接受率、
#                   实测 tok/s 与加速比、逐轮 c_k
#
# 用法: CUDA_VISIBLE_DEVICES=0 python scripts/bench_neural_c5.py [--all] [--grid] ...
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # 计划硬约束：只用 GPU0（须先于 evspark.train.distill import）
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / "hf_home"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

import evalpack
from evspark.checkpoints import default_ckpt_dir

RESULTS = str(Path(__file__).resolve().parents[1] / "benchmarks" / "eval_suite.json")
CKPT = str(default_ckpt_dir() / "L27_g12_80M_s1.pt")
STEP9_JSON = str(Path(__file__).resolve().parents[1] / "benchmarks" / "step9_layer_gamma_sweep.json")
PROMPT_PACK = str(evalpack.PROMPT_PACK)  # eval_suite 可覆盖指向其它 prompt 包

SEED_STEP6 = 20260821
SEED_GREEDY = 20260824  # 与 Step 5/7 贪心对拍同种子
CTX = 1024
GAMMA = 7  # 与训练一致（plans/12 §5）
N_TOKENS = 256
TEMPERATURE = 1.0
TOP_K = 4
DRAFTER_NAME = "neural_s3_d1024_g7_5M"


def log(msg: str) -> None:
    print(f"[bench-c5] {msg}", flush=True)


def dump(out: dict) -> None:
    Path(RESULTS).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(RESULTS + ".tmp")
    with tmp.open("w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    tmp.replace(RESULTS)
    log(f"已写入 {RESULTS}")


def load_prompts(tokenizer) -> list[dict]:
    """加载评估 prompt 包（PROMPT_PACK 可被 eval_suite 覆盖）。"""
    prompts = evalpack.load_prompt_pack(tokenizer, PROMPT_PACK, CTX)
    log(f"prompt 包 {PROMPT_PACK}：{len(prompts)} 条")
    return prompts


def load_drafter_only(ckpt_path: str, device: torch.device):
    """只加载 drafter 权重（不挂 hook）——align-check 用。"""
    from evspark.train.drafter import SCHEME_LAYERS, Drafter

    ck = torch.load(ckpt_path, map_location="cpu")
    drafter = Drafter.from_scheme(ck["scheme"], d_model=ck["d_model"], gamma=ck["gamma"])
    drafter.load_state_dict(ck["state_dict"])
    drafter = drafter.to(device)
    drafter.eval()
    for p in drafter.parameters():
        p.requires_grad_(False)
    meta = {k: v for k, v in ck.items() if k != "state_dict"}
    meta["layers"] = list(SCHEME_LAYERS[ck["scheme"]])
    meta["ctx_window"] = drafter.ctx_window
    return drafter, meta


# ---------------------------------------------------------------------------
# §2 对齐分析：滞后注入量化
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_ctx_shift(drafter, packs, scheme: str, gamma: int, device, cap: int, shift: int) -> dict:
    """与 ``train.distill.eval_drafter`` 同口径，仅 ctx_lens 平移 ``shift``。

    shift=0：训练口径（ctx_lens=a+1，H_ctx 含锚点自身 hidden）；
    shift=-1：decode 可得口径（ctx_lens=a，只到最后被消费位置）。
    """
    from evspark.train.distill import gather_batch, holdout_anchors
    from evspark.train.drafter import SCHEME_LAYERS, analytic_accept, predicted_tau

    layers = SCHEME_LAYERS[scheme]
    alpha_sum = None
    n = 0
    top1_pt = 0
    n_tok = 0
    per_name: dict[str, dict] = {}
    for pack in packs:
        anchors = holdout_anchors(int(pack.ids.size), pack.holdout_start, gamma, cap)
        if anchors.size == 0:
            continue
        ids = torch.as_tensor(pack.ids, dtype=torch.long, device=device)
        p_t = torch.as_tensor(pack.p_t, dtype=torch.float32, device=device)
        if layers:
            h_raw = torch.cat(
                [torch.as_tensor(pack.hidden[ln], dtype=torch.float32, device=device) for ln in layers],
                dim=-1,
            )
        else:
            h_raw = None
        a_t = torch.as_tensor(anchors, dtype=torch.long, device=device)
        anchor_ids, prev, _target, pt, ctx_lens = gather_batch(ids, p_t, a_t, gamma)
        _, probs, _, _ = drafter(anchor_ids, prev, ctx_lens + int(shift), h_raw=h_raw)
        cstar = analytic_accept(probs, pt)
        a_mean = cstar.mean(dim=0)
        n_b = int(cstar.shape[0])
        per_name[pack.name] = {
            "n_blocks": n_b,
            "alpha_bar": [float(x) for x in a_mean.cpu().tolist()],
            "tau_hat": float(predicted_tau(a_mean).item()),
            "top1_vs_pt": float((probs.argmax(-1) == pt.argmax(-1)).float().mean().item()),
        }
        alpha_sum = cstar.sum(dim=0) if alpha_sum is None else alpha_sum + cstar.sum(dim=0)
        n += n_b
        top1_pt += int((probs.argmax(-1) == pt.argmax(-1)).sum().item())
        n_tok += int(cstar.numel())
    alpha_bar = (alpha_sum / n).cpu()
    return {
        "ctx_shift": int(shift),
        "n_blocks": n,
        "n_positions": n_tok,
        "alpha_bar": [float(x) for x in alpha_bar.tolist()],
        "tau_hat": float(predicted_tau(alpha_bar).item()),
        "top1_vs_pt": (top1_pt / n_tok) if n_tok else float("nan"),
        "per_name": per_name,
    }


def run_align_check(evo, out: dict) -> dict:
    from evspark.train.distill import build_eval_packs, eval_drafter, pick_eval_records
    from evspark.train.drafter import SCHEME_LAYERS

    device = torch.device("cuda:0")
    ck = torch.load(CKPT, map_location="cpu")
    scheme, gamma = ck["scheme"], int(ck["gamma"])
    layers = SCHEME_LAYERS[scheme]
    del ck
    log(f"§2 对齐分析：scheme={scheme} γ={gamma} 注入层={layers}")

    drafter, meta = load_drafter_only(CKPT, device)
    log(f"drafter 加载完成（ctx_window={meta['ctx_window']}）")

    eval_recs = pick_eval_records(n_hg=16)
    log(f"构建 eval packs（{len(eval_recs)} 窗，S3 三层，与 Step 8/9 同协议）…")
    t0 = time.perf_counter()
    packs = build_eval_packs(evo, eval_recs, 2048, device, tuple(layers))
    log(f"eval packs 完成，{time.perf_counter() - t0:.1f}s")

    cap = 256
    log("arm A：训练口径 ctx_lens=a+1（含锚点 hidden）…")
    arm_trained = eval_ctx_shift(drafter, packs, scheme, gamma, device, cap, shift=0)
    log(f"  τ̂={arm_trained['tau_hat']:.4f} ᾱ={[round(x, 4) for x in arm_trained['alpha_bar']]}")
    log("arm B：decode 口径 ctx_lens=a（滞后一位）…")
    arm_lagged = eval_ctx_shift(drafter, packs, scheme, gamma, device, cap, shift=-1)
    log(f"  τ̂={arm_lagged['tau_hat']:.4f} ᾱ={[round(x, 4) for x in arm_lagged['alpha_bar']]}")

    # 交叉 sanity：shift=0 应与训练管线自带 eval_drafter 完全一致
    ref = eval_drafter(drafter, packs, scheme, gamma, device, cap)
    same = np.allclose(arm_trained["alpha_bar"], ref["alpha_bar"], atol=0.0, rtol=0.0)
    log(f"sanity：shift=0 vs eval_drafter 逐位一致={same}（eval_drafter τ̂={ref['tau_hat']:.4f}）")

    step9_ref = None
    if os.path.isfile(STEP9_JSON):
        with open(STEP9_JSON) as f:
            s9 = json.load(f)
        for cell in s9.get("cells", []):
            if os.path.basename(str(cell.get("ckpt", ""))) == os.path.basename(CKPT):
                step9_ref = {
                    "tau_hat": (cell.get("eval") or {}).get("tau_hat"),
                    "alpha_bar": (cell.get("eval") or {}).get("alpha_bar"),
                }
    d_tau = arm_lagged["tau_hat"] - arm_trained["tau_hat"]
    out["align_check"] = {
        "ckpt": CKPT,
        "scheme": scheme,
        "gamma": gamma,
        "layers": list(layers),
        "ctx_window": meta["ctx_window"],
        "cap_eval_anchors": cap,
        "question": "训练 H_ctx 含锚点自身 hidden（ctx_lens=a+1）；decode 锚点未消费、只能到最后被消费位置（ctx_lens=a）→ 一位滞后",
        "arm_trained_ctx_a+1": arm_trained,
        "arm_lagged_ctx_a": arm_lagged,
        "delta_tau_hat": d_tau,
        "delta_tau_hat_rel": d_tau / arm_trained["tau_hat"],
        "delta_alpha_bar": [
            float(b - a) for a, b in zip(arm_trained["alpha_bar"], arm_lagged["alpha_bar"])
        ],
        "sanity_shift0_equals_eval_drafter": bool(same),
        "step9_reference": step9_ref,
    }
    dump(out)
    return out["align_check"]


# ---------------------------------------------------------------------------
# 冒烟 / 贪心无损 / 网格
# ---------------------------------------------------------------------------


def run_smoke(model, nd, prompts, out: dict, n_tokens: int = 32) -> None:
    from evspark.specdec.block.loop import speculative_generate

    p = next(x for x in prompts if x["name"] == "lacz_coding")
    log(f"冒烟：{p['name']} 采样 n={n_tokens} γ={GAMMA}")
    seed = 20260830
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    spec = speculative_generate(
        model, nd, p["ids"], n_tokens, GAMMA, greedy=False,
        rng=np.random.default_rng(seed), temperature=TEMPERATURE, top_k=TOP_K,
    )
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    ks = [int(r.k) for r in spec.rounds_log]
    out["smoke"] = {
        "prompt": p["name"],
        "n_tokens": n_tokens,
        "gamma": GAMMA,
        "seed": seed,
        "wall_s": round(wall, 3),
        "n_rounds": len(ks),
        "ks": ks,
        "mean_tau": float(np.mean(ks) + 1.0),
        "emitted_head": [int(x) for x in spec.emitted_ids[:24]],
        "conf_head": [list(map(float, r.conf)) for r in spec.rounds_log[:4]],
        "t_draft_ms_median": float(np.median([r.t_draft for r in spec.rounds_log]) * 1000),
        "t_block_ms_median": float(np.median([r.t_block for r in spec.rounds_log]) * 1000),
    }
    log(
        f"  rounds={len(ks)} mean_τ={out['smoke']['mean_tau']:.3f} wall={wall:.2f}s "
        f"t_draft={out['smoke']['t_draft_ms_median']:.2f}ms ks={ks}"
    )
    dump(out)


def run_greedy(model, tokenizer, nd, out: dict, n_tokens: int, prompts: list[dict]) -> dict:
    from evspark.specdec.block.loop import native_greedy_reference, speculative_generate

    cases = []
    for p in prompts:
        name = p["name"]
        ids = p["ids"]
        log(f"贪心对拍 {name} n={n_tokens} γ={GAMMA}（神经 drafter）…")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        spec = speculative_generate(
            model, nd, ids, n_tokens, GAMMA, greedy=True,
            rng=np.random.default_rng(SEED_GREEDY), record_logits=True, rollback="slice",
        )
        torch.cuda.synchronize()
        t_spec = time.perf_counter() - t0
        nat_ids, nat_logits = evalpack.vortex_greedy(model, tokenizer, ids, n_tokens)
        ref_ids, _ip, ref_logits = native_greedy_reference(model, ids, n_tokens, record_logits=True)
        case = evalpack.summarize_greedy_case(
            name, spec, nat_ids, nat_logits, ref_ids, ref_logits, t_spec, 0.0, gamma=GAMMA
        )
        cases.append(case)
        log(
            f"  {name}: vs vortex 匹配={case['vs_vortex']['n_match_prefix']}/{n_tokens} "
            f"非tie={case['vs_vortex']['n_nontie_disagree']} tie={case['vs_vortex']['n_tie_flip']} "
            f"mean_k={case['mean_k']:.3f} rounds={case['n_rounds']}"
        )
        if case["vs_vortex"]["n_nontie_disagree"]:
            log(f"  非 tie 明细: {case['vs_vortex'].get('nontie')}")
        elif case["vs_vortex"]["n_tie_flip"]:
            log(f"  tie-flip 证据: {case['vs_vortex']['tie_flips'][0]}")
        del spec
        torch.cuda.empty_cache()
        dump_partial = out
        dump_partial["greedy_cases"] = cases
        dump(dump_partial)

    n_nontie = sum(c["vs_vortex"]["n_nontie_disagree"] for c in cases)
    n_tie = sum(c["vs_vortex"]["n_tie_flip"] for c in cases)
    out["greedy_cases"] = cases
    out["greedy_summary"] = {
        "n_prompts": len(cases),
        "n_tokens": n_tokens,
        "gamma": GAMMA,
        "seed": SEED_GREEDY,
        "n_nontie_disagree_total": n_nontie,
        "n_tie_flip_total": n_tie,
        "pass": n_nontie == 0,
        "note": "与 Step 5/7 同判据：非 tie 分歧 = 0；tie-flip 逐例 logit 证据",
    }
    dump(out)
    log(f"贪心汇总：非tie={n_nontie} tie-flip={n_tie} pass={n_nontie == 0}")
    return out["greedy_summary"]


def run_grid(model, tokenizer, nd, out: dict, n_tokens: int, prompts: list[dict]) -> dict:
    from evspark.specdec.block.loop import native_sample_reference, speculative_generate

    grid_prompts = prompts
    log("网格 prompt：" + ", ".join(p["name"] for p in grid_prompts))

    runs = []
    native_tok_s: dict[str, float] = {}
    for p in grid_prompts:
        name = p["name"]
        seed = evalpack.run_seed(name, DRAFTER_NAME, base=SEED_STEP6)
        log(f"--- region={p['region']} prompt={name} seed={seed} ---")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        spec = speculative_generate(
            model, nd, p["ids"], n_tokens, GAMMA, greedy=False,
            rng=np.random.default_rng(seed), temperature=TEMPERATURE, top_k=TOP_K,
            rollback="slice",
        )
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        ks = [int(r.k) for r in spec.rounds_log]
        m = evalpack.metrics_from_ks(ks, GAMMA, n_tokens, wall)

        log(f"  原生参照（native_sample_reference 同 prompt 同 n）…")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        native_sample_reference(
            model, p["ids"], n_tokens, np.random.default_rng(seed + 1),
            temperature=TEMPERATURE, top_k=TOP_K,
        )
        torch.cuda.synchronize()
        nat_wall = time.perf_counter() - t0
        nat_tok_s = n_tokens / nat_wall
        native_tok_s[name] = nat_tok_s

        conf_rounds = [list(map(float, r.conf)) for r in spec.rounds_log]
        conf_arr = np.asarray(conf_rounds, dtype=np.float64)  # [n_rounds, γ]
        rec = {
            "region": p["region"],
            "prompt": name,
            "drafter": DRAFTER_NAME,
            "seed": seed,
            "n_tokens": n_tokens,
            "gamma": GAMMA,
            "rollback": "slice",
            "ks": ks,
            "n_rounds": m["n_rounds"],
            "mean_k": m["mean_k"],
            "mean_tau": m["mean_tau"],
            "pos_cond_accept": m["pos_cond_accept"],
            "wall_s": wall,
            "tok_s": m["tok_s"],
            "native_tok_s": nat_tok_s,
            "native_wall_s": nat_wall,
            "speedup_vs_native": m["tok_s"] / nat_tok_s,
            "median_t_draft_ms": float(np.median([r.t_draft for r in spec.rounds_log]) * 1000),
            "median_t_block_ms": float(np.median([r.t_block for r in spec.rounds_log]) * 1000),
            "median_t_slice_ms": float(np.median([r.t_slice for r in spec.rounds_log]) * 1000),
            "median_t_verify_ms": float(np.median([r.t_verify for r in spec.rounds_log]) * 1000),
            "conf_pos_mean": [float(x) for x in conf_arr.mean(axis=0)],
            "conf_rounds": conf_rounds,
        }
        runs.append(rec)
        log(
            f"  τ={m['mean_tau']:.3f} tok/s={m['tok_s']:.2f} native={nat_tok_s:.2f} "
            f"加速={rec['speedup_vs_native']:.3f}× t_draft={rec['median_t_draft_ms']:.2f}ms "
            f"pos={[None if x is None else round(x, 3) for x in m['pos_cond_accept']]}"
        )
        del spec
        torch.cuda.empty_cache()
        out["grid"] = {"runs": runs}
        dump(out)

    lacz_native = native_tok_s.get("lacz_coding")
    out["grid"] = {
        "gamma": GAMMA,
        "n_tokens": n_tokens,
        "temperature": TEMPERATURE,
        "top_k": TOP_K,
        "ctx": CTX,
        "seed_base": SEED_STEP6,
        "drafter": DRAFTER_NAME,
        "native_protocol": "native_sample_reference 同 prompt 同 n_tokens（T=1.0/top_k=4）",
        "native_tok_s_per_prompt": native_tok_s,
        "lacz_native_tok_s": lacz_native,
        "runs": runs,
        "speedup_vs_lacz_native": {
            r["prompt"]: (r["tok_s"] / lacz_native if lacz_native else None) for r in runs
        },
    }
    dump(out)
    return out["grid"]


# ---------------------------------------------------------------------------


def main() -> None:
    global CKPT
    ap = argparse.ArgumentParser(description="Step 10 C.5 端到端集成测量（GPU0）")
    ap.add_argument("--align-check", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n-tokens", type=int, default=N_TOKENS)
    ap.add_argument("--smoke-tokens", type=int, default=32)
    ap.add_argument("--ckpt", type=str, default=CKPT)
    args = ap.parse_args()
    run_all = args.all or not (args.align_check or args.smoke or args.greedy or args.grid)

    CKPT = args.ckpt

    out = {}
    if os.path.isfile(RESULTS):
        with open(RESULTS) as f:
            out = json.load(f)
    out["env"] = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "model": "evo2_7b (1m ctx), bf16, use_kernels=True",
        "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    out["ckpt"] = CKPT
    out["gamma"] = GAMMA
    dump(out)

    from evo2 import Evo2

    log("加载 Evo2 7B（use_kernels=True）…")
    evo = Evo2("evo2_7b", use_kernels=True)
    model = evo.model
    model.eval()
    tokenizer = evo.tokenizer

    if args.align_check or run_all:
        log("=== §2 对齐分析（align-check）===")
        ac = run_align_check(evo, out)
        log(
            f"§2 结论：τ̂ 训练口径 {ac['arm_trained_ctx_a+1']['tau_hat']:.4f} → "
            f"滞后口径 {ac['arm_lagged_ctx_a']['tau_hat']:.4f} "
            f"（Δ={ac['delta_tau_hat']:+.4f}, {ac['delta_tau_hat_rel']:+.2%}）"
        )

    nd = None
    if args.smoke or args.greedy or args.grid or run_all:
        from evspark.specdec.block.neural_draft import NeuralDraftModel

        nd = NeuralDraftModel.from_checkpoint(model, CKPT, device="cuda:0")
        log(f"神经 drafter 已挂载（层 {nd.layer_names}，γ={nd.gamma}）")

    prompts = None
    if args.smoke or args.greedy or args.grid or run_all:
        prompts = load_prompts(tokenizer)

    try:
        if args.smoke or run_all:
            run_smoke(model, nd, prompts, out, n_tokens=args.smoke_tokens)
        if args.greedy or run_all:
            base5 = prompts[:5]
            log(f"=== 贪心无损对拍（{len(base5)} base prompt × {args.n_tokens} token）===")
            run_greedy(model, tokenizer, nd, out, args.n_tokens, base5)
        if args.grid or run_all:
            log(f"=== 网格（{len(prompts)} prompt × {args.n_tokens} token，γ={GAMMA} 采样）===")
            run_grid(model, tokenizer, nd, out, args.n_tokens, prompts)
    finally:
        if nd is not None:
            nd.close()

    dump(out)
    log("完成。")


if __name__ == "__main__":
    main()
