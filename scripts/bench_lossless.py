"""无损性正式测量：贪心等价 24-prompt × 1024 token + 采样 KL/SD 分布等价。

贪心判据：非 tie 分歧 = 0；tie-flip 逐例 logit 证据（bf16 kernel 归约顺序
差异，允许但记录）。prompt 来自 evspark 包内嵌评估包 ``eval_prompts_24.json``
（论文 24-prompt 套件，经 evalpack 加载，不依赖外部语料）。

采样等价（KL）：3 prompt × 12 链 × 512 token，spec vs native 同温度同 top_k；
一元/联合/条件 KL + TVD，并附 even/odd 原生自举噪声地板。

用法（单卡独占；ckpt 可为名字或本地路径）::

    CUDA_VISIBLE_DEVICES=0 python -u scripts/bench_lossless.py \
      --greedy-ckpt L27_g12_150M_s1 \
      --greedy-ckpt L27_final15_150M_s1 \
      --kl-ckpt L27_g12_150M_s1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / "hf_home"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

import evalpack

REPO = _ROOT
RESULTS = REPO / "benchmarks" / "losslessness.json"
SEED_GREEDY = 20260824  # 贪心对拍同种子（全历史一致）
SEED_KL_NATIVE = 7001
SEED_KL_SPEC = 7002
CTX = 1024
VOCAB = 512


def log(msg: str) -> None:
    print(f"[lossless] {msg}", flush=True)


def dump(out: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    tmp.replace(path)
    log(f"已写入 {path}")


# ---------------------------------------------------------------------------
# prompt 集（自包含评估包）
# ---------------------------------------------------------------------------


def build_prompt_suite(tokenizer) -> list[dict]:
    """论文 24-prompt 套件（evspark 包内嵌 eval_prompts_24.json），全部 CTX=1024。"""
    suite = evalpack.load_prompt_pack(tokenizer, evalpack.PROMPT_PACK, CTX)
    log(f"prompt 集合 {len(suite)} 条：" + ", ".join(p["name"] for p in suite))
    return suite


def load_extra_prompts(tokenizer, path: Path) -> list[dict]:
    """--extra-prompts-json：``[{name, region, seq}]`` 或 ``{"prompts": [...]}``。"""
    obj = json.loads(Path(path).read_text())
    specs = obj.get("prompts") if isinstance(obj, dict) else obj
    out = []
    for s in specs:
        ids, used = evalpack.tokenize_prompt(tokenizer, s["seq"], CTX)
        out.append(
            {
                "name": s["name"],
                "region": s.get("region", "?"),
                "source": s.get("source", s["name"]),
                "ctx": CTX,
                "offset": 0,
                "used_prefix": used[:80],
                "ids": ids,
            }
        )
    return out


# ---------------------------------------------------------------------------
# 贪心等价（Step 10/12/14 run_greedy 同机制，n_tokens 扩到 1024）
# ---------------------------------------------------------------------------


def run_greedy(model, tokenizer, nd, prompts, n_tokens: int, gamma: int, tag: str) -> dict:
    from evspark.specdec.block.loop import native_greedy_reference, speculative_generate

    cases = []
    for p in prompts:
        name = p["name"]
        log(f"贪心对拍 {name} n={n_tokens} γ={gamma} …")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        spec = speculative_generate(
            model, nd, p["ids"], n_tokens, gamma, greedy=True,
            rng=np.random.default_rng(SEED_GREEDY), record_logits=True, rollback="slice",
        )
        torch.cuda.synchronize()
        t_spec = time.perf_counter() - t0
        nat_ids, nat_logits = evalpack.vortex_greedy(model, tokenizer, p["ids"], n_tokens)
        ref_ids, _ip, ref_logits = native_greedy_reference(model, p["ids"], n_tokens, record_logits=True)
        case = evalpack.summarize_greedy_case(
            name, spec, nat_ids, nat_logits, ref_ids, ref_logits, t_spec, 0.0, gamma=gamma
        )
        cases.append(case)
        log(
            f"  {name}: 匹配={case['vs_vortex']['n_match_prefix']}/{n_tokens} "
            f"非tie={case['vs_vortex']['n_nontie_disagree']} tie={case['vs_vortex']['n_tie_flip']} "
            f"mean_k={case['mean_k']:.3f}"
        )
        del spec, nat_logits, ref_logits
        torch.cuda.empty_cache()
    n_nontie = sum(c["vs_vortex"]["n_nontie_disagree"] for c in cases)
    n_tie = sum(c["vs_vortex"]["n_tie_flip"] for c in cases)
    return {
        "ckpt_tag": tag,
        "gamma": gamma,
        "n_tokens": n_tokens,
        "seed": SEED_GREEDY,
        "n_prompts": len(cases),
        "cases": cases,
        "summary": {
            "n_nontie_disagree_total": n_nontie,
            "n_tie_flip_total": n_tie,
            "pass": n_nontie == 0,
            "note": "非 tie 分歧=0 判据（Step 5/7 同款）；tie-flip 逐例 logit 证据在 cases",
        },
    }


# ---------------------------------------------------------------------------
# 采样 KL/SD 等价（test_spec_loop 冒烟的 7B 正式版）
# ---------------------------------------------------------------------------


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    p = p / p.sum()
    q = q / q.sum()
    kl = 0.0
    for pi, qi in zip(p, q):
        if pi > 0.0:
            kl += float(pi * np.log(pi / max(qi, 1e-300)))
    return kl


def _tvd(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    p = p / p.sum()
    q = q / q.sum()
    return float(0.5 * np.abs(p - q).sum())


def _unigram(seq: np.ndarray, vocab: int = VOCAB) -> np.ndarray:
    return np.bincount(seq, minlength=vocab).astype(np.float64)


def _bigram_joint(seq: np.ndarray, start: int, vocab: int = VOCAB) -> np.ndarray:
    joint = np.zeros((vocab, vocab), dtype=np.float64)
    prev = start
    for tok in seq:
        joint[prev, int(tok)] += 1.0
        prev = int(tok)
    return joint


def _smoothed(counts: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    c = counts.astype(np.float64) + alpha
    return c / c.sum()


def _cond_kl(joint_p: np.ndarray, joint_q: np.ndarray) -> float:
    w = joint_q.sum(axis=1).astype(np.float64)
    if w.sum() <= 0:
        return 0.0
    w = w / w.sum()
    total = 0.0
    for i, wi in enumerate(w):
        if wi <= 0.0:
            continue
        total += wi * _kl(_smoothed(joint_p[i]), _smoothed(joint_q[i]))
    return float(total)


def run_kl(model, nd, prompts, n_chains: int, chain_len: int, gamma: int, tag: str) -> dict:
    from evspark.specdec.block.loop import native_sample_reference, speculative_generate

    per_prompt = {}
    for p in prompts:
        name = p["name"]
        log(f"采样等价 {name}：{n_chains} 链 × {chain_len} token …")
        native_parts, spec_parts, starts, taus = [], [], [], []
        for c in range(n_chains):
            native_parts.append(
                native_sample_reference(
                    model, p["ids"], chain_len,
                    np.random.default_rng(SEED_KL_NATIVE + 100 * c),
                    temperature=1.0, top_k=4,
                )
            )
            spec = speculative_generate(
                model, nd, p["ids"], chain_len, gamma, greedy=False,
                rng=np.random.default_rng(SEED_KL_SPEC + 100 * c),
                temperature=1.0, top_k=4, rollback="slice",
            )
            spec_parts.append(spec.emitted_ids)
            starts.append(int(p["ids"][0, -1].item()))
            taus.append(float(np.mean([r.k for r in spec.rounds_log]) + 1.0))
            del spec
            torch.cuda.empty_cache()

        def _pool(parts, idx=None) -> np.ndarray:
            joint = np.zeros((VOCAB, VOCAB), dtype=np.float64)
            it = range(len(parts)) if idx is None else idx
            for i in it:
                joint += _bigram_joint(parts[i], starts[i])
            return joint

        native = np.concatenate(native_parts)
        spec_ids = np.concatenate(spec_parts)
        uni_n, uni_s = _unigram(native), _unigram(spec_ids)
        joint_n, joint_s = _pool(native_parts), _pool(spec_parts)
        # even/odd 原生自举：同 N 噪声地板
        even, odd = list(range(0, n_chains, 2)), list(range(1, n_chains, 2))
        uni_ne = _unigram(np.concatenate([native_parts[i] for i in even]))
        uni_no = _unigram(np.concatenate([native_parts[i] for i in odd]))
        jn_e, jn_o = _pool(native_parts, even), _pool(native_parts, odd)
        rec = {
            "n_chains": n_chains,
            "chain_len": chain_len,
            "n_tokens": int(native.size),
            "mean_tau_spec": float(np.mean(taus)),
            "kl_unigram": _kl(_smoothed(uni_s), _smoothed(uni_n)),
            "tvd_unigram": _tvd(_smoothed(uni_s), _smoothed(uni_n)),
            "kl_joint": _kl(_smoothed(joint_s.ravel()), _smoothed(joint_n.ravel())),
            "kl_conditional": _cond_kl(joint_s, joint_n),
            "floor_kl_unigram_native_native": _kl(_smoothed(uni_ne), _smoothed(uni_no)),
            "floor_tvd_unigram_native_native": _tvd(_smoothed(uni_ne), _smoothed(uni_no)),
            "floor_kl_conditional_native_native": _cond_kl(jn_e, jn_o),
        }
        per_prompt[name] = rec
        log(
            f"  KL_uni={rec['kl_unigram']:.3e}（地板 {rec['floor_kl_unigram_native_native']:.3e}）"
            f" TVD={rec['tvd_unigram']:.4f}（地板 {rec['floor_tvd_unigram_native_native']:.4f}）"
            f" τ̄={rec['mean_tau_spec']:.2f}"
        )
    return {"ckpt_tag": tag, "gamma": gamma, "per_prompt": per_prompt}


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="无损性正式测量（贪心等价 + 采样 KL/SD）")
    ap.add_argument("--greedy-ckpt", action="append", default=[],
                    help="贪心对拍 checkpoint（可多次；如 xxx.pt,tag 或纯路径）")
    ap.add_argument("--kl-ckpt", type=str, default=None, help="采样 KL 测量 checkpoint")
    ap.add_argument("--kl-gamma", type=int, default=None,
                    help="KL 解码 γ′（默认 ckpt 训练 γ；须 ≤ 训练 γ，decode-γ 解耦）")
    ap.add_argument("--n-tokens", type=int, default=1024)
    ap.add_argument("--kl-chains", type=int, default=12)
    ap.add_argument("--kl-len", type=int, default=512)
    ap.add_argument("--kl-prompts", type=str, default="lacz_coding,chr21_intergenic,gtdb_DPLL01_coding")
    ap.add_argument("--extra-prompts-json", type=str, default=None,
                    help="追加 prompt [{name,region,seq}] 或 {\"prompts\": [...]}（自定义扩展）")
    ap.add_argument("--suite", choices=["full", "extra"], default="full",
                    help="full=24-prompt 套件+追加；extra=只跑追加（快速抽查）")
    ap.add_argument("--results", type=str, default=str(RESULTS))
    args = ap.parse_args()
    results_path = Path(args.results)

    out: dict = {}
    if results_path.exists():
        try:
            out = json.loads(results_path.read_text())
        except json.JSONDecodeError:
            out = {}
    out["env"] = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "protocol": "贪心非tie=0 判据 + 采样 KL/TVD（even/odd 原生自举地板）",
    }

    from evo2 import Evo2
    from evspark.specdec.block.neural_draft import NeuralDraftModel

    log("加载 Evo2 7B（use_kernels=True）…")
    evo = Evo2("evo2_7b", use_kernels=True)
    model = evo.model
    model.eval()
    tokenizer = evo.tokenizer

    prompts = build_prompt_suite(tokenizer)
    if args.extra_prompts_json:
        extra = load_extra_prompts(tokenizer, Path(args.extra_prompts_json))
        log(f"追加 {len(extra)} 条 extra prompt")
        prompts = (extra if args.suite == "extra" else prompts + extra)
        if args.suite == "extra" and not extra:
            raise RuntimeError("--suite extra 但 --extra-prompts-json 为空")
    out["prompt_names"] = [p["name"] for p in prompts]

    from evspark.checkpoints import ensure_ckpt

    greedy_all = out.setdefault("greedy", [])
    for spec in args.greedy_ckpt:
        ckpt, _, tag = spec.partition(",")
        ckpt = str(ensure_ckpt(ckpt))
        tag = tag or Path(ckpt).stem
        log(f"=== 贪心等价 ckpt={ckpt} tag={tag} ===")
        nd = NeuralDraftModel.from_checkpoint(model, ckpt, device="cuda:0")
        rec = run_greedy(model, tokenizer, nd, prompts, args.n_tokens, int(nd.gamma), tag)
        greedy_all = [r for r in greedy_all if r["ckpt_tag"] != tag] + [rec]
        out["greedy"] = greedy_all
        dump(out, results_path)
        nd.close()
        del nd
        torch.cuda.empty_cache()

    if args.kl_ckpt:
        ckpt = str(ensure_ckpt(args.kl_ckpt))
        tag = Path(ckpt).stem
        nd = NeuralDraftModel.from_checkpoint(model, ckpt, device="cuda:0")
        want = [x.strip() for x in args.kl_prompts.split(",") if x.strip()]
        kl_prompts = [p for p in prompts if p["name"] in want]
        missing = [w for w in want if w not in {p["name"] for p in prompts}]
        if missing:
            log(f"警告：KL prompt 不在集合内 {missing}（可用 {[p['name'] for p in prompts]}）")
        gamma_kl = int(args.kl_gamma) if args.kl_gamma else int(nd.gamma)
        if gamma_kl > int(nd.gamma):
            raise ValueError(f"KL γ′={gamma_kl} > 训练 γ={nd.gamma}（decode-γ 只允许更小）")
        tag = f"{tag}_dg{gamma_kl}" if gamma_kl != int(nd.gamma) else tag
        out["sampling_kl"] = run_kl(model, nd, kl_prompts, args.kl_chains, args.kl_len, gamma_kl, tag)
        dump(out, results_path)
        nd.close()

    log("完成。")


if __name__ == "__main__":
    main()
