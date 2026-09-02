"""Step 15 无损性扩展（plans/17 §3.4）：贪心等价 ≥16 prompt × 1024 token +
采样 KL/SD 分布等价正式测量（把 test_spec_loop 的 KL 冒烟升级为 7B 正式口径）。

贪心判据不变（Step 5/7/10/12/14 同款）：非 tie 分歧 = 0；tie-flip 逐例 logit 证据。
prompt 集（≥16，覆盖训练各源域 + 人源 + 随机；imgvr 下载完成后经
``--extra-prompts-json`` 追加）：

- C.5 内置 5（lacZ / chr21 / hg38_w0 / hg38_w235 / random）+ hg38 w2–w4 附加 3
- gtdb 留出编码区 2（step12_retrain.json 落盘 prompt）
- ncbi CDS 2、random 第二种子 1
- OG2 各源首序列各 1（mrna / ncrna / organelle / promoters / euk）

采样等价（KL）：3 prompt × 12 链 × 512 token，spec vs native 同温度同 top_k；
一元/联合/条件 KL + TVD，并附 even/odd 原生自举噪声地板（与测试套件同法）。

用法（GPU1 或空闲卡，单卡独占）::

    CUDA_VISIBLE_DEVICES=1 HF_HOME=./hf_home \\
      /home/dh/miniconda3/envs/evo2/bin/python -u scripts/bench_step15_lossless.py \\
      --greedy-ckpt benchmarks/phasec_ckpts/L27_d1024_g7_offline_step14.pt \\
      --greedy-ckpt benchmarks/phasec_ckpts/S3_d1024_g7_step12.pt \\
      --kl-ckpt benchmarks/phasec_ckpts/L27_d1024_g7_offline_step14.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / "hf_home"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmarks" / "step15_losslessness.json"
SEED_GREEDY = 20260824  # Step 5/7/10/12/14 贪心同种子
SEED_STEP6 = 20260821
SEED_KL_NATIVE = 7001
SEED_KL_SPEC = 7002
CTX = 1024
VOCAB = 512
# 评估 prompt 源键 → OG2 目录名（与 dump_c1_dataset.SOURCE_SUB 同映射）
OG2_PROMPT_SUBDIR = {
    "mrna": "mrna_splice_promoter",
    "ncrna": "ncrna",
    "organelle": "organelle",
    "promoters": "promoters",
    "euk": "eukaryotic_genic_windows",
}


def log(msg: str) -> None:
    print(f"[step15-lossless] {msg}", flush=True)


def load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dump(out: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    tmp.replace(path)
    log(f"已写入 {path}")


# ---------------------------------------------------------------------------
# prompt 集
# ---------------------------------------------------------------------------


def _og2_first_seq(sub: str, min_len: int = 1200) -> str:
    """首条完整 chunk 的序列；短记录源（如 promoters）用 N 分隔拼接到达 min_len。"""
    from train.data import STITCH_SEP
    from train.step12_data import complete_train_chunks, iter_og2_jsonl_records
    from specdec.genome import clean_seq

    files = complete_train_chunks(sub)
    if not files:
        raise RuntimeError(f"{sub} 无完整 chunk")
    buf = ""
    for path in files[:1]:
        for _rec, text in iter_og2_jsonl_records(path):
            seq = clean_seq(text)
            if not seq:
                continue
            buf = seq if not buf else buf + STITCH_SEP + seq
            if len(buf) >= min_len:
                return buf
    raise RuntimeError(f"{sub} 首 chunk 拼接后仍 <{min_len}bp")


def build_prompt_suite(tokenizer, ba, bss) -> list[dict]:
    """≥16 prompt；记录 region/source，全部 CTX=1024。"""
    hg38 = ba.load_hg38_windows(ba.HG38_CSV)
    repeat_sel = ba.select_repeat_windows(hg38, CTX)
    suite = list(ba.build_prompts(tokenizer, repeat_sel["windows"], CTX, SEED_STEP6))
    have = {p["name"] for p in suite}

    specs = []
    # hg38 附加窗（repeat 池内 w2–w4）
    for w in hg38[2:5]:
        specs.append({"name": f"hgw{w['index']}", "region": "repeat", "seq": w["seq"]})
    # gtdb 留出编码区（与 C.5 同源）
    c5 = load_mod("step12_c5_retest_mod", _SCRIPTS / "train" / "step12_c5_retest.py")
    for s in c5.build_gtdb_prompt_specs():
        specs.append({"name": s["name"], "region": "coding", "seq": s["seq"]})
    # ncbi CDS 2 条
    from train.data import load_ncbi_chunks

    for r in load_ncbi_chunks("train")[:2]:
        specs.append({"name": f"ncbi_{r.name}", "region": "coding", "seq": r.seq})
    # 随机第二种子
    specs.append({"name": "random_s2", "region": "random", "seq": ba.synthetic_random_acgt(CTX, 99)})
    # OG2 各源首序列
    for key, sub in OG2_PROMPT_SUBDIR.items():
        specs.append({"name": f"og2_{key}_first", "region": key, "seq": _og2_first_seq(sub)})

    for s in specs:
        if s["name"] in have:
            continue
        ids, used = ba.tokenize_text(tokenizer, s["seq"], CTX)
        suite.append(
            {
                "name": s["name"],
                "region": s.get("region", "?"),
                "source": s.get("seq", "")[:0] or s["name"],
                "ctx": CTX,
                "offset": 0,
                "used_prefix": used[:80],
                "ids": ids,
            }
        )
    log(f"prompt 集合 {len(suite)} 条：" + ", ".join(p["name"] for p in suite))
    return suite


def load_extra_prompts(tokenizer, ba, path: Path) -> list[dict]:
    """--extra-prompts-json：``[{name, region, seq}]`` 或 ``{"prompts": [...]}``
    （step15_imgvr_eval 产物为后者）。"""
    obj = json.loads(Path(path).read_text())
    specs = obj.get("prompts") if isinstance(obj, dict) else obj
    out = []
    for s in specs:
        ids, used = ba.tokenize_text(tokenizer, s["seq"], CTX)
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
    from specdec.block.loop import native_greedy_reference, speculative_generate

    bss = load_mod("bench_state_slicing_mod", _SCRIPTS / "bench_state_slicing.py")
    bss.GAMMA = gamma
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
        nat_ids, nat_logits = bss.vortex_greedy(model, tokenizer, p["ids"], n_tokens)
        ref_ids, _ip, ref_logits = native_greedy_reference(model, p["ids"], n_tokens, record_logits=True)
        case = bss.summarize_greedy_case(name, spec, nat_ids, nat_logits, ref_ids, ref_logits, t_spec, 0.0)
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
    from specdec.block.loop import native_sample_reference, speculative_generate

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
    ap = argparse.ArgumentParser(description="Step 15 无损性扩展（贪心等价 + 采样 KL/SD）")
    ap.add_argument("--greedy-ckpt", action="append", default=[],
                    help="贪心对拍 checkpoint（可多次；如 xxx.pt,tag 或纯路径）")
    ap.add_argument("--kl-ckpt", type=str, default=None, help="采样 KL 测量 checkpoint")
    ap.add_argument("--n-tokens", type=int, default=1024)
    ap.add_argument("--kl-chains", type=int, default=12)
    ap.add_argument("--kl-len", type=int, default=512)
    ap.add_argument("--kl-prompts", type=str, default="lacz_coding,chr21_intergenic,gtdb_DPLL01_coding")
    ap.add_argument("--extra-prompts-json", type=str, default=None,
                    help="追加 prompt [{name,region,seq}]（imgvr valid 落地后用）")
    ap.add_argument("--suite", choices=["full", "extra"], default="full",
                    help="full=内置 ≥17 prompt+追加；extra=只跑追加（快速抽查）")
    ap.add_argument("--results", type=str, default=str(RESULTS))
    args = ap.parse_args()
    results_path = Path(args.results)

    ba = load_mod("bench_acceptance_mod", _SCRIPTS / "bench_acceptance.py")
    bss = load_mod("bench_state_slicing_mod", _SCRIPTS / "bench_state_slicing.py")

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
    from specdec.block.neural_draft import NeuralDraftModel

    log("加载 Evo2 7B（use_kernels=True）…")
    evo = Evo2("evo2_7b", use_kernels=True)
    model = evo.model
    model.eval()
    tokenizer = evo.tokenizer

    prompts = build_prompt_suite(tokenizer, ba, bss)
    if args.extra_prompts_json:
        extra = load_extra_prompts(tokenizer, ba, Path(args.extra_prompts_json))
        log(f"追加 {len(extra)} 条 extra prompt")
        prompts = (extra if args.suite == "extra" else prompts + extra)
        if args.suite == "extra" and not extra:
            raise RuntimeError("--suite extra 但 --extra-prompts-json 为空")
    out["prompt_names"] = [p["name"] for p in prompts]

    greedy_all = out.setdefault("greedy", [])
    for spec in args.greedy_ckpt:
        ckpt, _, tag = spec.partition(",")
        ckpt = str(REPO / ckpt) if not os.path.isabs(ckpt) else ckpt
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
        ckpt = str(REPO / args.kl_ckpt) if not os.path.isabs(args.kl_ckpt) else args.kl_ckpt
        tag = Path(ckpt).stem
        nd = NeuralDraftModel.from_checkpoint(model, ckpt, device="cuda:0")
        want = [x.strip() for x in args.kl_prompts.split(",") if x.strip()]
        kl_prompts = [p for p in prompts if p["name"] in want]
        missing = [w for w in want if w not in {p["name"] for p in prompts}]
        if missing:
            log(f"警告：KL prompt 不在集合内 {missing}（可用 {[p['name'] for p in prompts]}）")
        out["sampling_kl"] = run_kl(model, nd, kl_prompts, args.kl_chains, args.kl_len, int(nd.gamma), tag)
        dump(out, results_path)
        nd.close()

    log("完成。")


if __name__ == "__main__":
    main()
