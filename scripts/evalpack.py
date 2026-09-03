"""评测 prompt 包 + 贪心/采样指标工具集（公开仓库自包含）。

论文 24-prompt 套件的序列随包发布于 ``evspark/data/eval_prompts_24.json``
（5 base + 2 gtdb 留出 + 17 扩展，seq 已截断至 ctx=1024），评测脚本不再依赖
hg38 窗口 CSV / OpenGenome2 / NCBI 原始语料来重建 prompt。

本模块集中三件事：

1. ``load_prompt_pack``：读 prompt 包并 tokenize 成评测记录（含 ids 张量）；
2. 网格指标：``run_seed`` / ``metrics_from_ks``（确定性种子、τ/位置条件接受率/
   投影加速，口径与论文一致）；
3. 贪心对拍：``vortex_greedy``（原生参照）+ ``compare_greedy`` /
   ``summarize_greedy_case``（非 tie 分歧必须为 0，tie-flip 逐例 logit 证据）。
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evspark.specdec.stats import position_conditional_accept_rates  # noqa: E402
from evspark.specdec.verifier import VerifyResult  # noqa: E402

REPO = _ROOT
# prompt 包随 evspark 包发布（pip 安装后同样可用）
import evspark  # noqa: E402

PROMPT_PACK = Path(evspark.__file__).resolve().parent / "data" / "eval_prompts_24.json"

CTX = 1024
TIE_GAP = 1e-3  # top1-top2 logit 间隙小于此值视为 bf16 数值 tie
T1_MS = 19.413  # Plan 01 实测：单步逐 token 解码（4090, bf16）
TV8_MS = 21.269  # Plan 01 实测：γ=8 块验证前向（Tv 在 γ∈[4,16] 几乎平坦）

HOMOPOLY_MIN_RUN = 6
ENTROPY_K = 5
UNIQUE_K = 10


# ---------------------------------------------------------------------------
# prompt 包
# ---------------------------------------------------------------------------


def tokenize_prompt(tokenizer, text: str, ctx: int = CTX, device: str = "cuda:0"):
    """取前 ``ctx`` bp 并 tokenize 成 [1, ctx] 张量；返回 (ids, used)。"""
    if len(text) < ctx:
        raise ValueError(f"序列长度 {len(text)} < ctx={ctx}")
    sub = text[:ctx]
    ids = tokenizer.tokenize(sub)
    t = torch.tensor(np.asarray(ids), dtype=torch.long)[None].to(device)
    return t, sub


def load_prompt_pack(
    tokenizer,
    path: str | Path | None = None,
    ctx: int = CTX,
    device: str = "cuda:0",
    names: list[str] | None = None,
) -> list[dict]:
    """加载 prompt 包 → 评测记录列表（name/region/ctx/ids/prompt_complexity 等）。

    ``names`` 给定时按名单过滤（保持包内顺序）。
    """
    pack_path = Path(path) if path else PROMPT_PACK
    obj = json.loads(pack_path.read_text())
    specs = obj.get("prompts") if isinstance(obj, dict) else obj
    want = set(names) if names else None
    out = []
    for s in specs:
        if want is not None and s["name"] not in want:
            continue
        ids, used = tokenize_prompt(tokenizer, s["seq"], ctx, device)
        out.append(
            {
                "name": s["name"],
                "region": s.get("region", "?"),
                "source": s.get("source", s["name"]),
                "ctx": ctx,
                "offset": 0,
                "used_prefix": used[:80],
                "ids": ids,
                "prompt_complexity": score_prompt_span(used, ctx),
            }
        )
    if want is not None:
        missing = want - {p["name"] for p in out}
        if missing:
            raise RuntimeError(f"prompt 包 {pack_path} 缺少: {sorted(missing)}")
    return out


# ---------------------------------------------------------------------------
# prompt 复杂度（元数据，不进判定）
# ---------------------------------------------------------------------------


def homopolymer_frac(seq: str, min_run: int = HOMOPOLY_MIN_RUN) -> float:
    n = 0
    i = 0
    L = len(seq)
    while i < L:
        j = i + 1
        while j < L and seq[j] == seq[i]:
            j += 1
        if j - i >= min_run:
            n += j - i
        i = j
    return (n / L) if L else 0.0


def kmer_entropy(seq: str, k: int = ENTROPY_K) -> float:
    n = len(seq) - k + 1
    if n <= 0:
        return float("nan")
    counts = Counter(seq[i : i + k] for i in range(n))
    ent = 0.0
    for v in counts.values():
        p = v / n
        ent -= p * float(np.log2(p))
    return float(ent)


def unique_kmer_rate(seq: str, k: int = UNIQUE_K) -> float:
    n = len(seq) - k + 1
    if n <= 0:
        return 1.0
    return len({seq[i : i + k] for i in range(n)}) / float(n)


def score_prompt_span(seq: str, ctx: int = CTX) -> dict:
    span = seq[:ctx]
    return {
        "entropy_k5": kmer_entropy(span, ENTROPY_K),
        "homopolymer_frac6": homopolymer_frac(span, HOMOPOLY_MIN_RUN),
        "unique_k10": unique_kmer_rate(span, UNIQUE_K),
        "prefix80": span[:80],
    }


# ---------------------------------------------------------------------------
# 网格指标
# ---------------------------------------------------------------------------


def run_seed(prompt_name: str, drafter_name: str, base: int) -> int:
    """稳定、与遍历顺序无关的逐 prompt 种子。"""
    h = 0
    for ch in f"{prompt_name}|{drafter_name}":
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return int(base + (h % 100000))


def _results_from_ks(ks: list[int], gamma: int) -> list[VerifyResult]:
    out = []
    for k in ks:
        k = int(k)
        mask = np.zeros(gamma, dtype=bool)
        if k > 0:
            mask[:k] = True
        out.append(
            VerifyResult(
                tokens=np.zeros(k + 1, dtype=np.int64),
                accepted_len=k,
                accepted_mask=mask,
                from_residual=False,
                fallback=False,
            )
        )
    return out


def metrics_from_ks(
    ks: list[int],
    gamma: int,
    n_tokens: int,
    wall_s: float,
    t1_ms: float = T1_MS,
    tv_ms: float = TV8_MS,
) -> dict:
    """逐轮接受数 ks → τ/位置条件接受率/tok\\ s/投影加速（论文口径）。"""
    ks_arr = np.asarray(ks, dtype=np.int64)
    mean_k = float(ks_arr.mean()) if ks_arr.size else 0.0
    mean_tau = mean_k + 1.0
    pos = position_conditional_accept_rates(_results_from_ks(ks, gamma))
    tok_s = (n_tokens / wall_s) if wall_s > 0 else float("nan")
    proj = mean_tau * t1_ms / tv_ms if tv_ms > 0 else float("nan")
    return {
        "n_rounds": int(ks_arr.size),
        "mean_k": mean_k,
        "mean_tau": mean_tau,
        "pos_cond_accept": [None if not np.isfinite(x) else float(x) for x in pos],
        "wall_s": wall_s,
        "tok_s": tok_s,
        "proj_speedup_slice": proj,
    }


# ---------------------------------------------------------------------------
# 贪心对拍（非 tie 分歧 = 0；tie-flip 逐例 logit 证据）
# ---------------------------------------------------------------------------


def vortex_greedy(model, tokenizer, prompt_ids, n_tokens: int):
    """原生 vortex 逐 token 贪心参照（cached_generation 路径）。"""
    from vortex.model.generation import Generator

    g = Generator(model, tokenizer, top_k=1, top_p=0.0, temperature=1.0)
    with torch.inference_mode():
        gen, scores, _ip = g.generate(
            device="cuda:0",
            input_ids=prompt_ids,
            num_tokens=n_tokens,
            cached_generation=True,
            print_generation=False,
            verbose=False,
            stop_at_eos=False,
            max_seqlen=int(prompt_ids.shape[1]) + n_tokens + 2,
        )
    tokens = gen[0].detach().cpu().numpy().astype(np.int64)
    logits = scores[0].detach().float().cpu().numpy().astype(np.float64)
    return tokens, logits


def first_disagreement(a: np.ndarray, b: np.ndarray) -> int | None:
    n = min(len(a), len(b))
    for j in range(n):
        if int(a[j]) != int(b[j]):
            return j
    if len(a) != len(b):
        return n
    return None


def top_gap(row: np.ndarray) -> tuple[float, int, int]:
    row = np.asarray(row, dtype=np.float64).ravel()
    top2 = np.argpartition(row, -2)[-2:]
    top2 = top2[np.argsort(row[top2])[::-1]]
    return float(row[top2[0]] - row[top2[1]]), int(top2[0]), int(top2[1])


def compare_greedy(spec_ids, spec_logits, nat_ids, nat_logits) -> dict:
    """非 tie 必须为 0；tie-flip / path-delta 落盘（判据与论文一致）。"""
    j = first_disagreement(spec_ids, nat_ids)
    rec = {
        "n_tokens": int(len(spec_ids)),
        "n_match_prefix": int(len(spec_ids) if j is None else j),
        "disagree_at": j,
        "n_nontie_disagree": 0,
        "n_tie_flip": 0,
        "tie_flips": [],
    }
    if j is None:
        return rec
    spec_gap, spec_a, spec_b = top_gap(spec_logits[j])
    nat_gap, nat_a, nat_b = top_gap(nat_logits[j])
    max_abs = float(np.max(np.abs(spec_logits[j] - nat_logits[j])))
    min_gap = min(spec_gap, nat_gap)
    plan_tie = (spec_gap < TIE_GAP) or (nat_gap < TIE_GAP)
    path_explains = max_abs + 1e-12 >= min_gap
    is_numeric = plan_tie or path_explains
    entry = {
        "pos": int(j),
        "spec_token": int(spec_ids[j]),
        "native_token": int(nat_ids[j]),
        "spec_argmax": spec_a,
        "native_argmax": nat_a,
        "spec_second": spec_b,
        "native_second": nat_b,
        "spec_top1_top2_gap": spec_gap,
        "native_top1_top2_gap": nat_gap,
        "logit_abs_diff_at_spec": float(
            abs(spec_logits[j, int(spec_ids[j])] - nat_logits[j, int(spec_ids[j])])
        ),
        "logit_abs_diff_at_native": float(
            abs(spec_logits[j, int(nat_ids[j])] - nat_logits[j, int(nat_ids[j])])
        ),
        "max_abs_dlogit": max_abs,
        "plan_tie": plan_tie,
        "path_delta_explains": path_explains,
    }
    if is_numeric:
        rec["n_tie_flip"] = 1
        rec["tie_flips"].append(entry)
    else:
        rec["n_nontie_disagree"] = 1
        rec["nontie"] = entry
    return rec


def summarize_greedy_case(
    name: str,
    spec,
    nat_ids,
    nat_logits,
    ref_ids,
    ref_logits,
    t_spec: float,
    t_nat: float,
    gamma: int,
) -> dict:
    cmp_vortex = compare_greedy(spec.emitted_ids, spec.emitted_logits, nat_ids, nat_logits)
    cmp_ref = compare_greedy(spec.emitted_ids, spec.emitted_logits, ref_ids, ref_logits)
    mean_k = float(np.mean([r.k for r in spec.rounds_log])) if spec.rounds_log else 0.0
    n_reject = sum(1 for r in spec.rounds_log if r.k < gamma)
    return {
        "name": name,
        "rollback": spec.rollback,
        "t_spec_s": round(t_spec, 3),
        "t_vortex_s": round(t_nat, 3),
        "n_rounds": len(spec.rounds_log),
        "mean_k": round(mean_k, 4),
        "n_reject_rounds": n_reject,
        "vs_vortex": cmp_vortex,
        "vs_step_ref": cmp_ref,
        "spec_head": [int(x) for x in spec.emitted_ids[:16]],
        "vortex_head": [int(x) for x in nat_ids[:16]],
    }
