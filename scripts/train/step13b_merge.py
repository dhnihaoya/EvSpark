"""Step 13b：冒烟池对拍 + 双卡结果合并 + 层组合判定（plans/16 §2.3）。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PIN_PATH = REPO / "benchmarks" / "step13_pool_pin.json"
STEP13_S3 = REPO / "benchmarks" / "step13_layer_rescan_S3.json"
OUT_PATH = REPO / "benchmarks" / "step13b_layer_rescan.json"

SINGLES = ("L6", "L16", "L20", "L23", "L27", "L30")
DOUBLES = ("D16_27", "D27_30")
SCHEME_LAYERS = {
    "L6": ["blocks.6"],
    "L16": ["blocks.16"],
    "L20": ["blocks.20"],
    "L23": ["blocks.23"],
    "L27": ["blocks.27"],
    "L30": ["blocks.30"],
    "D16_27": ["blocks.16", "blocks.27"],
    "D27_30": ["blocks.27", "blocks.30"],
    "S3": ["blocks.6", "blocks.16", "blocks.27"],
}


def load(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def check_smoke(smoke_path: Path, pin_path: Path = PIN_PATH) -> None:
    smoke = load(smoke_path)
    pin = load(pin_path)
    anchor = pin["step13_pool_stats_anchor"]
    stats = smoke["env"]["pool_stats"]
    mismatches = []
    for src, exp in anchor.items():
        got = stats.get(src) or {}
        if int(got.get("n_records") or -1) != int(exp["n_records"]) or int(got.get("bp") or -1) != int(exp["bp"]):
            mismatches.append((src, exp, got))
    if mismatches:
        raise SystemExit(f"池与 Step 13 锚点不一致，停查不强跑: {mismatches}")
    hold = set(smoke["env"].get("gtdb_holdout_pin") or smoke["env"].get("gtdb_holdout") or [])
    if smoke["env"].get("gtdb_holdout"):
        hold = set(smoke["env"]["gtdb_holdout"])
    want = set(pin["gtdb_holdout_pin"])
    if hold != want:
        raise SystemExit(f"gtdb holdout={sorted(hold)} ≠ pin {sorted(want)}")
    cells = smoke.get("cells") or []
    if not cells:
        raise SystemExit("冒烟 json 无 cells")
    fr = cells[0].get("freeze") or {}
    # 0.0 是合法值，不能用 `or` 兜底（`0.0 or 1` → 1 会把正确结果判成失败，冒烟实测踩过）
    emb_d = fr.get("embed_max_abs_delta")
    tgt_d = fr.get("target_max_abs_delta")
    if emb_d is None or tgt_d is None or float(emb_d) != 0.0 or float(tgt_d) != 0.0:
        raise SystemExit(f"冻结 Δ 非 0: {fr}")
    print(
        f"[step13b] 冒烟池对拍通过：{ {k: stats[k] for k in anchor} } "
        f"holdout={sorted(want)} freeze Δ=0",
        flush=True,
    )


def _tau(cell: dict) -> float:
    ev = cell.get("eval") or {}
    return float(ev.get("tau_hat"))


def merge_and_judge(paths: list[Path], out: Path = OUT_PATH) -> dict:
    cells = []
    envs = []
    for p in paths:
        d = load(p)
        envs.append(d.get("env"))
        for c in d.get("cells") or []:
            cells.append(c)
    by: dict[str, list[dict]] = {}
    for c in cells:
        tag = c.get("tag") or ""
        sch = c.get("scheme") or tag.split("_")[0]
        by.setdefault(sch, []).append(c)

    summary = {}
    spreads = []
    for sch, cs in by.items():
        taus = [_tau(c) for c in cs]
        mean = sum(taus) / len(taus)
        spread = max(taus) - min(taus) if len(taus) > 1 else 0.0
        spreads.append(spread)
        summary[sch] = {
            "n_seeds": len(taus),
            "taus": taus,
            "mean": mean,
            "spread": spread,
            "tags": [c.get("tag") for c in cs],
            "layers": cs[0].get("layers") or SCHEME_LAYERS.get(sch),
        }
    seed_spread = max(spreads) if spreads else 0.0

    singles = {s: summary[s] for s in SINGLES if s in summary}
    doubles = {s: summary[s] for s in DOUBLES if s in summary}
    best_single = max(singles, key=lambda s: singles[s]["mean"]) if singles else None
    best_double = max(doubles, key=lambda s: doubles[s]["mean"]) if doubles else None
    l27 = summary.get("L27")
    s3 = summary.get("S3")

    upgrade = False
    reason = []
    n_layers = 1
    scheme = "L27"
    layers = ["blocks.27"]

    if not singles or l27 is None:
        upgrade = True
        reason.append("缺单层或 L27 格子，无法按规则判定")
    elif seed_spread > 0.05:
        # 噪声吞掉层间差 → 维持 L27，不追加预算
        scheme, n_layers, layers = "L27", 1, ["blocks.27"]
        reason.append(f"种子极差 {seed_spread:.4f} > 0.05，层选在噪声内无差异，维持 L27 单层")
    else:
        assert best_single is not None and l27 is not None
        gap_vs_l27 = singles[best_single]["mean"] - l27["mean"]
        if gap_vs_l27 <= seed_spread:
            scheme, n_layers, layers = "L27", 1, ["blocks.27"]
            reason.append(
                f"单层榜首 {best_single} mean={singles[best_single]['mean']:.4f} "
                f"vs L27 {l27['mean']:.4f}，差 {gap_vs_l27:.4f} ≤ 种子极差 {seed_spread:.4f} → incumbent L27"
            )
        else:
            scheme = best_single
            n_layers, layers = 1, list(SCHEME_LAYERS[best_single])
            reason.append(
                f"单层榜首 {best_single} 显著优于 L27（Δ={gap_vs_l27:.4f} > 极差 {seed_spread:.4f}）"
            )

        if best_double and best_single:
            dmean = doubles[best_double]["mean"]
            smean = singles[best_single]["mean"] if scheme in singles else singles[best_single]["mean"]
            # 层数：最佳双层 vs 最佳单层（判定后的单层赢家）
            single_mean = summary[scheme]["mean"] if scheme in singles else smean
            gap_d = dmean - single_mean
            if gap_d <= seed_spread:
                reason.append(
                    f"最佳双层 {best_double} {dmean:.4f} vs 最佳单层 {scheme} {single_mean:.4f}，"
                    f"差 {gap_d:.4f} ≤ 极差 → 落 1 层"
                )
            elif gap_d >= 0.05:
                scheme = best_double
                n_layers, layers = 2, list(SCHEME_LAYERS[best_double])
                reason.append(
                    f"最佳双层 {best_double} 超单层 {gap_d:.4f} ≥ 0.05 且超极差 → 落 2 层"
                )
            else:
                reason.append(
                    f"最佳双层 {best_double} 超单层 {gap_d:.4f} 过极差但 <0.05，维持 1 层"
                )

        if s3 and n_layers <= 2:
            ref = summary[scheme]["mean"]
            gap_s3 = s3["mean"] - ref
            if gap_s3 > seed_spread and gap_s3 >= 0.05:
                upgrade = True
                reason.append(
                    f"S3 mean={s3['mean']:.4f} 显著超当前赢家 {scheme} {ref:.4f} "
                    f"（Δ={gap_s3:.4f}）→ 升级调度对话（3 层 3.8TB 需用户裁量）"
                )

        if s3 and all(s in summary for s in SINGLES):
            best_s_mean = max(singles[s]["mean"] for s in singles)
            if best_s_mean - s3["mean"] > seed_spread and best_s_mean - s3["mean"] >= 0.05:
                upgrade = True
                reason.append(
                    f"单层整体显著优于 S3（{best_s_mean:.4f} vs {s3['mean']:.4f}），存在未理解因素，升级"
                )

    step14_tag = {
        "L27": "L27_d1024_g7_100M_step14",
        "S3": "S3_d1024_g7_100M_step14",
        "D16_27": "D16_27_d1024_g7_100M_step14",
    }.get(scheme, f"{scheme}_d1024_g7_100M_step14")

    decision = {
        "scheme": scheme,
        "n_layers": n_layers,
        "layers": layers,
        "step14_tag": step14_tag,
        "upgrade": upgrade,
        "seed_spread": seed_spread,
        "best_single": best_single,
        "best_double": best_double,
        "reason": reason,
    }
    out_obj = {
        "env": envs[0] if envs else {},
        "n_cells": len(cells),
        "cells": [
            {
                "tag": c.get("tag"),
                "scheme": c.get("scheme"),
                "seed": c.get("seed"),
                "layers": c.get("layers"),
                "n_positions": c.get("n_positions"),
                "tau_hat": _tau(c),
                "freeze": c.get("freeze"),
                "ckpt": c.get("ckpt"),
                "eval": c.get("eval"),
            }
            for c in cells
        ],
        "by_scheme": summary,
        "decision": decision,
        "status": "done",
    }
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(out_obj, fh, indent=2, ensure_ascii=False)
    tmp.replace(out)
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    print(f"[step13b] 已写入 {out}", flush=True)
    if out.resolve() == OUT_PATH.resolve():
        _append_report(summary, decision, seed_spread)
    if upgrade:
        raise SystemExit(2)
    return out_obj


def _append_report(summary: dict, decision: dict, seed_spread: float) -> None:
    path = REPO / "notes" / "step13_evidence_hardening_report.md"
    if not path.exists():
        return
    text = path.read_text()
    marker = "## 3b — 层组合加固复扫"
    rows = []
    for sch, rec in sorted(summary.items()):
        taus = " / ".join(f"{t:.4f}" for t in rec["taus"])
        rows.append(
            f"| {sch} | {rec['n_seeds']} | {taus} | {rec['mean']:.4f} | {rec['spread']:.4f} |"
        )
    block = (
        f"\n{marker}（plans/16，2026-08-22）\n\n"
        f"18 格 × 3M × 2 种子，池锁定 Step 13（`benchmarks/step13_pool_pin.json`）。"
        f"种子极差（全格最大）={seed_spread:.4f}。\n\n"
        f"| scheme | 种子数 | τ̂ | mean | 极差 |\n|---|---:|---|---:|---:|\n"
        + "\n".join(rows)
        + "\n\n**判定**：scheme="
        + str(decision["scheme"])
        + f"，{decision['n_layers']} 层 {decision['layers']}。"
        + (" **升级调度对话。** " if decision["upgrade"] else " ")
        + " ".join(decision["reason"])
        + "\n\n"
    )
    if marker in text:
        pre, rest = text.split(marker, 1)
        # 换成新块：丢掉旧 3b 直到下一个 ## 或文末
        nxt = rest.find("\n## ")
        tail = rest[nxt:] if nxt >= 0 else ""
        path.write_text(pre.rstrip() + "\n" + block + tail)
    else:
        path.write_text(text.rstrip() + "\n" + block)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 13b 冒烟校验 / 合并判定")
    p.add_argument("--check-smoke", type=str, default=None)
    p.add_argument("--merge", nargs="+", default=None, help="step13b_layer_rescan_{a,b}.json")
    p.add_argument("--out", type=str, default=str(OUT_PATH))
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.check_smoke:
        check_smoke(Path(args.check_smoke))
    if args.merge:
        merge_and_judge([Path(x) for x in args.merge], Path(args.out))
    if not args.check_smoke and not args.merge:
        raise SystemExit("需要 --check-smoke 或 --merge")


if __name__ == "__main__":
    main()
