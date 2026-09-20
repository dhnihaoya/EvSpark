"""小规模 case study 的表格与入稿数字核验；仅用标准库，不加载模型。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics as st


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_case(root):
    folder = Path(root)/"benchmarks/phase5"
    records = []
    for p in folder.glob("*/result.json"):
        row = json.loads(p.read_text())
        row["source"] = p.relative_to(root).as_posix()
        records.append(row)
    rows = sorted((r for r in records if r["stage"] == "pilot"),
                  key=lambda r: (r["pattern"], r["seed"], r["backend"] != "native"))
    calibration = sorted((r for r in records if r["stage"] == "calibration"),
                         key=lambda r: (r["backend"] != "native", r["batch"]))
    assert len(rows) == 8 and len(calibration) == 7
    arms = {b: [r for r in rows if r["backend"] == b] for b in ("native", "evspark")}
    pairs = list(zip(arms["native"], arms["evspark"]))
    assert all((a["pattern"], a["seed"]) == (b["pattern"], b["seed"]) for a, b in pairs)
    ratios = [a["total_s"]/b["total_s"] for a, b in pairs]
    horizon = min(sum(r["budget_total_s"] for r in arm) for arm in arms.values())
    metrics = {"n_pairs": len(pairs), "n_seed_groups": len({r["seed"] for r in rows}),
               "median_speedup": st.median(ratios), "speedup_range": [min(ratios), max(ratios)],
               "common_budget_minutes": horizon/60, "mean_minutes": {},
               "checker_auc_means": {}, "pooled_4mer_entropy": {},
               "qualified_unique_within_budget": {}}
    for backend, arm in arms.items():
        metrics["mean_minutes"][backend] = st.mean(r["total_s"] for r in arm)/60
        metrics["checker_auc_means"][backend] = st.mean(r["checker"]["checker_auc"] for r in arm)
        counts = Counter(r["sequence"][i:i+4] for r in arm for i in range(len(r["sequence"])-3)
                         if set(r["sequence"][i:i+4]) <= set("ACGT"))
        total = sum(counts.values())
        metrics["pooled_4mer_entropy"][backend] = -sum(n/total*math.log2(n/total) for n in counts.values())
        elapsed, seen = 0., set()
        for r in arm:
            elapsed += r["budget_total_s"]
            if elapsed > horizon:
                break
            if r["qualified"]:
                seen.add(r["sequence"])
        metrics["qualified_unique_within_budget"][backend] = len(seen)
    metrics["evspark_scoring_share_percent"] = 100*sum(r["stages_s"]["scoring"] for r in arms["evspark"])/sum(r["total_s"] for r in arms["evspark"])
    return rows, calibration, metrics


def tables(root):
    rows, calibration, _ = read_case(root)
    selection = json.loads((Path(root)/"benchmarks/phase5/selected_batches.json").read_text())["selection"]
    names = {"native": "Native", "evspark": "EvSpark"}
    output = [r"""% 中文说明：直接从原始 result.json 生成，勿手工改表内数字。
\begin{table}[tbp]
\centering\small
\caption{Full-workflow batch calibration, separate from the case-study outputs.
The lowest complete time selects each backend's batch (bold), using the same
3,072\,bp M-pattern workload and calibration seed 2026091801.
The native batch-4 output fails the predictive qualification threshold;
quality is reported but does not select or exclude a batch.}
\label{tab:reg-calibration}
\begin{tabular}{@{}lrrr@{}}
\toprule
Backend & Batch & Complete time (min) & Checker AUROC \\
\midrule"""]
    for r in calibration:
        time = f'{r["total_s"]/60:.3f}'
        if r["batch"] == selection[r["backend"]]["batch"]:
            time = r"\textbf{"+time+"}"
        output.append(f'{names[r["backend"]]} & {r["batch"]} & {time} & {r["checker"]["checker_auc"]:.4f} '+r"\\")
    output.append(r"""\bottomrule
\end{tabular}
\end{table}

\begin{table}[tbp]
\centering\small
\caption{Every complete case-study design. Native uses batch~8; EvSpark uses
batch~4. M and S are the 768/768\,bp and 384/1,152\,bp open/closed targets.
Seeds~1/2 are 2026091901/2026091902, reused across targets.
All eight outputs meet both AUROC thresholds of 0.90.
Search and checker columns report predictive AUROC, not experimental function.}
\label{tab:reg-designs}
\begin{tabular}{@{}lrlrrr@{}}
\toprule
Pattern & Seed & Backend & Time (min) & Search & Checker \\
\midrule""")
    for r in rows:
        output.append(f'{"M" if r["pattern"] == "medium" else "S"} & {r["seed"]-2026091900} & {names[r["backend"]]} & {r["total_s"]/60:.3f} & {r["guide"]["ensemble_auc"]:.4f} & {r["checker"]["checker_auc"]:.4f} '+r"\\")
    output.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    return "\n".join(output)


def check_case(root, check, paper=None):
    """独立复算已入稿的均值、比值、预算、熵、表格及来源哈希。"""
    root = Path(root).resolve()
    folder = root/"benchmarks/phase5"
    manuscript = Path(paper) if paper else root/"paper/latex_v3/main.tex"
    text = manuscript.read_text()
    rows, calibration, metrics = read_case(root)
    audit = json.loads((folder/"pilot_audit.json").read_text())
    check("P5 审计状态", int(audit["status"] == "passed"), 1, tol=0)
    check("P5 已完成设计数", len(rows), 8, tol=0)
    check("P5 设计与后端组合无重复", len({(r['pattern'],r['seed'],r['backend']) for r in rows}), 8, tol=0)
    for item in audit["sources"]:
        for name, expected in item["files"].items():
            path = root/Path(item["source"]).parent/name
            check(f"P5 原始 SHA {path.parent.name}/{name}", int(sha(path) == expected), 1, tol=0)
    check("P5 轨迹候选总数", sum(r["candidate_count"] for r in audit["sources"]), 5640, tol=0)
    for path, expected in audit["calibration_sources"].items():
        check(f"P5 校准 SHA {Path(path).parent.name}", int(sha(root/path) == expected), 1, tol=0)
    for r in rows:
        key = f'{r["pattern"]}/{r["seed"]}/{r["backend"]}'
        check(f"P5 完整输出 {key}", int(r["status"] == "complete" and r["timing_eligible"] and not r["resumed"] and len(r["sequence"]) == 3072), 1, tol=0)
        check(f"P5 双阈值资格 {key}", int(r["qualified"] == (r["guide"]["ensemble_auc"] >= .90 and r["checker"]["checker_auc"] >= .90)), 1, tol=0)
        check(f"P5 阶段计时求和 {key}", sum(v for k,v in r["stages_s"].items() if k != "recovery"), r["total_s"], tol=1e-6)
    selection = json.loads((folder/"selected_batches.json").read_text())
    summary = json.loads((folder/"pilot_summary.json").read_text())
    for b in ("native", "evspark"):
        best = min((r for r in calibration if r["backend"] == b), key=lambda r:r["total_s"])
        check(f"P5 最快完整 batch {b}", best["batch"], selection["selection"][b]["batch"], tol=0)
        arm = [r for r in rows if r["backend"] == b]
        check(f"P5 全部预测合格 {b}", sum(r["qualified"] for r in arm), 4, tol=0)
        check(f"P5 精确唯一输出 {b}", len({r['sequence'] for r in arm}), 4, tol=0)
        identities = [sum(x == y for x,y in zip(a['sequence'],c['sequence']))/3072 for a,c in itertools.combinations(arm,2)]
        for i,(got,want) in enumerate(zip(identities,summary['arms'][b]['diversity']['aligned_identity'])):
            check(f"P5 两两位置一致率 {b}/{i}", got, want, tol=1e-12)
    frozen = json.loads((folder/"case_study_writing_metrics.json").read_text())
    def compare(got, want, path):
        if isinstance(got, dict):
            assert got.keys() == want.keys(), path
            for k in got: compare(got[k], want[k], path+"/"+k)
        elif isinstance(got, list):
            assert len(got) == len(want), path
            for i,(a,b) in enumerate(zip(got,want)): compare(a,b,path+f"/{i}")
        else:
            check("P5 文字指标 "+path, got, want, tol=1e-10)
    compare(metrics, frozen, "")
    for name in ("phase5_case_study_draft.tex", "phase5_case_methods.tex", "phase5_case_tables.tex"):
        fragment = (root/"paper/latex_v3"/name).read_text().strip()
        check(f"P5 主稿包含 {name}", int(fragment in text), 1, tol=0)
    check("P5 全部表格由原始数据复算", int(tables(root).strip() in text), 1, tol=0)
    # 同时检查正文中的数值，不只比对一个旁路指标文件。
    for phrase in (f'{metrics["mean_minutes"]["native"]:.2f} to {metrics["mean_minutes"]["evspark"]:.2f}',
                   f'{metrics["median_speedup"]:.2f}$\\times$',
                   f'{metrics["speedup_range"][0]:.2f}$\\times$ to {metrics["speedup_range"][1]:.2f}$\\times$',
                   f'{metrics["common_budget_minutes"]:.2f}\\,min',
                   f'{metrics["evspark_scoring_share_percent"]:.1f}\\%'):
        check("P5 正文显示 "+phrase, int(phrase in text), 1, tol=0)
    for name, expected in json.loads((folder/"case_paper_materials.json").read_text())["files"].items():
        check("P5 入稿来源 SHA "+Path(name).name, int(sha(root/name) == expected), 1, tol=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--paper", type=Path)
    ap.add_argument("--write-tables", action="store_true")
    args = ap.parse_args()
    if args.write_tables:
        (args.root/"paper/latex_v3/phase5_case_tables.tex").write_text(tables(args.root))
        return
    failures, count = [], 0
    def check(name, got, want, tol=.005):
        nonlocal count
        count += 1
        if abs(got-want) > tol:
            failures.append({"name": name, "got": got, "want": want})
    check_case(args.root, check, args.paper)
    print(json.dumps({"checks": count, "failures": failures}, ensure_ascii=False, indent=2))
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
