"""Step 15 imgvr 留出评估集构造（plans/17 §1.4）。

从 ``data_imgvr_valid_chunk1.jsonl.gz``（train chunk 未用的官方 valid 划分，
天然留出）取最长 contig，照 ``step12_data.build_gtdb_prompt_meta`` 的 ORF 密集窗法
构造 C.5 prompt；产出三份消费格式：

- ``benchmarks/step15_imgvr_prompts.json``：``[{name, region, seq, ...}]``
  —— ``step15_c5_suite.py`` 的 extra prompt、``bench_step15_lossless.py
  --extra-prompts-json`` 共用；
- stdout 摘要（contig/ORF 统计）供报告引用。

叙事定位（用户 2026-08-23 指示）：imgvr 是**通用加速模型的域覆盖**数据源与
留出泛化测量，不做噬菌体应用故事线。

用法::

    /home/dh/miniconda3/envs/evo2/bin/python scripts/train/step15_imgvr_eval.py \\
        --n-contig 3 --ctx 1024 --window 2048
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from specdec.genome import clean_seq  # noqa: E402
from train.step12_data import orf_span_bp, pick_orf_dense_window  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
IMGVR_VALID = (
    REPO / "opengenome2" / "json" / "pretraining_or_both_phases"
    / "imgvr_untagged" / "data_imgvr_valid_chunk1.jsonl.gz"
)
OUT = REPO / "benchmarks" / "step15_imgvr_prompts.json"


def iter_contigs(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = clean_seq(str(rec.get("text", "")))
            if text:
                yield str(rec.get("record", "")), text


def main() -> None:
    ap = argparse.ArgumentParser(description="imgvr valid → C.5 留出 prompt 构造")
    ap.add_argument("--valid", type=str, default=str(IMGVR_VALID))
    ap.add_argument("--n-contig", type=int, default=3)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--out", type=str, default=str(OUT))
    args = ap.parse_args()
    path = Path(args.valid)

    # 取最长 n_contig 条（≥ window + ctx 余量）
    best: list[tuple[int, str, str]] = []
    n_rec = 0
    bp = 0
    for rec, seq in iter_contigs(path):
        n_rec += 1
        bp += len(seq)
        if len(seq) < args.window:
            continue
        best.append((len(seq), rec, seq))
    best.sort(key=lambda t: -t[0])
    picked = best[: args.n_contig]
    if len(picked) < args.n_contig:
        raise RuntimeError(
            f"≥{args.window}bp 的 contig 仅 {len(picked)} 条（需 {args.n_contig}；"
            f"valid 共 {n_rec} 条 / {bp:,}bp）"
        )

    specs = []
    for rank, (ln, rec, seq) in enumerate(picked, 1):
        score, si, start = pick_orf_dense_window([seq], window=args.window)
        win = seq[start : start + args.window]
        specs.append(
            {
                "name": f"imgvr_c{rank}_coding",
                "region": "coding",
                "source": f"opengenome2:imgvr_untagged valid {rec}",
                "seq": win[: args.ctx],
                "meta": {
                    "accession": rec,
                    "contig_len": int(ln),
                    "window_start": int(start),
                    "orf_span_bp": int(score),
                    "full_window": win,
                },
            }
        )
        print(
            f"[imgvr-eval] {rec}: contig {ln:,}bp ORF 窗@{start} 最长 ORF {score}bp"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(
            {
                "note": "imgvr valid 官方划分（train 未用）；ORF 密集窗法同 gtdb prompt；"
                        "定位=通用加速模型域覆盖的留出泛化测量（非应用故事线）",
                "valid_stats": {"n_records": n_rec, "bp": bp, "n_ge_window": len(best)},
                "prompts": specs,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )
    tmp.replace(out)
    print(f"[imgvr-eval] 已写出 {len(specs)} 条 prompt → {out}")


if __name__ == "__main__":
    main()
