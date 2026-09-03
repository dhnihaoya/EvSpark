"""训练语料池加载：OG2 正式档多源池 + GTDB 留出协议。

按 OG2 源分池加权：

- ``gtdb_v220_imgpr``：GTDB 细菌/古菌基因组 contig（记录带 accession）。
  按 accession 前 6 字符分组（一个 IMG 提交前缀 ≈ 一个基因组），两遍流式：
  第一遍统计各基因组 bp、确定性选 2 个留出基因组；第二遍留出组整组保留
  （评估 + 评测 prompt 用），其余按 keep_prob 子抽样拼接进训练池。
- ``mrna_splice_promoter`` / ``ncrna`` / ``organelle`` / ``promoters`` /
  ``eukaryotic_genic_windows``：逐文件并行子抽样拼接（``gz_intact`` 宽容口径），
  keep_prob 按目标池 bp / 估算原始 bp 自标定。

完整性判据：无 ``.aria2`` 伴侣即完整；有伴侣但 ``gzip -t`` 全流 CRC 通过视为
**陈旧标记**、内容完备（见 ``gz_intact``——双源 aria2 竞写时另一条进程可能攥着
已收完文件的控制文件不放）。注意本模块的宽容口径只覆盖 gtdb 路径
（``iter_og2_jsonl_records`` / ``complete_train_chunks``）；其余 OG2 源走的
``train.data.iter_og2_jsonl`` 仍是旧强校验，两套口径并存是有意的。

自检（只读盘点，不建池）::

    python -m evspark.train.corpora --scan
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evspark.specdec.genome import clean_seq
from evspark.train.data import (
    LabeledSeq,
    OG2_JSON_DIR,
    STITCH_CHUNK_BP,
    iter_og2_jsonl,
    load_ncbi_chunks,
    records_to_chunks,
)

# gz → 原始文本 bp 的经验比率（gtdb valid chunk 实测 3.26；json 逐行开销使
# 短记录源略低，这里统一取 3.2，仅用于 keep_prob 标定，±30% 误差无害）
GZ_RAW_RATIO = 3.2

# 训练池目标 bp（采样有放回，池只需足够多样性；实际 bp 落盘 pool_stats）
TARGET_BP = {
    "gtdb_v220_imgpr": 150_000_000,
    "mrna_splice_promoter": 150_000_000,
    "ncrna": 30_000_000,
    "organelle": 20_000_000,
    "promoters": 5_000_000,
    "eukaryotic_genic_windows": 100_000_000,
}

# gtdb 留出基因组的准入窗：bp 下限保证评估窗与 prompt 足量，上限避免浪费
# 最大基因组；valid_chunk1 的基因组（PAFS01）从训练池排除以保持其留出属性
GTDB_HOLDOUT_MIN_BP = 5_000_000
GTDB_HOLDOUT_MAX_BP = 60_000_000
GTDB_VALID_PREFIX = "PAFS01"

STOPS = ("TAA", "TAG", "TGA")
_REVCOMP = str.maketrans("ACGTN", "TGCAN")


def gz_intact(path: str | Path) -> bool:
    """完整性判定：无 ``.aria2`` 伴侣 → True；有伴侣但 ``gzip -t`` 全流 CRC 通过 →
    True（**陈旧标记**：双源两条 aria2 竞写同一批文件时，另一条进程可能攥着
    已收完文件的控制文件不放，2026-08-22 凌晨 chunk6/7 即此情形——文件字节数
    已与远端一致且 CRC 全过，标记文件不可信，内容校验才是硬证据）；否则 False。
    ``gzip -t`` 对 3GB 文件约 ~55s。"""
    path = Path(path)
    if not Path(str(path) + ".aria2").exists():
        return True
    import subprocess

    return subprocess.run(["gzip", "-t", str(path)], capture_output=True).returncode == 0


def complete_train_chunks(sub: str) -> list[Path]:
    """某 OG2 源目录下完整的 train chunk，按文件名排序。

    完整 = 无 ``.aria2`` 伴侣，或伴侣存在但 gzip 全流 CRC 通过（见 ``gz_intact``）。
    """
    out: list[Path] = []
    for p in sorted((OG2_JSON_DIR / sub).glob("*.jsonl.gz")):
        if "train" not in p.name:
            continue
        if not gz_intact(p):
            continue
        out.append(p)
    return out


def select_chunk_files(available: list[Path], only: set[str] | None) -> list[Path]:
    """按文件名白名单筛 chunk；``only is None`` 时原样返回。缺失则报错。"""
    if only is None:
        return list(available)
    want = set(only)
    files = [p for p in available if p.name in want]
    missing = want - {p.name for p in files}
    if missing:
        raise RuntimeError(
            f"chunk 白名单有未就绪文件: {sorted(missing)}；"
            f"可用 {[p.name for p in available]}"
        )
    return files


def file_meta(p: Path) -> dict:
    return {"file": p.name, "gz_bytes": p.stat().st_size}


def iter_og2_jsonl_records(path: str | Path):
    """同 ``train.data.iter_og2_jsonl`` 但产出 ``(record, text)``。

    仅 gtdb_v220_imgpr 记录带 ``record``（accession）字段；其余源该字段为空串。
    完整性校验与 ``complete_train_chunks`` 同口径（``gz_intact``：无 .aria2，或
    gzip 全流 CRC 通过的陈旧标记）。
    """
    import gzip

    path = Path(path)
    if not path.exists():
        raise RuntimeError(f"{path} 不存在")
    if not gz_intact(path):
        raise RuntimeError(f"{path} 未通过完整性校验（.aria2 伴侣在且 gzip CRC 失败）")
    with gzip.open(path, "rt", encoding="utf-8") as fh:  # type: ignore[arg-type]
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict) or not rec.get("text"):
                continue
            yield str(rec.get("record", "")), str(rec["text"])


def accession_family(record: str) -> str:
    """accession（如 PAFS01000061.1）→ 基因组家族键（提交前缀前 6 字符）。"""
    return record.split(".")[0][:6]


# ---------------------------------------------------------------------------
# gtdb 两遍流式
# ---------------------------------------------------------------------------


def _gtdb_pass1(path: Path) -> dict[str, list[int]]:
    """第一遍：各基因组家族 [n_contigs, bp]。"""
    n_rec: Counter = Counter()
    bp: Counter = Counter()
    for record, text in iter_og2_jsonl_records(path):
        fam = accession_family(record)
        n_rec[fam] += 1
        bp[fam] += len(text)
    return {fam: [int(n_rec[fam]), int(bp[fam])] for fam in n_rec}


def _gtdb_pass2(
    path: Path,
    seed: int,
    keep_prob: float,
    holdout_prefixes: tuple[str, ...],
    exclude_prefixes: tuple[str, ...],
) -> tuple[list[str], dict[str, list[tuple[str, str]]]]:
    """第二遍：训练子抽样序列 + 留出基因组全量 contig（accession 去重由父进程做）。

    注意 holdout 判断必须在 exclude 之前——exclude 集合包含 holdout 家族本身
    （load_gtdb 里 exclude = holdout ∪ {PAFS01}），顺序反了会把留出家族整族
    扔掉、hold 一条不剩（冒烟 #2 实测踩过）。
    """
    rng = np.random.default_rng(seed)
    kept: list[str] = []
    hold: dict[str, list[tuple[str, str]]] = {f: [] for f in holdout_prefixes}
    for record, text in iter_og2_jsonl_records(path):
        fam = accession_family(record)
        if fam in hold:
            hold[fam].append((record, text))
            continue
        if fam in exclude_prefixes:
            continue
        if rng.random() >= keep_prob:
            continue
        seq = clean_seq(text)
        if seq:
            kept.append(seq)
    return kept, hold


def choose_holdout_prefixes(
    stats: dict[str, list[int]],
    n_hold: int = 2,
    seed: int = 20260821,
) -> list[str]:
    """确定性选留出基因组：bp 在准入窗内的候选中按种子抽样。"""
    cand = sorted(
        f
        for f, (_, bp) in stats.items()
        if len(f) == 6 and GTDB_HOLDOUT_MIN_BP <= bp <= GTDB_HOLDOUT_MAX_BP
    )
    if len(cand) < n_hold:
        cand = sorted(
            f for f, (_, bp) in stats.items() if len(f) == 6 and bp >= GTDB_HOLDOUT_MIN_BP
        )
    if len(cand) < n_hold:
        raise RuntimeError(
            f"gtdb 候选留出基因组不足（需 {n_hold}，得 {len(cand)}；"
            f"家族数={len(stats)}，bp≥{GTDB_HOLDOUT_MIN_BP} 的={len(cand)}）"
        )
    rng = np.random.default_rng(seed)
    pick = sorted(str(x) for x in rng.choice(cand, size=n_hold, replace=False).tolist())
    return pick


def load_gtdb(
    seed: int = 20260821,
    n_chunks: int = 2,
    n_holdout: int = 2,
    target_bp: int | None = None,
    only: set[str] | None = None,
    holdout_pin: tuple[str, ...] | None = None,
) -> tuple[list[LabeledSeq], dict]:
    """gtdb_v220_imgpr → (训练池, 元数据含留出基因组 contig)。

    元数据键：chunks / keep_prob / est_raw_bp / holdout（家族统计）/
    holdout_contigs（家族 → [(accession, seq)]，评估与复测 prompt 用）。

    ``only``：chunk 文件名白名单（池锁定，Step 13b）；给定时不再按 ``n_chunks`` 截断。
    ``holdout_pin``：跳过 ``choose_holdout_prefixes``，直接用给定家族；pin 家族
    仍并入 exclude（与自动选出时相同）。
    """
    target_bp = int(target_bp or TARGET_BP["gtdb_v220_imgpr"])
    available = complete_train_chunks("gtdb_v220_imgpr")
    if only is not None:
        files = select_chunk_files(available, only)
    else:
        files = available[:n_chunks]
        if len(files) < n_chunks:
            raise RuntimeError(
                f"gtdb_v220_imgpr 完整 train chunk 不足：需 {n_chunks}，"
                f"得 {len(files)}（{[p.name for p in available]}）；请等下载完成（.aria2 消失）"
            )

    with ProcessPoolExecutor(max_workers=len(files)) as ex:
        per_file = list(ex.map(_gtdb_pass1, files))
    stats: dict[str, list[int]] = {}
    for st in per_file:
        for fam, (n, bp) in st.items():
            cur = stats.setdefault(fam, [0, 0])
            cur[0] += n
            cur[1] += bp

    if holdout_pin:
        holdout = [str(x) for x in holdout_pin]
        missing_pin = [f for f in holdout if f not in stats]
        if missing_pin:
            raise RuntimeError(
                f"gtdb holdout_pin 不在本批基因组家族中: {missing_pin}；"
                f"家族数={len(stats)}"
            )
    else:
        holdout = choose_holdout_prefixes(stats, n_hold=n_holdout, seed=seed)
    exclude = set(holdout)
    if any(GTDB_VALID_PREFIX in s for s in stats):
        exclude.add(GTDB_VALID_PREFIX)

    est_raw = sum(f.stat().st_size for f in files) * GZ_RAW_RATIO
    keep_prob = min(1.0, target_bp / max(est_raw, 1))
    with ProcessPoolExecutor(max_workers=len(files)) as ex:
        per_file2 = list(
            ex.map(
                partial(
                    _gtdb_pass2,
                    keep_prob=keep_prob,
                    holdout_prefixes=tuple(holdout),
                    exclude_prefixes=tuple(sorted(exclude)),
                ),
                files,
                [seed + i for i in range(len(files))],
            )
        )
    kept: list[str] = []
    hold_acc: dict[str, dict[str, str]] = {f: {} for f in holdout}
    for ks, hold in per_file2:
        kept.extend(ks)
        for fam, contigs in hold.items():
            for acc, text in contigs:
                hold_acc[fam].setdefault(acc, clean_seq(text))

    chunks = records_to_chunks(kept, chunk_bp=STITCH_CHUNK_BP)
    pool = [
        LabeledSeq(name=f"gtdb_imgpr:{j}", seq=ch, source="opengenome2:gtdb_v220_imgpr")
        for j, ch in enumerate(chunks)
    ]
    holdout_contigs = {
        fam: sorted(((a, s) for a, s in d.items() if s), key=lambda t: -len(t[1]))
        for fam, d in hold_acc.items()
    }
    meta = {
        "chunks": [file_meta(f) for f in files],
        "keep_prob": keep_prob,
        "est_raw_bp": int(est_raw),
        "n_genome_families": len(stats),
        "holdout": {
            f: {"n_contigs": stats[f][0], "bp": stats[f][1]} for f in holdout
        },
        "excluded_prefixes": sorted(exclude),
        "holdout_contigs": holdout_contigs,
    }
    return pool, meta


# ---------------------------------------------------------------------------
# 其余 OG2 源（逐文件并行；gz_intact 宽容口径，与 gtdb 一致）
# ---------------------------------------------------------------------------


def _load_og2_one(rel: str, seed: int, keep_prob: float) -> list[LabeledSeq]:
    """单文件子抽样拼接。用 ``iter_og2_jsonl_records``（gz_intact 宽容口径），
    使带陈旧 ``.aria2`` 但 CRC 已过的 euk/mrna 增补 chunk 能入池。"""
    path = OG2_JSON_DIR / rel
    rng = np.random.default_rng(int(seed))
    kept: list[str] = []
    for _record, text in iter_og2_jsonl_records(path):
        if rng.random() >= keep_prob:
            continue
        seq = clean_seq(text)
        if seq:
            kept.append(seq)
    chunks = records_to_chunks(kept, chunk_bp=STITCH_CHUNK_BP)
    tag = Path(rel).parent.name
    return [
        LabeledSeq(name=f"og2_{tag}:{j}", seq=ch, source=str(path))
        for j, ch in enumerate(chunks)
    ]


def load_og2_source(
    sub: str,
    seed: int,
    target_bp: int | None = None,
    only: set[str] | None = None,
) -> tuple[list[LabeledSeq], dict]:
    """一个 OG2 源 → (子抽样拼接池, 元数据)。逐文件并行，语义与串行 specs 等价。

    ``only``：chunk 文件名白名单（池锁定）；``None`` 用全部完整 train chunk。
    """
    target_bp = int(target_bp or TARGET_BP[sub])
    files = select_chunk_files(complete_train_chunks(sub), only)
    meta: dict = {"chunks": [file_meta(f) for f in files], "keep_prob": None}
    if not files:
        return [], meta
    est_raw = sum(f.stat().st_size for f in files) * GZ_RAW_RATIO
    keep_prob = min(1.0, target_bp / max(est_raw, 1))
    rels = [f"{sub}/{f.name}" for f in files]
    with ProcessPoolExecutor(max_workers=min(8, len(files))) as ex:
        parts = list(
            ex.map(partial(_load_og2_one, keep_prob=keep_prob), rels, [seed + i for i in range(len(rels))])
        )
    pool = [r for part in parts for r in part]
    meta["keep_prob"] = keep_prob
    meta["est_raw_bp"] = int(est_raw)
    return pool, meta


# ---------------------------------------------------------------------------
# gtdb 留出基因组的评估窗 / ORF 密集编码区 prompt
# ---------------------------------------------------------------------------


def revcomp(seq: str) -> str:
    return seq.translate(_REVCOMP)[::-1]


def orf_span_bp(window: str) -> int:
    """窗内 6 帧最长无终止密码子延伸（bp）。

    量级（供选窗参考）：随机 2kb 窗典型 300–450bp（最长无终止游程
    ≈ ln(682)/−ln(1−3/64) ≈ 135 密码子）；含 ≥300 密码子完整基因的编码窗
    ≥900bp。取 top-k contig 上的全局最大窗，编码窗通常明显胜出。"""
    best = 0
    for strand in (window, revcomp(window)):
        for frame in range(3):
            last_stop = frame - 3
            for pos in range(frame, len(strand) - 2, 3):
                if strand[pos : pos + 3] in STOPS:
                    last_stop = pos
                gap = pos - last_stop
                if gap > best:
                    best = gap
    return best


def pick_orf_dense_window(
    contigs: list[str],
    window: int = 2048,
    stride: int = 256,
    top_k: int = 3,
) -> tuple[int, int, int]:
    """(最长 ORF 延伸 bp, contig 下标, 窗起点)：在最长 top_k 条 contig 上滑窗取最大。"""
    best = (-1, 0, 0)
    for si, seq in enumerate(contigs[:top_k]):
        if len(seq) < window:
            continue
        for start in range(0, len(seq) - window + 1, stride):
            sc = orf_span_bp(seq[start : start + window])
            if sc > best[0]:
                best = (sc, si, start)
    return best


def gtdb_holdout_eval_records(meta: dict, per_family: int = 2) -> list[LabeledSeq]:
    """留出基因组 → 评估记录（每家族最长 per_family 条 contig，末对齐窗由
    ``build_eval_packs``/``holdout_eval_window`` 统一截取；家族整体未参与训练，
    窗内全位置可评估）。"""
    out: list[LabeledSeq] = []
    for fam in sorted(meta.get("holdout_contigs", {})):
        for acc, seq in meta["holdout_contigs"][fam][:per_family]:
            out.append(
                LabeledSeq(
                    name=f"gtdb_holdout_{fam}:{acc}",
                    seq=seq,
                    source=f"opengenome2:gtdb_v220_imgpr/{fam}",
                )
            )
    return out


def build_gtdb_prompt_meta(meta: dict, ctx: int = 1024, window: int = 2048) -> list[dict]:
    """留出基因组 → ORF 密集编码区窗 prompt 描述（训练时落盘；论文套件已内嵌于
    ``evspark/data/eval_prompts_24.json``）。"""
    prompts: list[dict] = []
    for fam in sorted(meta.get("holdout_contigs", {})):
        contigs = [s for _, s in meta["holdout_contigs"][fam]]
        if not contigs:
            continue
        score, si, start = pick_orf_dense_window(contigs, window=window)
        if score < 0:
            continue
        seq = contigs[si][start : start + window]
        prompts.append(
            {
                "name": f"gtdb_{fam}_coding",
                "prefix": fam,
                "window_start": int(start),
                "contig_len": len(contigs[si]),
                "orf_span_bp": int(score),
                "prompt_seq": seq[:ctx],
                "source": f"opengenome2:gtdb_v220_imgpr 留出基因组 {fam}（ORF 密集窗）",
            }
        )
    return prompts


# ---------------------------------------------------------------------------
# 盘点 CLI
# ---------------------------------------------------------------------------


def scan_inventory(include: Iterable[str] | None = None) -> dict:
    """只读盘点。``include`` 限制源集合（默认 ``TARGET_BP`` 全部键）。

    未点名 euk 时不要把正在下的 euk chunk 拉进 ``gz_intact``（每次 gzip -t
    数分钟，且与用户 aria2 争读）。
    """
    inv: dict = {}
    keys = list(include) if include is not None else list(TARGET_BP)
    for sub in sorted(keys):
        files = complete_train_chunks(sub)
        partial = [
            p.name for p in sorted((OG2_JSON_DIR / sub).glob("*.jsonl.gz")) if "train" in p.name and Path(str(p) + ".aria2").exists()
        ]
        inv[sub] = {
            "complete_train_chunks": [p.name for p in files],
            "gz_bytes_total": sum(p.stat().st_size for p in files),
            "est_raw_bp": int(sum(p.stat().st_size for p in files) * GZ_RAW_RATIO),
            # 带 .aria2 标记的 train chunk；其中若同时出现在 complete 列表，
            # 为 gzip CRC 验证过的陈旧标记（见 gz_intact），真未下完者只在标记列表
            "with_aria2_marker": partial,
        }
    return inv


def main() -> None:
    p = argparse.ArgumentParser(description="Step 12 OG2 盘点 / 数据侧自检")
    p.add_argument("--scan", action="store_true", help="只读盘点各源完整 chunk")
    args = p.parse_args()
    if args.scan:
        print(json.dumps(scan_inventory(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# 留出评测记录与池锁定（自历史 step11/step12 训练驱动收敛而来）
# ---------------------------------------------------------------------------

# 留物种泛化窗：每个 holdout 物种取前 2 个 chunk 进 eval（整体未参与训练）
HOLDOUT_EVAL_CHUNKS_PER_SPECIES = 2


def build_holdout_eval_records() -> list[LabeledSeq]:
    """NCBI 编码区 holdout（按物种整体留出）的评估记录，每物种取前若干 chunk。"""
    by_species: dict[str, list[LabeledSeq]] = {}
    for rec in load_ncbi_chunks("holdout"):
        by_species.setdefault(rec.source.rsplit("/", 1)[-1], []).append(rec)
    out: list[LabeledSeq] = []
    for species in sorted(by_species):
        for rec in by_species[species][:HOLDOUT_EVAL_CHUNKS_PER_SPECIES]:
            out.append(LabeledSeq(name=f"ncbi_holdout_{rec.name}", seq=rec.seq, source=rec.source))
    return out


def load_pool_pin(path: str | Path) -> dict:
    """训练池锁定 JSON（保证跨种子/跨格可比）：必须含 ``chunks`` 字段。"""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or "chunks" not in data:
        raise RuntimeError(f"pool pin 缺少 chunks 字段: {path}")
    return data


def parse_holdout_pin(s: str | None) -> tuple[str, ...] | None:
    """逗号分隔的 holdout 前缀串 → tuple；空输入返回 None（不锁定）。"""
    if not s:
        return None
    pin = tuple(x.strip() for x in str(s).split(",") if x.strip())
    return pin or None
