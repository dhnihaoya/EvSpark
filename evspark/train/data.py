"""窗口采样：hg38 + dna_samples，OpenGenome2 fasta 目录接口预留。

训练用每条序列前 90% 连续前缀（与 Step 2 ``split_contiguous`` 同协议）；
hold-out 为末 10%。本工作包不读取 ``opengenome2/``，仅当调用方显式传入
``fasta_dir`` 且文件就绪（无 ``.aria2``）时才扫 fasta。

Step 11 新增（plans/13）：NCBI 多物种编码区 chunk 加载、OG2 完整 jsonl
chunk 流式读取（带 .aria2 完整性校验）、短序列拼接 ``records_to_chunks``
与多源加权采样 ``make_mixed_sampler``。既有函数行为不变。
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evspark.specdec.genome import clean_seq, load_sequences
from evspark.specdec.markov import HOLD_OUT_FRAC, split_contiguous

REPO = Path(__file__).resolve().parents[2]
HG38_CSV = REPO / "data" / "genomes" / "hg38_sequences.csv"
DNA_SAMPLES = REPO / "data" / "dna_samples.json"
OG2_FASTA = REPO / "opengenome2" / "fasta"
NCBI_CODING_DIR = REPO / "data" / "ncbi_coding"
OG2_JSON_DIR = REPO / "opengenome2" / "json" / "pretraining_or_both_phases"

# Step 11：短序列（CDS / ncRNA / promoter）拼接成长 chunk 的分隔符与目标长度
STITCH_SEP = "N" * 16
STITCH_CHUNK_BP = 50_000

# Step 11 OG2 辅池：当前已下完（无 .aria2）的两个 train chunk + 子抽样比例
OG2_TRAIN_SPECS: tuple[tuple[str, float], ...] = (
    ("ncrna/data_ncrna_train_chunk4.jsonl.gz", 0.02),
    ("promoters/data_promoters_train_chunk1.jsonl.gz", 0.10),
)

FASTA_GLOBS = (
    "*.fasta",
    "*.fasta.gz",
    "*.fa",
    "*.fa.gz",
    "*.fna",
    "*.fna.gz",
)


@dataclass
class LabeledSeq:
    name: str
    seq: str
    source: str = ""


def seq_to_ids(seq: str) -> np.ndarray:
    """清洗后的 ASCII DNA → 字节 token id（CharLevelTokenizer 同口径）。"""
    if not seq:
        return np.empty(0, dtype=np.int64)
    return np.frombuffer(seq.encode("ascii"), dtype=np.uint8).astype(np.int64)


def load_hg38(path: str | Path = HG38_CSV) -> list[LabeledSeq]:
    import csv

    path = Path(path)
    out: list[LabeledSeq] = []
    with path.open(newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            seq = clean_seq(row["seq"])
            if not seq:
                continue
            ident = row.get("id") or f"hg38_{i}"
            chrom = row.get("chrom", "")
            start = row.get("start", "")
            end = row.get("end", "")
            out.append(
                LabeledSeq(
                    name=str(ident),
                    seq=seq,
                    source=f"{chrom}:{start}-{end}",
                )
            )
    return out


def load_dna_samples(path: str | Path = DNA_SAMPLES) -> list[LabeledSeq]:
    path = Path(path)
    with path.open() as fh:
        data = json.load(fh)
    out: list[LabeledSeq] = []
    for name, rec in data.items():
        seq = clean_seq(rec["seq"] if isinstance(rec, dict) else rec)
        if not seq:
            continue
        src = rec.get("source", "") if isinstance(rec, dict) else ""
        out.append(LabeledSeq(name=str(name), seq=seq, source=src))
    return out


def iter_fasta_directory(root: str | Path) -> Iterator[LabeledSeq]:
    """OpenGenome2 fasta 目录接口。跳过仍在下载的 ``*.aria2`` 伴随文件。"""
    root = Path(root)
    if not root.exists():
        return
    seen: set[Path] = set()
    for pattern in FASTA_GLOBS:
        for path in sorted(root.rglob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            if Path(str(path) + ".aria2").exists():
                continue
            try:
                seqs = load_sequences(path)
            except (OSError, ValueError):
                continue
            for i, seq in enumerate(seqs):
                yield LabeledSeq(name=f"{path.name}:{i}", seq=seq, source=str(path))


def split_corpus(
    records: list[LabeledSeq],
    test_frac: float = HOLD_OUT_FRAC,
) -> tuple[list[LabeledSeq], list[LabeledSeq]]:
    """每条序列连续切片：前缀训练、后缀 hold-out（禁止 shuffle）。"""
    train: list[LabeledSeq] = []
    hold: list[LabeledSeq] = []
    for rec in records:
        tr, te = split_contiguous(rec.seq, test_frac=test_frac)
        if tr:
            train.append(LabeledSeq(name=rec.name, seq=tr, source=rec.source))
        if te:
            hold.append(LabeledSeq(name=rec.name, seq=te, source=rec.source))
    return train, hold


def load_default_corpus(
    fasta_dir: str | Path | None = None,
) -> list[LabeledSeq]:
    records = load_hg38() + load_dna_samples()
    if fasta_dir is not None:
        records.extend(list(iter_fasta_directory(fasta_dir)))
    return records


def sample_window_ids(
    train_recs: list[LabeledSeq],
    window: int,
    rng: np.random.Generator,
    min_len: int | None = None,
) -> np.ndarray:
    """从训练前缀均匀抽一条，再均匀抽长度为 ``window`` 的切片。"""
    min_len = int(min_len or window)
    pool = [r for r in train_recs if len(r.seq) >= min_len]
    if not pool:
        raise RuntimeError(f"没有长度 ≥ {min_len} 的训练序列")
    rec = pool[int(rng.integers(0, len(pool)))]
    seq = rec.seq
    w = min(int(window), len(seq))
    if len(seq) == w:
        start = 0
    else:
        start = int(rng.integers(0, len(seq) - w + 1))
    return seq_to_ids(seq[start : start + w])


def sample_anchors(
    length: int,
    gamma: int,
    n_anchors: int,
    rng: np.random.Generator,
    min_ctx: int = 128,
) -> np.ndarray:
    """合法锚点 a 满足：a ≥ min_ctx-1 且 a+γ < length（有 γ 个待预测 token）。"""
    lo = max(int(min_ctx) - 1, 0)
    hi = int(length) - int(gamma) - 1
    if hi < 0:
        return np.empty(0, dtype=np.int64)
    if hi < lo:
        lo = 0
    pool = np.arange(lo, hi + 1, dtype=np.int64)
    if pool.size == 0:
        return pool
    n = min(int(n_anchors), int(pool.size))
    if n == pool.size:
        return pool.copy()
    pick = rng.choice(pool, size=n, replace=False)
    return np.sort(pick.astype(np.int64))


def holdout_eval_window(
    full_seq: str,
    test_frac: float = HOLD_OUT_FRAC,
    window: int = 2048,
) -> tuple[np.ndarray, int, int]:
    """取以序列末尾对齐、长度 ≤ window 的切片，返回 (ids, holdout_start_in_window, split_in_full)。

    hold-out 目标落在 ``ids[holdout_start:]``；前文来自训练前缀，无标签泄漏。
    """
    n = len(full_seq)
    split = int(n * (1.0 - test_frac))
    w = min(int(window), n)
    start = n - w
    ids = seq_to_ids(full_seq[start : start + w])
    holdout_in_window = max(0, split - start)
    return ids, holdout_in_window, split


# ---------------------------------------------------------------------------
# Step 11（plans/13）：多源混合语料
# ---------------------------------------------------------------------------


def records_to_chunks(
    seqs,
    chunk_bp: int = STITCH_CHUNK_BP,
    sep: str = STITCH_SEP,
) -> list[str]:
    """把短序列按给定顺序用 ``sep`` 拼接成 ≤ ``chunk_bp`` 的长 chunk。

    CDS / ncRNA / promoter 等短记录无法直接喂 2k 窗采样；拼接形态与 Evo2
    预训练 "stitched" 语料一致（OG2 README：mRNA/5kb windows stitched）。
    """
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for seq in seqs:
        if not seq:
            continue
        add = len(seq) if not cur else len(sep) + len(seq)
        if cur and cur_len + add > int(chunk_bp):
            chunks.append(sep.join(cur))
            cur = [seq]
            cur_len = len(seq)
        else:
            cur.append(seq)
            cur_len += add
    if cur:
        chunks.append(sep.join(cur))
    return chunks


def load_ncbi_chunks(
    subdir: str = "train",
    root: str | Path = NCBI_CODING_DIR,
) -> list[LabeledSeq]:
    """读取 ``data/ncbi_coding/{train,holdout}/*.jsonl``（``fetch_ncbi_coding.py`` 产物）。"""
    root = Path(root) / subdir
    out: list[LabeledSeq] = []
    if not root.exists():
        return out
    for path in sorted(root.glob("*.jsonl")):
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                seq = clean_seq(rec["seq"])
                if not seq:
                    continue
                out.append(
                    LabeledSeq(
                        name=str(rec.get("name") or f"{path.stem}:{len(out)}"),
                        seq=seq,
                        source=f"ncbi_coding/{subdir}/{rec.get('species', path.stem)}",
                    )
                )
    return out


def iter_og2_jsonl(path: str | Path) -> Iterator[str]:
    """流式读取 OG2 jsonl(.gz) 的 ``text`` 字段。带 .aria2 伴侣的一律拒绝。"""
    import gzip

    path = Path(path)
    if Path(str(path) + ".aria2").exists():
        raise RuntimeError(f"{path} 有 .aria2 伴侣（下载未完成），按约定视为不存在")
    if not path.exists():
        raise RuntimeError(f"{path} 不存在")
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:  # type: ignore[arg-type]
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = rec.get("text") if isinstance(rec, dict) else None
            if text:
                yield str(text)


def load_og2_pool(
    specs: tuple[tuple[str, float], ...] = OG2_TRAIN_SPECS,
    seed: int = 20260821,
    chunk_bp: int = STITCH_CHUNK_BP,
) -> list[LabeledSeq]:
    """OG2 完整 train chunk → 确定性子抽样 → 拼接 chunk 池。

    ``specs`` = (相对 ``OG2_JSON_DIR`` 路径, keep_prob)。只用无 .aria2 的
    完整文件（``iter_og2_jsonl`` 内部强校验）；子抽样种子按文件序确定。
    """
    pool: list[LabeledSeq] = []
    for i, (rel, keep_prob) in enumerate(specs):
        path = OG2_JSON_DIR / rel
        rng = np.random.default_rng(int(seed) + i)
        kept: list[str] = []
        n_seen = 0
        for text in iter_og2_jsonl(path):
            n_seen += 1
            if rng.random() >= keep_prob:
                continue
            seq = clean_seq(text)
            if seq:
                kept.append(seq)
        chunks = records_to_chunks(kept, chunk_bp=chunk_bp)
        tag = Path(rel).parent.name
        for j, ch in enumerate(chunks):
            pool.append(LabeledSeq(name=f"og2_{tag}:{j}", seq=ch, source=str(path)))
    return pool


def make_mixed_sampler(
    pools: dict[str, list[LabeledSeq]],
    weights: dict[str, float],
    counts: dict[str, int] | None = None,
):
    """多源加权窗口采样器：先按权重抽源，再源内均匀抽记录、均匀抽窗。

    返回与 ``sample_window_ids`` 同签名 ``(train_recs, window, rng)`` 的闭包
    （``train_recs`` 忽略，仅对齐 ``train_one_cell`` 注入点）；权重落盘由调用方负责。
    传入 ``counts`` 字典时按源累计实际抽样次数（核验用）。
    """
    names = [n for n in pools if pools[n]]
    if not names:
        raise RuntimeError("混合采样池为空")
    w = np.array([float(weights.get(n, 0.0)) for n in names], dtype=np.float64)
    if w.sum() <= 0:
        raise RuntimeError(f"混合采样权重和为 0: {weights}")
    w = w / w.sum()

    def sampler(_recs, window: int, rng: np.random.Generator) -> np.ndarray:
        src = names[int(rng.choice(len(names), p=w))]
        if counts is not None:
            counts[src] = counts.get(src, 0) + 1
        return sample_window_ids(pools[src], window, rng)

    return sampler
