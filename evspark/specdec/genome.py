"""参考基因组读写与清洗。

清洗规则（plans/04 §2）：大写化；空白（空格/制表/换行）丢弃；
非 ACGT 字符（含 N 与其它 IUPAC 简并码）统一映射为 N。
FASTA 跳过 header；CSV 取序列列。
"""

from __future__ import annotations

import csv
import gzip
from collections.abc import Iterator
from pathlib import Path

ALPHABET = "ACGTN"
N_SYMBOLS = len(ALPHABET)
CHAR_TO_ID: dict[str, int] = {c: i for i, c in enumerate(ALPHABET)}
ID_TO_CHAR: dict[int, str] = {i: c for c, i in CHAR_TO_ID.items()}
N_ID = CHAR_TO_ID["N"]

_SEQ_COL_CANDIDATES = ("seq", "sequence", "dna", "SEQ", "Sequence", "DNA")


def _clean_translate_table() -> tuple[bytes, bytes]:
    table = bytearray([ord("N")] * 256)
    for src, dst in zip(b"ACGTacgtNn", b"ACGTACGTNN"):
        table[src] = dst
    return bytes(table), b" \t\r\n"


_CLEAN_TABLE, _CLEAN_DELETE = _clean_translate_table()


def clean_seq(seq: str) -> str:
    """大写化，丢弃空白，其余非 ACGT 映射为 N。"""
    if not seq:
        return ""
    raw = seq.encode("ascii", errors="replace")
    return raw.translate(_CLEAN_TABLE, _CLEAN_DELETE).decode("ascii")


def _open_text(path: Path):
    if path.suffix == ".gz" or path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return open(path, "rt", encoding="utf-8", newline="")


def _logical_name(path: Path) -> str:
    name = path.name
    if name.endswith(".gz"):
        name = name[:-3]
    return name


def _is_csv_name(name: str) -> bool:
    return name.endswith(".csv")


def iter_raw_sequences(path: str | Path) -> Iterator[str]:
    """逐条产出未清洗的序列字符串（FASTA 按 record；CSV 按序列列）。"""
    path = Path(path)
    name = _logical_name(path)
    with _open_text(path) as fh:
        if _is_csv_name(name):
            yield from _iter_csv_raw(fh)
        else:
            yield from _iter_fasta_raw(fh)


def _iter_fasta_raw(fh) -> Iterator[str]:
    chunks: list[str] = []
    saw_header = False
    for line in fh:
        if line.startswith(">"):
            if chunks:
                yield "".join(chunks)
                chunks = []
            saw_header = True
            continue
        chunks.append(line.rstrip("\r\n"))
    if chunks:
        yield "".join(chunks)
    elif not saw_header:
        return


def _iter_csv_raw(fh) -> Iterator[str]:
    reader = csv.DictReader(fh)
    if reader.fieldnames is None:
        return
    fields = [f.strip() if isinstance(f, str) else f for f in reader.fieldnames]
    reader.fieldnames = fields
    col = None
    lower = {f.lower(): f for f in fields if f}
    for cand in _SEQ_COL_CANDIDATES:
        if cand in reader.fieldnames:
            col = cand
            break
        if cand.lower() in lower:
            col = lower[cand.lower()]
            break
    if col is None:
        # 只有一列则用之；否则取最长字段名含 seq 的列
        if len(fields) == 1:
            col = fields[0]
        else:
            raise ValueError(f"CSV 找不到序列列，表头={fields!r}")
    for row in reader:
        yield row.get(col) or ""


def load_sequences(path: str | Path) -> list[str]:
    """读取 FASTA/CSV（可 .gz），返回清洗后的非空序列列表。"""
    out: list[str] = []
    for raw in iter_raw_sequences(path):
        cleaned = clean_seq(raw)
        if cleaned:
            out.append(cleaned)
    return out


def cleaning_stats(raw_seqs: list[str]) -> dict[str, float | int]:
    """统计清洗丢弃/映射比例，供报告使用。"""
    n_raw_chars = 0
    n_whitespace = 0
    n_acgt = 0
    n_n_orig = 0
    n_other = 0
    acgt = set("ACGTacgt")
    nn = set("Nn")
    ws = set(" \t\r\n")
    for raw in raw_seqs:
        n_raw_chars += len(raw)
        for ch in raw:
            if ch in ws:
                n_whitespace += 1
            elif ch in acgt:
                n_acgt += 1
            elif ch in nn:
                n_n_orig += 1
            else:
                n_other += 1
    n_kept = n_acgt + n_n_orig + n_other  # 空白丢弃后的长度
    return {
        "n_records_raw": len(raw_seqs),
        "n_raw_chars": n_raw_chars,
        "n_whitespace_stripped": n_whitespace,
        "n_acgt": n_acgt,
        "n_n_original": n_n_orig,
        "n_other_mapped_to_n": n_other,
        "n_bases_after_ws": n_kept,
        "frac_ws_stripped": (n_whitespace / n_raw_chars) if n_raw_chars else 0.0,
        "frac_mapped_to_n": ((n_n_orig + n_other) / n_kept) if n_kept else 0.0,
        "frac_non_acgt_in_raw": ((n_whitespace + n_n_orig + n_other) / n_raw_chars)
        if n_raw_chars
        else 0.0,
    }
