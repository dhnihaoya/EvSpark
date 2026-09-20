"""DNA 表型度量（Phase 4）：GC、6 帧完整 ORF、k-mer。无 torch / 无外部二进制。

ORF 规则（预注册）：标准遗传密码；起始密码子仅 ATG；终止 TAA/TAG/TGA；
六帧（+ 与 revcomp）；完整 ORF 最短 ``MIN_ORF_BP``（默认 60）。
k-mer 只统计不含 N 的窗口。
"""

from __future__ import annotations

from collections import Counter

from evspark.specdec.genome import clean_seq

ACGT = "ACGT"
STOPS = frozenset({"TAA", "TAG", "TGA"})
START = "ATG"
_REVCOMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")

# CharLevelTokenizer：A/C/G/T/N = 65/67/71/84/78
_ID_TO_BASE = {65: "A", 67: "C", 71: "G", 84: "T", 78: "N"}


def ids_to_dna(ids) -> str:
    """token id → ACGTN；词表其余位置（含 eod/pad）映射为 N。"""
    chars = []
    for x in ids:
        chars.append(_ID_TO_BASE.get(int(x), "N"))
    return "".join(chars)


def revcomp(seq: str) -> str:
    return seq.translate(_REVCOMP)[::-1]


def gc_fraction(seq: str) -> float:
    """GC / (A+C+G+T)。无 ACGT 时返回 NaN（调用方丢弃）。"""
    s = clean_seq(seq)
    n_gc = s.count("G") + s.count("C")
    n_atgc = n_gc + s.count("A") + s.count("T")
    if n_atgc == 0:
        return float("nan")
    return n_gc / n_atgc


def complete_orfs(seq: str, *, min_bp: int = 60) -> list[dict]:
    """六帧 ATG–stop 完整 ORF，长度 ≥ min_bp。

    每个起始密码子独立扫到同框第一个终止；允许嵌套。坐标相对输入链
    （负链 ORF 的 start/end 是 revcomp 上的坐标，带 ``strand=-1``）。
    """
    s = clean_seq(seq)
    out: list[dict] = []
    for strand_seq, sign in ((s, 1), (revcomp(s), -1)):
        n = len(strand_seq)
        for frame in range(3):
            i = frame
            while i + 2 < n:
                if strand_seq[i : i + 3] == START:
                    j = i + 3
                    while j + 2 < n:
                        codon = strand_seq[j : j + 3]
                        if codon in STOPS:
                            length = (j + 3) - i
                            if length >= int(min_bp):
                                out.append(
                                    {
                                        "start": i,
                                        "end": j + 3,
                                        "length": length,
                                        "strand": sign,
                                        "frame": frame,
                                    }
                                )
                            break
                        j += 3
                i += 3
    return out


def orf_metrics(seq: str, *, min_bp: int = 60) -> dict:
    orfs = complete_orfs(seq, min_bp=min_bp)
    n = len(clean_seq(seq))
    lengths = [o["length"] for o in orfs]
    longest = max(lengths) if lengths else 0
    coding = sum(lengths)
    # 允许重叠，覆盖率可 >1；夹到用于描述的 [0, ∞)
    frac = (coding / n) if n else 0.0
    return {
        "n_orfs": len(orfs),
        "longest_orf_bp": int(longest),
        "coding_bp": int(coding),
        "coding_frac": float(frac),
        "orf_lengths": lengths,
    }


def kmer_counts(seq: str, k: int) -> Counter:
    if k < 1:
        raise ValueError("k 须 ≥ 1")
    s = clean_seq(seq)
    c: Counter = Counter()
    for i in range(0, len(s) - k + 1):
        mer = s[i : i + k]
        if all(ch in ACGT for ch in mer):
            c[mer] += 1
    return c


def kmer_tvd(counts_a: Counter, counts_b: Counter) -> float:
    """全变差 ½∑|p−q|。两边都空 → 0。"""
    keys = set(counts_a) | set(counts_b)
    na = float(sum(counts_a.values()))
    nb = float(sum(counts_b.values()))
    if na <= 0.0 and nb <= 0.0:
        return 0.0
    if na <= 0.0 or nb <= 0.0:
        return 1.0
    acc = 0.0
    for k in keys:
        acc += abs(counts_a[k] / na - counts_b[k] / nb)
    return 0.5 * acc


def sequence_features(dna: str, *, min_orf_bp: int = 60) -> dict:
    """一条续写的全部 CPU 表型（不含 NLL / 外部工具）。"""
    dna = clean_seq(dna)
    om = orf_metrics(dna, min_bp=min_orf_bp)
    return {
        "length": len(dna),
        "gc": gc_fraction(dna),
        "n_orfs": om["n_orfs"],
        "longest_orf_bp": om["longest_orf_bp"],
        "coding_frac": om["coding_frac"],
        "dinuc": dict(kmer_counts(dna, 2)),
        "trinuc": dict(kmer_counts(dna, 3)),
        "tetranuc": dict(kmer_counts(dna, 4)),
    }
