"""Step 11：NCBI 多物种编码区语料抓取（plans/13 §1.1，立等可取数据源）。

模式沿用 ``scripts/fetch_dna_samples.py``（Plan 01 验证 eutils 直连可用）：
``efetch db=nuccore rettype=fasta_cds_na`` 按物种抓全基因组 CDS 集（5'→3' 编码链），
确定性子抽样到物种 bp 上限，再用 16×N 分隔拼接成 ~50kb chunk（与 Evo2 预训练
"stitched" 语料同形态），落盘 ``data/ncbi_coding/{train,holdout}/<key>.jsonl``。

纪律：
- 礼貌限速：默认 3 并发线程（eutils 无 key 限速 3 req/s；单请求传输分钟级，
  3 并发远低于请求频率上限），每请求间隔 ≥3.5s，失败指数退避重试 5 次；
  传输走 gzip 编码（实测本机被 OG2 下载占满带宽时纯文本会读超时）；
- holdout 物种（3 个，按物种整体留出）只进 ``holdout/``，训练侧绝不读取；
- E. coli K-12（NC_000913.3）刻意**不抓**：lacZ 前缀经 dna_samples 进入训练是
  Step 8/9 既有协议，再把全基因组 CDS 加进来会让主判据 lacZ τ 的解读变浑；
  其近缘（Shigella/Klebsiella/Yersinia 等）覆盖同一域统计。

用法::

    source ~/miniconda3/etc/profile.d/conda.sh && conda activate evo2
    python -m evspark.train.fetch_ncbi_coding          # 全量抓取
    python -m evspark.train.fetch_ncbi_coding --only mycoplasma_genitalium  # 单物种试通
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import zlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np

from evspark.specdec.genome import clean_seq
from evspark.train.data import records_to_chunks

OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "ncbi_coding"
SLEEP_S = 3.5           # eutils 无 API key 限速 3 req/s；取更保守值
RETRY = 5
CHUNK_BP = 50_000
CAP_BP = 1_600_000      # 大基因组 CDS 保留上限（子抽样后）
SEED = 20260821

# (key, [accessions], 说明, cap_bp)；accessions 多条时顺序抓取合并（同物种多染色体）
TRAIN_SPECIES: list[tuple[str, list[str], str, int]] = [
    # ---- 肠杆菌目（E. coli 近缘）----
    ("shigella_flexneri", ["NC_004337.2"], "Shigella flexneri 2a 301（E. coli 近缘）", CAP_BP),
    ("klebsiella_pneumoniae", ["NC_009648.1"], "Klebsiella pneumoniae MGH 78578", CAP_BP),
    ("yersinia_pestis", ["NC_003143.1"], "Yersinia pestis CO92", CAP_BP),
    # ---- 其他革兰氏阴性 ----
    ("pseudomonas_aeruginosa", ["NC_002516.2"], "Pseudomonas aeruginosa PAO1", CAP_BP),
    ("vibrio_cholerae", ["NC_002505.1", "NC_002506.1"], "Vibrio cholerae N16961 chr I+II", CAP_BP),
    ("neisseria_meningitidis", ["NC_003112.2"], "Neisseria meningitidis MC58", CAP_BP),
    ("haemophilus_influenzae", ["NC_000907.1"], "Haemophilus influenzae Rd KW20", CAP_BP),
    ("helicobacter_pylori", ["NC_000915.1"], "Helicobacter pylori 26695", CAP_BP),
    ("campylobacter_jejuni", ["NC_002163.1"], "Campylobacter jejuni NCTC 11168", CAP_BP),
    ("caulobacter_crescentus", ["NC_002696.2"], "Caulobacter crescentus CB15", CAP_BP),
    ("rickettsia_prowazekii", ["NC_000963.1"], "Rickettsia prowazekii Madrid E", CAP_BP),
    ("chlamydia_trachomatis", ["NC_000117.1"], "Chlamydia trachomatis D/UW-3/CX", CAP_BP),
    # ---- 革兰氏阳性 ----
    ("staphylococcus_aureus", ["NC_002745.2"], "Staphylococcus aureus N315", CAP_BP),
    ("streptococcus_pneumoniae", ["NC_003098.1"], "Streptococcus pneumoniae R6", CAP_BP),
    ("listeria_monocytogenes", ["NC_003210.1"], "Listeria monocytogenes EGD-e", CAP_BP),
    ("clostridioides_difficile", ["NC_009089.1"], "Clostridioides difficile 630", CAP_BP),
    ("mycobacterium_tuberculosis", ["NC_000962.3"], "Mycobacterium tuberculosis H37Rv", CAP_BP),
    ("streptomyces_coelicolor", ["NC_003888.3"], "Streptomyces coelicolor A3(2)", CAP_BP),
    ("lactococcus_lactis", ["NC_002662.1"], "Lactococcus lactis IL1403", CAP_BP),
    ("mycoplasma_genitalium", ["NC_000908.2"], "Mycoplasma genitalium G37（最小基因组）", CAP_BP),
    # ---- 其他细菌门 ----
    ("synechocystis_pcc6803", ["NC_000911.1"], "Synechocystis sp. PCC 6803（蓝藻）", CAP_BP),
    ("thermus_thermophilus", ["NC_005835.1"], "Thermus thermophilus HB27", CAP_BP),
    ("deinococcus_radiodurans", ["NC_001263.1"], "Deinococcus radiodurans R1 chr1", CAP_BP),
    ("aquifex_aeolicus", ["NC_000918.1"], "Aquifex aeolicus VF5", CAP_BP),
    ("thermotoga_maritima", ["NC_000853.1"], "Thermotoga maritima MSB8", CAP_BP),
    # ---- 古菌（少量）----
    ("methanocaldococcus_jannaschii", ["NC_000909.1"], "Methanocaldococcus jannaschii", CAP_BP),
    ("sulfolobus_solfataricus", ["NC_002754.1"], "Sulfolobus solfataricus P2", CAP_BP),
    ("pyrococcus_furiosus", ["NC_003413.1"], "Pyrococcus furiosus DSM 3638", CAP_BP),
    # ---- 真核（少量：酵母两条染色体）----
    ("saccharomyces_cerevisiae", ["NC_001136.10", "NC_001144.5"],
     "S. cerevisiae S288c chr IV + chr XII 编码区", CAP_BP),
    # ---- 病毒/噬菌体（少量，全保留）----
    ("phage_lambda", ["NC_001416.1"], "Bacteriophage lambda", 0),
    ("phage_t4", ["NC_000866.4"], "Bacteriophage T4", 0),
    ("phage_t7", ["NC_001604.1"], "Bacteriophage T7", 0),
    ("phix174", ["NC_001422.1"], "phiX174", 0),
    ("hiv1", ["NC_001802.1"], "HIV-1 HXB2", 0),
    ("sars_cov2", ["NC_045512.2"], "SARS-CoV-2 Wuhan-Hu-1", 0),
]

# 按物种整体留出（泛化检验，plans/13 §2）：近缘 / 革兰氏阳性 / 远缘各一
HOLDOUT_SPECIES: list[tuple[str, list[str], str, int]] = [
    ("salmonella_enterica", ["NC_004631.1"], "Salmonella enterica Typhi Ty2（E. coli 近缘留出）", CAP_BP),
    ("bacillus_subtilis", ["NC_000964.3"], "Bacillus subtilis 168（革兰氏阳性留出）", CAP_BP),
    ("bacteroides_thetaiotaomicron", ["NC_004663.1"], "Bacteroides thetaiotaomicron VPI-5482（远缘留出）", CAP_BP),
]


def log(msg: str) -> None:
    print(f"[fetch_ncbi] {msg}", flush=True)


def efetch_cds(acc: str) -> str:
    """整基因组 CDS 集（fasta_cds_na）。版本号失效时去掉版本号重试一次。

    必须 gzip 编码：本机下行带宽被 OG2 下载占满，纯文本单基因组 ~5MB 会
    超时（实测 2.3KB/s）；gzip 后 ~3 倍压缩（实测 12KB/s 线速，28s/MBp 区间）。
    """
    import gzip

    def _url(a: str) -> str:
        return (
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore"
            f"&id={a}&rettype=fasta_cds_na&retmode=text"
        )

    last_err: Exception | None = None
    for attempt in range(RETRY):
        acc_try = acc if attempt == 0 else (acc.split(".")[0] if attempt == 1 else acc)
        try:
            req = urllib.request.Request(
                _url(acc_try), headers={"Accept-Encoding": "gzip"}
            )
            with urllib.request.urlopen(req, timeout=600) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip" or raw[:2] == b"\x1f\x8b":
                    raw = gzip.decompress(raw)
                text = raw.decode()
            if ">" in text:
                return text
            raise RuntimeError(f"响应无 FASTA 记录（{acc_try}）：{text[:200]!r}")
        except Exception as e:  # noqa: BLE001 - 抓取脚本要兜底所有网络异常
            last_err = e
            wait = 5.0 * (attempt + 1)
            log(f"  {acc} 第 {attempt + 1} 次失败：{e}；{wait:.0f}s 后重试")
            time.sleep(wait)
    raise RuntimeError(f"fetch failed: {acc}: {last_err}")


def parse_fasta(text: str) -> list[str]:
    seqs: list[str] = []
    cur: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur))
                cur = []
        else:
            cur.append(line.strip())
    if cur:
        seqs.append("".join(cur))
    return seqs


def subsample_bp(seqs: list[str], cap_bp: int, rng: np.random.Generator) -> list[str]:
    """超过 cap 时随机子抽样 CDS（保持基因组相对顺序），否则全保留。"""
    if cap_bp <= 0:
        return seqs
    total = sum(len(s) for s in seqs)
    if total <= cap_bp:
        return seqs
    order = rng.permutation(len(seqs))
    keep: list[int] = []
    acc = 0
    for i in order:
        keep.append(int(i))
        acc += len(seqs[int(i)])
        if acc >= cap_bp:
            break
    return [seqs[i] for i in sorted(keep)]


def fetch_one(
    entry: tuple[str, list[str], str, int],
    group: str,
) -> dict:
    """单物种全流程（抓取→清洗→子抽样→拼接→落盘），返回 manifest 条目。

    幂等：目标文件已存在则跳过抓取、按文件内容回填条目。多线程调用安全
    （物种间文件不相交；manifest 条目由主线程按序组装）。
    """
    key, accs, note, cap = entry
    out_dir = OUT_DIR / group
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{key}.jsonl"
    if out_path.exists():
        n_chunks = 0
        bp = 0
        with out_path.open() as fh:
            for line in fh:
                if line.strip():
                    n_chunks += 1
                    bp += len(json.loads(line)["seq"])
        log(f"跳过已存在 {out_path.name}（幂等；重抓请先删）：{n_chunks} chunks / {bp:,}bp")
        return {
            "key": key,
            "group": group,
            "accessions": accs,
            "note": note,
            "n_chunks": n_chunks,
            "bp_kept": bp,
            "file": str(out_path),
            "reused_existing": True,
        }
    log(f"[{group}] {key}（{note}）accessions={accs}")
    raw_seqs: list[str] = []
    for acc in accs:
        t0 = time.time()
        text = efetch_cds(acc)
        got = parse_fasta(text)
        log(f"  {acc}: {len(got)} 条 CDS，{time.time() - t0:.1f}s")
        raw_seqs.extend(got)
        time.sleep(SLEEP_S)
    cleaned = [clean_seq(s) for s in raw_seqs]
    cleaned = [s for s in cleaned if len(s) >= 99]  # 去掉过短/空记录
    n_total = len(cleaned)
    bp_total = sum(len(s) for s in cleaned)
    acgtn = sum(sum(c in "ACGTN" for c in s) for s in cleaned)
    rng = np.random.default_rng(SEED + zlib.crc32(key.encode()) % 100_000)
    kept = subsample_bp(cleaned, cap, rng)
    bp_kept = sum(len(s) for s in kept)
    chunks = records_to_chunks(kept, chunk_bp=CHUNK_BP)
    tmp_path = out_path.with_suffix(".jsonl.tmp")
    with tmp_path.open("w") as fh:
        for i, ch in enumerate(chunks):
            fh.write(json.dumps({"name": f"{key}:{i}", "species": key, "seq": ch}) + "\n")
    tmp_path.replace(out_path)
    log(
        f"  → {key}: {n_total} 条 CDS / {bp_total:,}bp，保留 {bp_kept:,}bp，"
        f"{len(chunks)} chunks，ACGTN={acgtn / max(bp_total, 1):.4f}"
    )
    return {
        "key": key,
        "group": group,
        "accessions": accs,
        "note": note,
        "n_cds_total": n_total,
        "bp_total": bp_total,
        "bp_kept": bp_kept,
        "cap_bp": cap,
        "n_chunks": len(chunks),
        "acgtn_frac": round(acgtn / max(bp_total, 1), 6),
        "file": str(out_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Step 11 NCBI 多物种编码区抓取")
    ap.add_argument("--only", type=str, default=None, help="逗号分隔物种 key 子集（试通用）")
    ap.add_argument("--workers", type=int, default=3,
                    help="并发抓取线程数（默认 3；eutils 无 key 限速 3 req/s，"
                         "单请求传输分钟级，3 并发仍远低于限速）")
    args = ap.parse_args()
    only = {s.strip() for s in args.only.split(",") if s.strip()} if args.only else None

    manifest: dict = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "NCBI eutils efetch db=nuccore rettype=fasta_cds_na（整基因组 CDS 集，5'→3' 编码链）",
        "params": {
            "sleep_s": SLEEP_S,
            "retry": RETRY,
            "workers": args.workers,
            "chunk_bp": CHUNK_BP,
            "cap_bp": CAP_BP,
            "seed": SEED,
            "separator": "N" * 16,
            "ecoli_k12_excluded": "NC_000913.3 刻意不抓（lacZ 主判据口径保持干净）",
        },
        "species": [],
        "failures": [],
    }
    t0 = time.time()
    jobs: list[tuple[int, str, tuple[str, list[str], str, int]]] = []
    for group, entries in (("train", TRAIN_SPECIES), ("holdout", HOLDOUT_SPECIES)):
        for entry in entries:
            if only and entry[0] not in only:
                continue
            jobs.append((len(jobs), group, entry))

    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futs = {pool.submit(fetch_one, entry, group): ix for ix, group, entry in jobs}
        for fut in as_completed(futs):
            ix = futs[fut]
            group, entry = jobs[ix][1], jobs[ix][2]
            try:
                results[ix] = fut.result()
            except Exception as e:  # noqa: BLE001
                log(f"!! {entry[0]} 失败：{e}")
                manifest["failures"].append(
                    {"key": entry[0], "group": group, "error": str(e)}
                )
    manifest["species"] = [results[ix] for ix, _, _ in jobs if ix in results]
    manifest["elapsed_s"] = round(time.time() - t0, 1)
    n_train = sum(1 for s in manifest["species"] if s["group"] == "train")
    n_train_bacteria = sum(
        1
        for s in manifest["species"]
        if s["group"] == "train"
        and s["key"]
        not in {
            "methanocaldococcus_jannaschii", "sulfolobus_solfataricus", "pyrococcus_furiosus",
            "saccharomyces_cerevisiae", "phage_lambda", "phage_t4", "phage_t7",
            "phix174", "hiv1", "sars_cov2",
        }
    )
    manifest["summary"] = {
        "n_species_train": n_train,
        "n_species_train_bacteria": n_train_bacteria,
        "n_species_holdout": sum(1 for s in manifest["species"] if s["group"] == "holdout"),
        "bp_train": sum(s["bp_kept"] for s in manifest["species"] if s["group"] == "train"),
        "bp_holdout": sum(s["bp_kept"] for s in manifest["species"] if s["group"] == "holdout"),
        "n_failures": len(manifest["failures"]),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    log(f"manifest 落盘 {OUT_DIR / 'manifest.json'}：{json.dumps(manifest['summary'], ensure_ascii=False)}")
    if only:
        return
    if n_train_bacteria < 20:
        raise RuntimeError(f"训练集细菌物种数 {n_train_bacteria} < 20，未达 plans/13 要求")
    if manifest["failures"]:
        raise RuntimeError(f"有抓取失败：{manifest['failures']}")


if __name__ == "__main__":
    main()
