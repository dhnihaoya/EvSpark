"""C.1 正式期 teacher-only 压缩落盘（plans/16 / plans/02 C.1）。

无状态 4k 窗前向（Plan 01 全并行预填上限内的最大整齐窗口），写出：

- ``ids`` uint8、``top8_idx`` uint16、``top8_prob`` fp16、``resid_mass`` fp16
- ``hidden`` int8 [K, N, 4096] + ``hidden_scale`` fp16 [K, N]（逐 token absmax；
  ``--scale-fp32`` 时 scale 存 fp32——末段 blocks.30/31 残差 ~1e11 溢出 fp16 用）

分片 ~50 万位置 / shard（``np.savez`` 无压缩）+ sidecar json；断点续传跳过已完整 shard。
采样配比 = Step 14（七源 ×0.9 + euk 0.10）。禁止动用户 aria2。

用法::

    CUDA_VISIBLE_DEVICES=1 HF_HOME=./hf_home \\
      python -u scripts/dump_c1_dataset.py --smoke

    CUDA_VISIBLE_DEVICES=0 python -u scripts/dump_c1_dataset.py \\
      --scheme L27 --sources gtdb,mrna,hg38,ncbi --total-positions 300000000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / "hf_home"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import numpy as np

from specdec.genome import clean_seq
from train.c1_pack import (
    HIDDEN_DIM,
    TOPK,
    arrays_crc,
    dequantize_hidden,
    entropy_bits,
    entropy_hist,
    finalize_hist,
    merge_hists,
    pack_topk,
    quantize_hidden,
    reconstruct_pt,
)
from train.data import (
    STITCH_SEP,
    load_default_corpus,
    load_dna_samples,
    load_ncbi_chunks,
    seq_to_ids,
    split_corpus,
)
from train.drafter import SCHEME_LAYERS
from train.step12_data import (
    GZ_RAW_RATIO,
    complete_train_chunks,
    iter_og2_jsonl_records,
)

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "dump_c1"
WINDOW = 4096
SHARD_POS = 500_000
TOTAL_POS = 300_000_000

# 与 step12_retrain.MIX_WEIGHTS + --with-euk 同口径（不 import 以免 setdefault CUDA=2）
MIX_SEVEN = {
    "ncbi": 0.15,
    "gtdb": 0.20,
    "mrna": 0.35,
    "ncrna": 0.08,
    "organelle": 0.04,
    "promoters": 0.03,
    "hg38": 0.15,
}
EUK_WEIGHT = 0.10
IMGVR_WEIGHT = 0.10  # Step 15：与 euk 同法接入（plans/17 §1.2 初值）
SOURCE_SUB = {
    "gtdb": "gtdb_v220_imgpr",
    "mrna": "mrna_splice_promoter",
    "ncrna": "ncrna",
    "organelle": "organelle",
    "promoters": "promoters",
    "euk": "eukaryotic_genic_windows",
    "imgvr": "imgvr_untagged",
}

# Plan 01 §2.3 锚点（bench_entropy.py / baseline.md）
PLAN01_ENTROPY = {
    "coding_ecoli_lacZ": {"entropy_mean": 0.394, "frac_lt_1_5bit": 0.926},
    "intergenic_human_chr21": {"entropy_mean": 1.598, "frac_lt_1_5bit": 0.285},
}


def log(msg: str) -> None:
    print(f"[c1dump] {msg}", flush=True)


def step14_weights(with_euk: bool = True, with_imgvr: bool = False) -> dict[str, float]:
    """dump 配额权重。Step 15 起 imgvr 与 euk 同法（各 0.10、七源等比让出）。

    注意：这是**配额公式**；训练侧终版配比以 ``train.step15_mix.FINAL_MIX``
    为唯一权威（两表在 imgvr 配额上同为 0.10×total，其余源差异见该模块注释）。
    """
    w = dict(MIX_SEVEN)
    keep = 1.0
    if with_euk:
        keep -= EUK_WEIGHT
    if with_imgvr:
        keep -= IMGVR_WEIGHT
    w = {k: v * keep for k, v in w.items()}
    if with_euk:
        w["euk"] = EUK_WEIGHT
    if with_imgvr:
        w["imgvr"] = IMGVR_WEIGHT
    tot = sum(w.values())
    return {k: v / tot for k, v in w.items()}


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
    tmp.replace(path)


def resolve_layers(scheme: str | None, layers_csv: str | None) -> tuple[str, ...]:
    if layers_csv:
        return tuple(x.strip() for x in layers_csv.split(",") if x.strip())
    sch = scheme or "L27"
    if sch not in SCHEME_LAYERS:
        raise RuntimeError(f"未知 scheme={sch}，可选 {sorted(SCHEME_LAYERS)}")
    layers = SCHEME_LAYERS[sch]
    if not layers:
        raise RuntimeError(f"scheme={sch} 无注入层，不能落盘 hidden")
    return layers


def iter_stitched_windows(seqs, window: int, sep: str = STITCH_SEP):
    buf = ""
    for seq in seqs:
        if not seq:
            continue
        buf = seq if not buf else buf + sep + seq
        while len(buf) >= window:
            yield buf[:window]
            buf = buf[window:]


def iter_og2_seqs(sub: str, keep_prob: float, seed: int, files=None):
    files = list(files) if files is not None else complete_train_chunks(sub)
    rng = np.random.default_rng(int(seed))
    for path in files:
        for _record, text in iter_og2_jsonl_records(path):
            if rng.random() >= keep_prob:
                continue
            seq = clean_seq(text)
            if seq:
                yield seq


def source_files_and_est(name: str, files_filter: set[str] | None = None) -> tuple[list, int]:
    if name == "hg38":
        recs, _ = split_corpus(load_default_corpus())
        bp = sum(len(r.seq) for r in recs)
        return recs, bp
    if name == "ncbi":
        recs = load_ncbi_chunks("train")
        bp = sum(len(r.seq) for r in recs)
        return recs, bp
    sub = SOURCE_SUB[name]
    files = complete_train_chunks(sub)
    if files_filter is not None:
        files = [p for p in files if p.name in files_filter]
        if not files:
            raise RuntimeError(f"{sub} 里没有匹配 --source-files 的 chunk: {sorted(files_filter)}")
    est = int(sum(p.stat().st_size for p in files) * GZ_RAW_RATIO)
    return files, est


def iter_source_windows(name: str, keep_prob: float, seed: int, window: int, files_filter=None):
    if name in ("hg38", "ncbi"):
        recs, _ = source_files_and_est(name)
        yield from iter_stitched_windows((r.seq for r in recs), window)
        return
    sub = SOURCE_SUB[name]
    files = None
    if files_filter is not None:
        files = [p for p in complete_train_chunks(sub) if p.name in files_filter]
    yield from iter_stitched_windows(iter_og2_seqs(sub, keep_prob, seed, files=files), window)


def existing_progress(out_dir: Path, source: str, shard_tag: str = "") -> tuple[int, int]:
    n_pos = 0
    next_idx = 0
    for p in sorted(out_dir.glob(f"{source}{shard_tag}_*.json")):
        try:
            meta = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not meta.get("complete"):
            continue
        n_pos += int(meta.get("n_pos") or 0)
        next_idx = max(next_idx, int(meta.get("shard_idx", -1)) + 1)
    return n_pos, next_idx


class ShardWriter:
    def __init__(self, out_dir: Path, source: str, layers: tuple[str, ...], shard_pos: int, start_idx: int,
                 shard_tag: str = ""):
        self.out_dir = out_dir
        self.source = source  # meta["source"]（训练器配比映射用，不带 tag）
        self.name_prefix = f"{source}{shard_tag}"  # 分片文件名（双卡分流互不覆盖）
        self.layers = layers
        self.shard_pos = int(shard_pos)
        self.idx = int(start_idx)
        self._reset()

    def _reset(self) -> None:
        self.ids: list[np.ndarray] = []
        self.tidx: list[np.ndarray] = []
        self.tval: list[np.ndarray] = []
        self.resid: list[np.ndarray] = []
        self.h_q: list[np.ndarray] = []
        self.h_s: list[np.ndarray] = []
        self.n = 0

    def add(self, ids, tidx, tval, resid, hidden_q, hidden_scale) -> list[dict]:
        self.ids.append(ids)
        self.tidx.append(tidx)
        self.tval.append(tval)
        self.resid.append(resid)
        self.h_q.append(hidden_q)
        self.h_s.append(hidden_scale)
        self.n += int(ids.shape[0])
        flushed = []
        while self.n >= self.shard_pos:
            flushed.append(self.flush(take=self.shard_pos))
        return flushed

    def flush(self, take: int | None = None) -> dict:
        if self.n <= 0:
            return {}
        take = int(take or self.n)
        ids = np.concatenate(self.ids, axis=0)
        tidx = np.concatenate(self.tidx, axis=0)
        tval = np.concatenate(self.tval, axis=0)
        resid = np.concatenate(self.resid, axis=0)
        hq = np.concatenate(self.h_q, axis=1)  # [K, N, D]
        hs = np.concatenate(self.h_s, axis=1)  # [K, N]
        head, rest = take, ids.shape[0] - take
        shard = {
            "ids": ids[:head],
            "top8_idx": tidx[:head],
            "top8_prob": tval[:head],
            "resid_mass": resid[:head],
            "hidden": hq[:, :head],
            "hidden_scale": hs[:, :head],
        }
        if rest > 0:
            self.ids = [ids[head:]]
            self.tidx = [tidx[head:]]
            self.tval = [tval[head:]]
            self.resid = [resid[head:]]
            self.h_q = [hq[:, head:]]
            self.h_s = [hs[:, head:]]
            self.n = rest
        else:
            self._reset()
        return self._write(shard)

    def _write(self, shard: dict) -> dict:
        name = f"{self.name_prefix}_{self.idx:05d}"
        npz = self.out_dir / f"{name}.npz"
        meta_path = self.out_dir / f"{name}.json"
        crc = arrays_crc(
            shard["ids"], shard["top8_idx"], shard["top8_prob"],
            shard["resid_mass"], shard["hidden"], shard["hidden_scale"],
        )
        np.savez(
            npz,
            ids=shard["ids"],
            top8_idx=shard["top8_idx"],
            top8_prob=shard["top8_prob"],
            resid_mass=shard["resid_mass"],
            hidden=shard["hidden"],
            hidden_scale=shard["hidden_scale"],
        )
        meta = {
            "complete": True,
            "source": self.source,
            "shard_idx": self.idx,
            "n_pos": int(shard["ids"].shape[0]),
            "layers": list(self.layers),
            "shapes": {k: list(v.shape) for k, v in shard.items()},
            "dtypes": {k: str(v.dtype) for k, v in shard.items()},
            "crc32": crc,
            "npz": npz.name,
            "bytes": npz.stat().st_size,
        }
        save_json(meta_path, meta)
        log(f"shard {name} n={meta['n_pos']} crc={crc} {meta['bytes'] / 1e9:.2f}GB")
        self.idx += 1
        return meta

    def close(self) -> dict | None:
        if self.n <= 0:
            return None
        return self.flush()


def verify_shard(meta_path: Path) -> None:
    meta = json.loads(meta_path.read_text())
    npz = meta_path.with_suffix(".npz")
    data = np.load(npz)
    need = ("ids", "top8_idx", "top8_prob", "resid_mass", "hidden", "hidden_scale")
    for k in need:
        if k not in data:
            raise RuntimeError(f"{npz} 缺字段 {k}")
    crc = arrays_crc(*(data[k] for k in need))
    if crc != meta.get("crc32"):
        raise RuntimeError(f"{npz} CRC 不符：盘面 {meta.get('crc32')} 重算 {crc}")
    n = int(data["ids"].shape[0])
    if data["hidden"].ndim != 3 or data["hidden"].shape[1] != n:
        raise RuntimeError(f"{npz} hidden 形状异常 {data['hidden'].shape}")
    # 随机 8 个位置：反量化有限、top-8 质量和 + resid ≈ 1
    rng = np.random.default_rng(0)
    pick = rng.choice(n, size=min(8, n), replace=False)
    mass = data["top8_prob"][pick].astype(np.float32).sum(-1) + data["resid_mass"][pick].astype(np.float32)
    if np.max(np.abs(mass - 1.0)) > 0.02:
        raise RuntimeError(f"{npz} top8+resid 质量和偏离 1: {mass}")
    k0 = int(data["hidden"].shape[0])
    _ = dequantize_hidden(data["hidden"][0, pick], data["hidden_scale"][0, pick])
    _ = reconstruct_pt(data["top8_idx"][pick], data["top8_prob"][pick])
    if k0 < 1:
        raise RuntimeError(f"{npz} K=0")


def pack_window(logits, emb, layers: tuple[str, ...], ids_np: np.ndarray, scale_dtype=np.float16):
    import torch

    from train.distill import unpack_logits

    probs = torch.softmax(unpack_logits(logits)[0].float(), dim=-1)
    p_np = probs.detach().cpu().numpy()
    tidx, tval, resid = pack_topk(p_np, k=TOPK)
    hq = []
    hs = []
    for name in layers:
        h = emb[name]
        if h.ndim == 3:
            h = h[0]
        q, s = quantize_hidden(h.detach().float().cpu().numpy(), scale_dtype=scale_dtype)
        if q.shape[-1] != HIDDEN_DIM:
            raise RuntimeError(f"{name} hidden dim {q.shape} != {HIDDEN_DIM}")
        hq.append(q)
        hs.append(s)
    hidden_q = np.stack(hq, axis=0)
    hidden_s = np.stack(hs, axis=0)
    ids_u8 = ids_np.astype(np.uint8)
    ent = entropy_bits(p_np)
    return ids_u8, tidx, tval, resid, hidden_q, hidden_s, ent


def dump_source(
    *,
    evo,
    name: str,
    layers: tuple[str, ...],
    quota: int,
    keep_prob: float,
    seed: int,
    window: int,
    shard_pos: int,
    out_dir: Path,
    device,
    max_positions: int | None,
    shard_tag: str = "",
    files_filter: set[str] | None = None,
    scale_dtype=np.float16,
) -> dict:
    import torch

    from train.distill import teacher_forward

    already, next_idx = existing_progress(out_dir, name, shard_tag)
    target = min(int(quota), int(max_positions or quota))
    if already >= target:
        log(f"{name}{shard_tag}: 已有 {already:,} ≥ 目标 {target:,}，跳过")
        return {"source": name, "n_pos": already, "skipped": True, "shards": []}

    writer = ShardWriter(out_dir, name, layers, shard_pos, next_idx, shard_tag=shard_tag)
    skip_pos = already
    n_new = 0
    hist = None
    shards: list[dict] = []
    t0 = time.perf_counter()
    n_fwd = 0
    t_fwd = 0.0
    log(f"{name}{shard_tag}: 续跑 already={already:,} next_shard={next_idx} quota={target:,} keep_prob={keep_prob:.6g}")
    for seq in iter_source_windows(name, keep_prob, seed, window, files_filter=files_filter):
        if already + n_new >= target:
            break
        if skip_pos >= window:
            skip_pos -= window
            continue
        ids_np = seq_to_ids(seq)
        ids_t = torch.as_tensor(ids_np, dtype=torch.long, device=device)[None]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        logits, emb = teacher_forward(evo, ids_t, layers)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_fwd += time.perf_counter() - t1
        n_fwd += 1
        ids_u8, tidx, tval, resid, hq, hs, ent = pack_window(logits, emb, layers, ids_np, scale_dtype=scale_dtype)
        del logits, emb, ids_t
        hist = merge_hists(hist, entropy_hist(ent))
        for rec in writer.add(ids_u8, tidx, tval, resid, hq, hs):
            verify_shard((out_dir / rec["npz"]).with_suffix(".json"))
            shards.append(rec)
        n_new += int(ids_u8.shape[0])
        if n_fwd <= 2 or n_fwd % 20 == 0:
            tok_s = (n_fwd * window) / max(t_fwd, 1e-6)
            log(f"{name}: win={n_fwd} new_pos={n_new:,} {tok_s:.0f} tok/s")
    last = writer.close()
    if last:
        verify_shard((out_dir / last["npz"]).with_suffix(".json"))
        shards.append(last)
        n_new = existing_progress(out_dir, name, shard_tag)[0] - already
    total = existing_progress(out_dir, name, shard_tag)[0]
    return {
        "source": name,
        "n_pos": total,
        "n_new": n_new,
        "quota": target,
        "keep_prob": keep_prob,
        "shortfall": max(0, target - total),
        "shards": shards,
        "entropy_hist": finalize_hist(hist) if hist else None,
        "elapsed_s": time.perf_counter() - t0,
        "tok_s": (n_fwd * window) / max(t_fwd, 1e-6) if n_fwd else None,
    }


def qc_probe(evo, device, window: int) -> dict:
    """Plan 01 §2.3 同三条序列的熵对照（质控红线）。"""
    import torch

    from train.distill import teacher_forward, unpack_logits

    samples = {r.name: r.seq for r in load_dna_samples()}
    rng = np.random.default_rng(42)
    samples["random_acgt"] = "".join("ACGT"[int(i)] for i in rng.integers(0, 4, size=min(window, 4096)))
    out = {}
    for name, seq in samples.items():
        seq = seq[:window]
        ids = seq_to_ids(seq)
        ids_t = torch.as_tensor(ids, dtype=torch.long, device=device)[None]
        logits, _ = teacher_forward(evo, ids_t, ())
        probs = torch.softmax(unpack_logits(logits)[0].float(), dim=-1)
        p = probs[:-1].detach().cpu().numpy()
        ent = entropy_bits(p)
        rec = {
            "n_positions": int(ent.size),
            "entropy_mean": float(np.mean(ent)),
            "entropy_median": float(np.median(ent)),
            "frac_lt_1_5bit": float((ent < 1.5).mean()),
            "frac_gt_1_9bit": float((ent > 1.9).mean()),
            "hist": entropy_hist(ent),
        }
        if name in PLAN01_ENTROPY:
            anc = PLAN01_ENTROPY[name]
            rec["plan01"] = anc
            rec["delta_mean"] = rec["entropy_mean"] - anc["entropy_mean"]
            rec["delta_frac_lt_1_5"] = rec["frac_lt_1_5bit"] - anc["frac_lt_1_5bit"]
            rec["pass"] = abs(rec["delta_mean"]) <= 0.08 and abs(rec["delta_frac_lt_1_5"]) <= 0.05
        elif name == "random_acgt":
            rec["pass"] = rec["entropy_mean"] >= 1.90
        out[name] = rec
        log(
            f"QC {name}: mean={rec['entropy_mean']:.3f} <1.5={rec['frac_lt_1_5bit']:.3f} "
            f"pass={rec.get('pass')}"
        )
        del logits, ids_t
    return out


def update_manifest(out_dir: Path, extra: dict, name: str = "manifest.json") -> dict:
    path = out_dir / name
    old = {}
    if path.exists():
        try:
            old = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            old = {}
    man = {**old, **extra, "updated": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_json(path, man)
    return man


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="C.1 teacher-only 压缩落盘")
    p.add_argument("--scheme", default="L27", help="SCHEME_LAYERS 名，默认 L27")
    p.add_argument("--layers", default=None, help="覆盖 scheme，逗号分隔 blocks.N")
    p.add_argument("--sources", default=None, help="逗号分隔源子集；默认 Step 14 全源")
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--window", type=int, default=WINDOW)
    p.add_argument("--shard-pos", type=int, default=SHARD_POS)
    p.add_argument("--total-positions", type=int, default=TOTAL_POS)
    p.add_argument("--max-positions", type=int, default=None, help="本进程位置上限（冒烟）")
    p.add_argument("--seed", type=int, default=20260821)
    p.add_argument("--no-euk", action="store_true")
    p.add_argument("--no-imgvr", action="store_true",
                   help="不落盘 imgvr（Step 15 前口径复现用）")
    p.add_argument("--no-kernels", action="store_true")
    p.add_argument("--skip-qc-probe", action="store_true")
    p.add_argument("--qc-only", action="store_true", help="只跑 Plan 01 熵探针，不落盘")
    p.add_argument("--smoke", action="store_true", help="GPU1 冒烟：10 万位置、窗 2048、本地小源")
    p.add_argument("--manifest-name", default="manifest.json", help="本进程写入的 manifest 文件名")
    p.add_argument("--source-files", default=None,
                   help="限定 --sources 单源的 chunk 文件名白名单（逗号分隔；双卡分流用）")
    p.add_argument("--shard-tag", default="",
                   help="分片文件名后缀标签（如 A/B；meta.source 不变，双卡分流互不覆盖）")
    p.add_argument("--scale-fp32", action="store_true",
                   help="hidden 量化 scale 用 fp32（末段 blocks.30/31 残差 ~1e11 会溢出 fp16；"
                        "Hyena 层默认 fp16 不变，读取侧 dtype 无关）")
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    import torch

    from train.distill import assert_layers_exist, load_evo2

    if args.smoke:
        args.window = min(args.window, 2048)
        args.shard_pos = min(args.shard_pos, 50_000)
        args.max_positions = args.max_positions or 100_000
        args.sources = args.sources or "hg38,ncbi,promoters"
        log("冒烟模式：window=2048（GPU1 24GB；4k 前向 Plan 01 测 31.4GB，48G 卡上正式跑）")

    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA")
    device = torch.device("cuda:0")
    layers = resolve_layers(args.scheme, args.layers)
    with_euk = not args.no_euk
    with_imgvr = not (args.no_imgvr or getattr(args, "smoke", False))
    weights = step14_weights(with_euk=with_euk, with_imgvr=with_imgvr)
    if args.sources:
        want = [s.strip() for s in args.sources.split(",") if s.strip()]
    else:
        want = [k for k in weights]
    for s in want:
        if s not in weights and s not in SOURCE_SUB and s not in ("hg38", "ncbi"):
            raise RuntimeError(f"未知源 {s}")
    files_filter = None
    if args.source_files:
        if len(want) != 1 or want[0] in ("hg38", "ncbi"):
            raise RuntimeError("--source-files 仅支持单一 OG2 源分流")
        files_filter = {x.strip() for x in args.source_files.split(",") if x.strip()}
    # 正式落盘：配额按全局 Step 14 权重 × 3e8，子集不重新归一（双卡各跑一半源）
    # 冒烟：在所选源之间把 max_positions 摊开
    sub_w = {k: weights.get(k, 0.0) for k in want}
    if sum(sub_w.values()) <= 0:
        sub_w = {k: 1.0 / len(want) for k in want}
    if args.smoke:
        z = sum(sub_w.values()) or 1.0
        sub_w = {k: v / z for k, v in sub_w.items()}
        total = int(args.max_positions or 100_000)
    else:
        total = int(args.total_positions)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    props = torch.cuda.get_device_properties(0)
    log(
        f"visible={os.environ.get('CUDA_VISIBLE_DEVICES')!r} {props.name} "
        f"{props.total_memory / 1e9:.1f}GB layers={layers} window={args.window}"
    )
    evo = load_evo2(use_kernels=not args.no_kernels)
    assert_layers_exist(evo, layers)

    qc = None
    if args.qc_only or not args.skip_qc_probe:
        try:
            qc = qc_probe(evo, device, min(args.window, 4096))
            failed = [k for k, v in qc.items() if v.get("pass") is False]
            if failed:
                raise RuntimeError(f"Plan 01 熵探针未过: {failed}")
        except Exception as exc:
            if "out of memory" in str(exc).lower():
                log(f"QC 探针 OOM（窗 {args.window}），跳过探针、继续落盘：{exc}")
                qc = {"skipped": "oom", "error": str(exc)}
            else:
                raise
        if args.qc_only:
            rec = {"qc_probe": qc, "status": "qc_only"}
            save_json(out_dir / "qc_probe.json", rec)
            return rec

    per_source = {}
    for name in want:
        files, est = source_files_and_est(name, files_filter)
        quota = int(round(sub_w[name] * total))
        keep = min(1.0, quota / max(int(est), 1))
        n_files = len(files) if isinstance(files, list) else 0
        log(f"计划 {name}{args.shard_tag}: quota={quota:,} est_bp={int(est):,} keep_prob={keep:.6g} files={n_files}")
        if name in SOURCE_SUB and n_files == 0:
            log(f"警告：源 {name} 无完整 chunk，跳过")
            per_source[name] = {"source": name, "n_pos": 0, "skipped": True, "reason": "no_chunks"}
            continue
        try:
            per_source[name] = dump_source(
                evo=evo,
                name=name,
                layers=layers,
                quota=quota,
                keep_prob=keep,
                seed=args.seed + 17 * sum(ord(c) for c in name),
                window=args.window,
                shard_pos=args.shard_pos,
                out_dir=out_dir,
                scale_dtype=(np.float32 if args.scale_fp32 else np.float16),
                device=device,
                max_positions=args.max_positions,
                shard_tag=args.shard_tag,
                files_filter=files_filter,
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                log(f"{name} OOM：{exc}")
                raise
            raise

    n_total = sum(int(v.get("n_pos") or 0) for v in per_source.values())
    mix_hist = None
    for v in per_source.values():
        h = v.get("entropy_hist")
        if h and h.get("n"):
            mix_hist = merge_hists(mix_hist, {**h, "mean": h.get("mean") or 0.0})
    man = update_manifest(
        out_dir,
        {
            "status": "done",
            "scheme": args.scheme,
            "layers": list(layers),
            "window": args.window,
            "shard_pos": args.shard_pos,
            "weights": sub_w,
            "with_euk": with_euk,
            "n_positions": n_total,
            "per_source": {
                k: {kk: vv for kk, vv in v.items() if kk != "shards"}
                | {"n_shards": len(v.get("shards") or [])}
                for k, v in per_source.items()
            },
            "mixture_entropy": finalize_hist(mix_hist) if mix_hist else None,
            "qc_probe": qc,
            "note": (
                "窗长 4096 取 Plan 01 全并行预填上限内最大整齐窗口；"
                "偏离 plans/02 的 4k–16k 区间取下限（16k 带态分块 3000 tok/s 过慢）"
            ),
        },
        name=args.manifest_name,
    )
    log(f"完成 n_pos={n_total:,} manifest={out_dir / args.manifest_name}")
    return man


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
