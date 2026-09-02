"""C.1 落盘压缩/解压（CPU，无 evo2 依赖）。

p_t：top-8 (idx uint16, prob fp16) + 残差质量 fp16。
hidden：逐 token absmax 的 int8 + scale fp16。
"""

from __future__ import annotations

import zlib

import numpy as np

VOCAB_SIZE = 512
HIDDEN_DIM = 4096
TOPK = 8


def quantize_hidden(h: np.ndarray, scale_dtype=np.float16) -> tuple[np.ndarray, np.ndarray]:
    """``h`` [L, D] float → (int8 [L, D], scale [L])，逐 token absmax。

    ``scale_dtype`` 默认 fp16（Hyena 层 hidden absmax ~O(1–1e3) 足够）；末段
    blocks.30/31 的残差流 hidden absmax ~1e11 会溢出 fp16（Step 16 实测，
    scale 全表变 inf）——此类层必须传 ``np.float32``。溢出守卫直接抛错，
    避免整轮 dump 静默产出废数据。
    """
    if h.ndim != 2:
        raise ValueError(f"hidden 期望 [L, D]，得到 {h.shape}")
    scale = np.max(np.abs(h.astype(np.float32)), axis=-1)
    scale = np.maximum(scale, 1e-8)
    dt = np.dtype(scale_dtype)
    if np.isinf(scale.astype(dt)).any():
        raise ValueError(
            f"hidden absmax 超出 scale dtype {dt} 表示范围（max={scale.max():.3g}）；"
            f"该层 hidden 量级过大，请用 scale_dtype=np.float32（dump 侧 --scale-fp32）"
        )
    q = np.clip(np.rint(h.astype(np.float32) / scale[:, None] * 127.0), -127, 127).astype(np.int8)
    return q, scale.astype(dt)


def dequantize_hidden(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    s = np.asarray(scale, dtype=np.float32)
    return q.astype(np.float32) * (s[:, None] / 127.0)


def pack_topk(probs: np.ndarray, k: int = TOPK) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``probs`` [L, V] → (idx uint16 [L,k], val fp16 [L,k], resid fp16 [L])。"""
    if probs.ndim != 2:
        raise ValueError(f"probs 期望 [L, V]，得到 {probs.shape}")
    k = min(int(k), int(probs.shape[-1]))
    # argpartition 无序；再对 top-k 按值降序
    part = np.argpartition(-probs, kth=k - 1, axis=-1)[:, :k]
    gathered = np.take_along_axis(probs, part, axis=-1)
    order = np.argsort(-gathered, axis=-1)
    idx = np.take_along_axis(part, order, axis=-1).astype(np.uint16)
    val = np.take_along_axis(gathered, order, axis=-1).astype(np.float16)
    resid = (1.0 - gathered.astype(np.float64).sum(axis=-1)).astype(np.float16)
    return idx, val, resid


def reconstruct_pt(
    idx: np.ndarray,
    val: np.ndarray,
    resid: np.ndarray | None = None,
    vocab: int = VOCAB_SIZE,
) -> np.ndarray:
    """稀疏重建（残差质量不摊回词表，仅可从 ``resid`` 读取）。"""
    L, k = idx.shape
    out = np.zeros((L, vocab), dtype=np.float32)
    rows = np.arange(L)[:, None]
    out[rows, idx.astype(np.int64)] = val.astype(np.float32)
    return out


def entropy_bits(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs.astype(np.float64), 1e-12, 1.0)
    return -(p * np.log2(p)).sum(axis=-1)


def entropy_hist(ent: np.ndarray, bin_width: float = 0.1, hi: float = 2.0) -> dict:
    edges = np.arange(0.0, hi + bin_width * 0.5, bin_width)
    hist, _ = np.histogram(ent, bins=edges)
    return {
        "edges": [float(x) for x in edges.tolist()],
        "counts": [int(x) for x in hist.tolist()],
        "n": int(ent.size),
        "mean": float(np.mean(ent)) if ent.size else float("nan"),
        "median": float(np.median(ent)) if ent.size else float("nan"),
        "frac_lt_1_5bit": float((ent < 1.5).mean()) if ent.size else float("nan"),
        "frac_gt_1_9bit": float((ent > 1.9).mean()) if ent.size else float("nan"),
    }


def merge_hists(a: dict | None, b: dict) -> dict:
    if a is None:
        return {
            "edges": list(b["edges"]),
            "counts": list(b["counts"]),
            "n": int(b["n"]),
            "sum_ent": float(b["mean"]) * int(b["n"]) if b["n"] else 0.0,
        }
    if a["edges"] != b["edges"]:
        raise ValueError("熵直方图 bin 不一致")
    n = int(a["n"]) + int(b["n"])
    prev_sum = float(a.get("sum_ent", float(a.get("mean", 0.0)) * int(a["n"])))
    return {
        "edges": list(a["edges"]),
        "counts": [int(x) + int(y) for x, y in zip(a["counts"], b["counts"])],
        "n": n,
        "sum_ent": prev_sum + float(b["mean"]) * int(b["n"]),
    }


def finalize_hist(h: dict) -> dict:
    n = int(h.get("n") or 0)
    mean = (float(h.get("sum_ent", 0.0)) / n) if n else float("nan")
    return {
        "edges": h["edges"],
        "counts": h["counts"],
        "n": n,
        "mean": mean,
    }


def arrays_crc(*arrays: np.ndarray) -> str:
    c = 0
    for a in arrays:
        c = zlib.crc32(np.ascontiguousarray(a).tobytes(), c)
    return f"{c & 0xffffffff:08x}"
