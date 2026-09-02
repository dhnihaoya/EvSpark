"""在线蒸馏循环 + 免解码评估 + 取层×宽度消融（GPU2）。

用法（必须可见设备为物理 GPU2）::

    CUDA_VISIBLE_DEVICES=2 HF_HOME=./hf_home \\
      /home/dh/miniconda3/envs/evo2/bin/python scripts/train/distill.py --grid
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / "hf_home"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import numpy as np
import torch

from specdec.markov import DEFAULT_ALPHA, MarkovDraftModel, MarkovTable
from train.data import (
    LabeledSeq,
    holdout_eval_window,
    load_default_corpus,
    load_dna_samples,
    load_hg38,
    sample_anchors,
    sample_window_ids,
    seq_to_ids,
    split_corpus,
)
from train.drafter import (
    CE_W,
    CONF_W,
    DEFAULT_GAMMA,
    MASK_ID,
    SCHEME_LAYERS,
    TARGET_HIDDEN,
    TV_W,
    VOCAB_SIZE,
    Drafter,
    analytic_accept,
    apply_transform_torch,
    distill_losses,
    predicted_tau,
)

REPO = Path(__file__).resolve().parents[2]
RESULTS_PATH = REPO / "benchmarks" / "phasec_smoke.json"
CKPT_DIR = REPO / "benchmarks" / "phasec_ckpts"
STEP6_TAU_LACZ = 2.61  # Step 6 编码区 Markov 实测 τ，对照锚

# DDP（Step 12）：torchrun 注入 RANK/WORLD_SIZE；非分布式时 RANK=0 行为与
# 历版完全一致。rank≠0 静默（日志/落盘/评估/存 ckpt 都只在 rank0）
_RANK = int(os.environ.get("RANK", "0"))
ALL_INJECT_LAYERS = (
    "blocks.6",
    "blocks.7",
    "blocks.15",
    "blocks.16",
    "blocks.20",
    "blocks.23",
    "blocks.27",
    "blocks.30",
    "blocks.31",
)
GRID_D512 = (("S1", 512), ("S0", 512), ("S2", 512), ("S3", 512))


def log(msg: str) -> None:
    if _RANK != 0:
        return
    print(f"[phasec] {msg}", flush=True)


def seed_all(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def unpack_logits(raw):
    if isinstance(raw, tuple):
        return raw[0]
    return raw


def probe_target(evo) -> dict[str, torch.Tensor]:
    """只拷若干切片，避免把 7B 全量 clone 进内存。"""
    blk = evo.model.blocks[31]
    p0 = next(blk.parameters())
    return {
        "embed": evo.model.embedding_layer.weight.detach()[:8, :8].cpu().clone(),
        "blocks31": p0.detach().flatten()[:64].cpu().clone(),
    }


def probe_max_abs_diff(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    mx = 0.0
    for k in a:
        mx = max(mx, float((a[k].float() - b[k].float()).abs().max().item()))
    return mx


def concat_hidden(emb: dict[str, torch.Tensor], layer_names: tuple[str, ...]) -> torch.Tensor | None:
    if not layer_names:
        return None
    pieces = []
    for name in layer_names:
        h = emb[name]
        if h.ndim == 3:
            h = h[0]
        pieces.append(h.float())
    return torch.cat(pieces, dim=-1)


@torch.no_grad()
def teacher_forward(evo, ids: torch.Tensor, layer_names: tuple[str, ...]):
    if layer_names:
        raw, emb = evo.forward(ids, return_embeddings=True, layer_names=list(layer_names))
    else:
        raw, emb = evo.forward(ids, return_embeddings=False)
        emb = {}
    logits = unpack_logits(raw)
    return logits, emb


def load_evo2(use_kernels: bool = True):
    from evo2 import Evo2

    log("加载 evo2_7b …")
    evo = Evo2("evo2_7b", use_kernels=use_kernels)
    evo.model.eval()
    for p in evo.model.parameters():
        p.requires_grad_(False)
    return evo


def assert_layers_exist(evo, names: tuple[str, ...]) -> None:
    mods = {n for n, _ in evo.model.named_modules()}
    missing = [n for n in names if n not in mods]
    if missing:
        sample = sorted(m for m in mods if m.startswith("blocks."))[:12]
        raise RuntimeError(f"找不到层 {missing}；blocks.* 示例 {sample}")


@dataclass
class EvalPack:
    name: str
    ids: np.ndarray
    p_t: np.ndarray  # [L, V] float32
    hidden: dict[str, np.ndarray]  # layer -> [L, 4096]
    holdout_start: int


def holdout_anchors(length: int, holdout_start: int, gamma: int, cap: int) -> np.ndarray:
    lo = max(int(holdout_start) - 1, 0)
    hi = int(length) - int(gamma) - 1
    if hi < lo:
        return np.empty(0, dtype=np.int64)
    pool = np.arange(lo, hi + 1, dtype=np.int64)
    if pool.size <= cap:
        return pool
    ix = np.linspace(0, pool.size - 1, cap).astype(np.int64)
    return pool[ix]


def gather_batch(
    ids: torch.Tensor,
    p_t: torch.Tensor,
    anchors: torch.Tensor,
    gamma: int,
):
    prev = torch.stack([ids[a : a + gamma] for a in anchors])
    target = torch.stack([ids[a + 1 : a + 1 + gamma] for a in anchors])
    pt = torch.stack([p_t[a : a + gamma] for a in anchors])
    return ids[anchors], prev, target, pt, anchors + 1


@torch.no_grad()
def eval_drafter(
    drafter: Drafter,
    packs: list[EvalPack],
    scheme: str,
    gamma: int,
    device: torch.device,
    cap_anchors: int,
) -> dict:
    layers = SCHEME_LAYERS[scheme]
    alpha_sum = None
    n = 0
    top1_pt = 0
    top1_gt = 0
    n_tok = 0
    per_name: dict[str, dict] = {}
    drafter.eval()
    for pack in packs:
        anchors = holdout_anchors(int(pack.ids.size), pack.holdout_start, gamma, cap_anchors)
        if anchors.size == 0:
            continue
        ids = torch.as_tensor(pack.ids, dtype=torch.long, device=device)
        p_t = torch.as_tensor(pack.p_t, dtype=torch.float32, device=device)
        if layers:
            h_raw = torch.cat(
                [torch.as_tensor(pack.hidden[ln], dtype=torch.float32, device=device) for ln in layers],
                dim=-1,
            )
        else:
            h_raw = None
        a_t = torch.as_tensor(anchors, dtype=torch.long, device=device)
        anchor_ids, prev, target, pt, ctx_lens = gather_batch(ids, p_t, a_t, gamma)
        _, probs, _, _ = drafter(anchor_ids, prev, ctx_lens, h_raw=h_raw)
        cstar = analytic_accept(probs, pt)
        a_mean = cstar.mean(dim=0)
        n_b = int(cstar.shape[0])
        per_name[pack.name] = {
            "n_blocks": n_b,
            "alpha_bar": [float(x) for x in a_mean.cpu().tolist()],
            "tau_hat": float(predicted_tau(a_mean).item()),
            "top1_vs_pt": float((probs.argmax(-1) == pt.argmax(-1)).float().mean().item()),
        }
        if alpha_sum is None:
            alpha_sum = cstar.sum(dim=0)
        else:
            alpha_sum = alpha_sum + cstar.sum(dim=0)
        n += n_b
        top1_pt += int((probs.argmax(-1) == pt.argmax(-1)).sum().item())
        top1_gt += int((probs.argmax(-1) == target).sum().item())
        n_tok += int(cstar.numel())
    drafter.train()
    if n == 0 or alpha_sum is None:
        return {"n_blocks": 0, "alpha_bar": [], "tau_hat": float("nan"), "per_name": per_name}
    alpha_bar = (alpha_sum / n).cpu()
    tau = float(predicted_tau(alpha_bar).item())
    return {
        "n_blocks": n,
        "n_positions": n_tok,
        "alpha_bar": [float(x) for x in alpha_bar.tolist()],
        "tau_hat": tau,
        "top1_vs_pt": (top1_pt / n_tok) if n_tok else float("nan"),
        "top1_vs_gt": (top1_gt / n_tok) if n_tok else float("nan"),
        "per_name": per_name,
    }


def eval_markov(
    table_model: MarkovDraftModel,
    packs: list[EvalPack],
    gamma: int,
    cap_anchors: int,
) -> dict:
    alpha_sum = None
    n = 0
    top1_pt = 0
    top1_gt = 0
    n_tok = 0
    per_name: dict[str, dict] = {}
    for pack in packs:
        anchors = holdout_anchors(int(pack.ids.size), pack.holdout_start, gamma, cap_anchors)
        if anchors.size == 0:
            continue
        ids = pack.ids
        p_t = pack.p_t
        B = int(anchors.size)
        pd = np.zeros((B, gamma, VOCAB_SIZE), dtype=np.float64)
        tgt = np.zeros((B, gamma), dtype=np.int64)
        pt_b = np.zeros((B, gamma, VOCAB_SIZE), dtype=np.float64)
        for i, a in enumerate(anchors):
            a = int(a)
            buf = ids[: a + 1]
            for k in range(gamma):
                pd[i, k] = table_model.probs(buf)
                nxt = int(ids[a + 1 + k])
                tgt[i, k] = nxt
                pt_b[i, k] = p_t[a + k]
                buf = np.concatenate([buf, np.array([nxt], dtype=ids.dtype)])
        cstar = 1.0 - 0.5 * np.abs(pd - pt_b).sum(axis=-1)
        a_mean = cstar.mean(axis=0)
        per_name[pack.name] = {
            "n_blocks": B,
            "alpha_bar": [float(x) for x in a_mean.tolist()],
            "tau_hat": float(1.0 + np.cumprod(a_mean).sum()),
            "top1_vs_pt": float((pd.argmax(-1) == pt_b.argmax(-1)).mean()),
        }
        if alpha_sum is None:
            alpha_sum = cstar.sum(axis=0)
        else:
            alpha_sum = alpha_sum + cstar.sum(axis=0)
        n += B
        top1_pt += int((pd.argmax(-1) == pt_b.argmax(-1)).sum())
        top1_gt += int((pd.argmax(-1) == tgt).sum())
        n_tok += B * gamma
    if n == 0 or alpha_sum is None:
        return {"n_blocks": 0, "alpha_bar": [], "tau_hat": float("nan"), "per_name": per_name}
    alpha_bar = alpha_sum / n
    tau = float(1.0 + np.cumprod(alpha_bar).sum())
    return {
        "n_blocks": n,
        "n_positions": n_tok,
        "alpha_bar": [float(x) for x in alpha_bar.tolist()],
        "tau_hat": tau,
        "top1_vs_pt": (top1_pt / n_tok) if n_tok else float("nan"),
        "top1_vs_gt": (top1_gt / n_tok) if n_tok else float("nan"),
        "per_name": per_name,
        "note": "解析接受率，teacher-forced 前缀；τ̂ 为近似公式",
    }


def pick_eval_records(n_hg: int = 16) -> list[LabeledSeq]:
    hg = load_hg38()
    dna = load_dna_samples()
    if not hg:
        return list(dna)
    ix = np.unique(np.linspace(0, len(hg) - 1, num=min(n_hg, len(hg))).astype(int))
    return [hg[int(i)] for i in ix] + list(dna)


def build_eval_packs(
    evo,
    records: list[LabeledSeq],
    window: int,
    device: torch.device,
    layer_names: tuple[str, ...],
) -> list[EvalPack]:
    packs: list[EvalPack] = []
    for rec in records:
        ids_np, hold_s, _ = holdout_eval_window(rec.seq, window=window)
        if ids_np.size < DEFAULT_GAMMA + 8:
            continue
        ids_t = torch.as_tensor(ids_np, dtype=torch.long, device=device)[None]
        t0 = time.perf_counter()
        logits, emb = teacher_forward(evo, ids_t, layer_names)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        p_t = apply_transform_torch(unpack_logits(logits)[0], temperature=1.0, top_k=4)
        hidden = {}
        for name, h in (emb or {}).items():
            hh = h[0] if h.ndim == 3 else h
            hidden[name] = hh.detach().float().cpu().numpy()
        packs.append(
            EvalPack(
                name=rec.name,
                ids=ids_np,
                p_t=p_t.detach().float().cpu().numpy(),
                hidden=hidden,
                holdout_start=int(hold_s),
            )
        )
        log(f"eval teacher {rec.name} L={ids_np.size} {dt:.2f}s holdout_start={hold_s}")
        del logits, emb, p_t, ids_t
    return packs


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
    tmp.replace(path)


def train_one_cell(
    *,
    evo,
    train_recs: list[LabeledSeq],
    eval_packs: list[EvalPack],
    markov_metrics: dict,
    scheme: str,
    d_model: int,
    gamma: int,
    window: int,
    n_anchors: int,
    min_ctx: int,
    max_positions: int,
    lr: float,
    eval_every: int,
    seed: int,
    device: torch.device,
    ckpt_dir: Path,
    cap_eval_anchors: int,
    ctx_window: int = 64,
    ckpt_name: str | None = None,
    window_sampler=None,
    ddp_world_size: int = 1,
    compile_drafter: bool = False,
    amp: bool = False,
) -> dict:
    """在线蒸馏一格。``ddp_world_size > 1`` 时（Step 12 起，torchrun 启动）：
    teacher/drafter 每卡各一份，各 rank 独立抽窗抽 64 锚，前向/反向走 DDP 包装
    （梯度跨 rank 平均 → 单步有效 batch = n_anchors×world），位置计数按全局
    （n_anchors×γ×world/步），优化器步数相应减半；日志/评估/ckpt/落盘仅 rank0。
    默认 ``ddp_world_size=1`` 时与 Step 8–11 单卡路径逐行为等。
    ``compile_drafter=True`` 时用 ``torch.compile`` 包装训练前向（锚点数/窗长
    固定，无动态形状重编译）；评估仍走 eager 以保持 Step 8–11 口径。kernel
    融合改变归约顺序，loss/权重与 eager 不逐位一致，仅在确认吞吐收益后启用。
    ``amp=True`` 时 drafter 前向走 bf16 autocast（损失留在 autocast 外、内部
    全程 .float() 与 fp32 路径同数值——BCE 是 autocast 禁 op；参数与优化器
    保持 fp32，bf16 无需 GradScaler；teacher 本就 bf16 原生 no_grad，不受影响）。"""
    seed_all(seed)
    if window_sampler is None:
        window_sampler = sample_window_ids
    drafter = Drafter.from_scheme(scheme, d_model=d_model, gamma=gamma, ctx_window=ctx_window)
    w_emb = evo.model.embedding_layer.weight.detach().float().cpu()
    drafter.load_frozen_embedding(w_emb)
    drafter.to(device)
    ddp = None
    if int(ddp_world_size) > 1:
        ddp = torch.nn.parallel.DistributedDataParallel(
            drafter,
            device_ids=[torch.cuda.current_device()],
            find_unused_parameters=False,  # drafter 前向固定图，实测无未用参数（冒烟验证）
            broadcast_buffers=False,
        )
    fwd_model = ddp if ddp is not None else drafter
    if compile_drafter:
        fwd_model = torch.compile(fwd_model)
    frozen_before = {k: v.detach().cpu().clone() for k, v in drafter.frozen_named_tensors().items()}
    target_before = probe_target(evo)
    opt = torch.optim.AdamW(drafter.trainable_parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    pos_per_step = max(n_anchors * gamma * int(ddp_world_size), 1)
    n_steps_est = max(1, int(np.ceil(max_positions / pos_per_step)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps_est, eta_min=lr * 0.1)

    # 各 rank 采样流错开（同种子体系下加 rank 偏移）；rank0/非分布式 = 原种子
    rng = np.random.default_rng(seed + 10_000 * _RANK)
    layers = SCHEME_LAYERS[scheme]
    curve: list[dict] = []
    n_pos = 0
    step = 0
    t_target_sum = 0.0
    t_draft_sum = 0.0
    n_timed = 0
    last_eval = None
    t_cell0 = time.perf_counter()
    log(
        f"开始 {scheme} d={d_model} ctx_window={ctx_window} "
        f"可训练参数 {sum(p.numel() for p in drafter.trainable_parameters()):,}"
        f" 目标 {max_positions:,} 位置"
    )

    while n_pos < max_positions:
        ids_np = window_sampler(train_recs, window, rng)
        anchors_np = sample_anchors(int(ids_np.size), gamma, n_anchors, rng, min_ctx=min_ctx)
        if anchors_np.size == 0:
            if ddp is not None:
                # 各 rank 步数必须严格对称，否则 allreduce 挂死；直接报错好过挂起
                raise RuntimeError("DDP 模式下锚点为空（窗过短？）——拒绝不对称步")
            continue
        ids_t = torch.as_tensor(ids_np, dtype=torch.long, device=device)[None]
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits, emb = teacher_forward(evo, ids_t, layers)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_target = time.perf_counter() - t0
        p_t_full = apply_transform_torch(unpack_logits(logits)[0], temperature=1.0, top_k=4)
        h_raw = concat_hidden(emb or {}, layers)
        if h_raw is not None:
            h_raw = h_raw.detach()
        del logits, emb, ids_t

        ids = torch.as_tensor(ids_np, dtype=torch.long, device=device)
        a_t = torch.as_tensor(anchors_np, dtype=torch.long, device=device)
        anchor_ids, prev, target, pt, ctx_lens = gather_batch(ids, p_t_full, a_t, gamma)
        pt = pt.detach()

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        amp_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if amp and device.type == "cuda"
            else contextlib.nullcontext()
        )
        with amp_ctx:
            logits, probs, conf, _ = fwd_model(anchor_ids, prev, ctx_lens, h_raw=h_raw)
        # 损失留在 autocast 外：distill_losses 内部全程 .float()，与 fp32 路径同数值；
        # F.binary_cross_entropy 是 autocast 禁 op（unsafe to autocast），必须在外
        br = distill_losses(logits, pt, target, conf, p_d=probs)
        br.total.backward()
        torch.nn.utils.clip_grad_norm_(drafter.trainable_parameters(), 1.0)
        opt.step()
        sched.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_draft = time.perf_counter() - t1

        n_this = int(anchors_np.size) * gamma * int(ddp_world_size)  # 全局监督位置
        n_pos += n_this
        step += 1
        t_target_sum += t_target
        t_draft_sum += t_draft
        n_timed += 1
        with torch.no_grad():
            a_mean = float(analytic_accept(probs, pt).mean().item())
        rec = {
            "step": step,
            "n_pos": n_pos,
            "loss": float(br.total.item()),
            "ce": float(br.ce.item()),
            "tv": float(br.tv.item()),
            "conf": float(br.conf.item()),
            "train_cstar": a_mean,
            "t_target_s": t_target,
            "t_draft_s": t_draft,
            "lr": float(opt.param_groups[0]["lr"]),
        }
        if step <= 3 or step % 20 == 0:
            log(
                f"{scheme}/d{d_model} step={step} pos={n_pos} loss={rec['loss']:.4f} "
                f"c*={a_mean:.3f} tgt={t_target:.2f}s draft={t_draft:.3f}s"
            )
        if step == 1 or step % 25 == 0:
            curve.append(rec)
        do_eval = (n_pos >= max_positions) or (
            eval_every > 0 and n_pos // eval_every > (n_pos - n_this) // eval_every and n_pos >= eval_every
        )
        if do_eval and _RANK == 0:
            last_eval = eval_drafter(
                drafter, eval_packs, scheme, gamma, device, cap_eval_anchors
            )
            log(f"eval {scheme}/d{d_model} τ̂={last_eval.get('tau_hat')} ᾱ={last_eval.get('alpha_bar')}")
        del logits, probs, conf, br, p_t_full, h_raw, ids, a_t

    frozen_after = {k: v.detach().cpu() for k, v in drafter.frozen_named_tensors().items()}
    emb_delta = float((frozen_before["embed.weight"] - frozen_after["embed.weight"]).abs().max().item())
    target_delta = probe_max_abs_diff(target_before, probe_target(evo))
    if last_eval is None and _RANK == 0:
        last_eval = eval_drafter(drafter, eval_packs, scheme, gamma, device, cap_eval_anchors)

    ckpt_path = None
    if _RANK == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / (ckpt_name if ckpt_name else f"{scheme}_d{d_model}.pt")
        torch.save(
            {
                "scheme": scheme,
                "d_model": d_model,
                "gamma": gamma,
                "state_dict": drafter.state_dict(),
                "n_pos": n_pos,
                "mask_id": MASK_ID,
            },
            ckpt_path,
        )

    avg_tgt = t_target_sum / max(n_timed, 1)
    avg_dr = t_draft_sum / max(n_timed, 1)
    pos_per_s = n_pos / max(time.perf_counter() - t_cell0, 1e-6)
    peak = None
    if device.type == "cuda":
        peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    out = {
        "scheme": scheme,
        "layers": list(layers),
        "d_model": d_model,
        "gamma": gamma,
        "n_positions": n_pos,
        "n_steps": step,
        "window": window,
        "n_anchors": n_anchors,
        "ctx_window": ctx_window,
        "lr": lr,
        "mask_id": MASK_ID,
        "loss_weights": {"ce": CE_W, "tv": TV_W, "conf": CONF_W},
        "trainable_params": int(sum(p.numel() for p in drafter.trainable_parameters())),
        "freeze": {"embed_max_abs_delta": emb_delta, "target_max_abs_delta": target_delta},
        "throughput": {
            "target_s_per_step": avg_tgt,
            "drafter_s_per_step": avg_dr,
            "positions_per_s": pos_per_s,
            "target_tok_s": (window / avg_tgt) if avg_tgt > 0 else float("nan"),
            "peak_mem_gb": peak,
        },
        "train_curve": curve,
        "eval": last_eval,
        "markov_floor": markov_metrics,
        "ckpt": str(ckpt_path),
        "elapsed_s": time.perf_counter() - t_cell0,
    }
    log(
        f"完成 {scheme}/d{d_model} τ̂={(last_eval or {}).get('tau_hat')} "
        f"freeze embed Δ={emb_delta:.2e} target Δ={target_delta:.2e} "
        f"{avg_tgt:.2f}s/tgt {avg_dr:.3f}s/draft {pos_per_s:.1f} pos/s"
    )
    del drafter, opt, sched
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def estimate_formal_hours(target_tok_s: float, n_positions: float = 3.0e8) -> dict:
    """C.1 正式期离线生成工时。对照 Plan 01 分块预填 ~3000 tok/s。"""
    chunked = 3000.0
    online = float(target_tok_s) if target_tok_s and target_tok_s > 0 else float("nan")
    return {
        "n_positions": n_positions,
        "plan01_chunked_tok_s": chunked,
        "smoke_teacher_tok_s": online,
        "hours_at_plan01_3000": (n_positions / chunked) / 3600.0,
        "hours_at_smoke_teacher": ((n_positions / online) / 3600.0) if online == online and online > 0 else None,
        "storage_note": "p_t top-8 fp16+残差 ~32B/pos；1 层 hidden int8 ~4KB/pos；3e8 pos ≈ 1.2TB",
    }


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA（Plan 10 约定 GPU2）")
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    log(
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} "
        f"name={props.name} mem={props.total_memory / 1e9:.1f}GB"
    )
    if props.total_memory < 30e9:
        log("警告：可见卡 <30GB，可能不是 GPU2（48GB）")
    seed_all(args.seed)
    corpus = load_default_corpus(fasta_dir=args.fasta_dir)
    train_recs, _ = split_corpus(corpus)
    log(f"训练序列 {len(train_recs)} 条，bp={sum(len(r.seq) for r in train_recs):,}")

    evo = load_evo2(use_kernels=not args.no_kernels)
    assert_layers_exist(evo, ALL_INJECT_LAYERS)
    sample = [n for n, _ in evo.model.named_modules() if n.startswith("blocks.")][:8]
    log(f"layer probe ok；blocks.* 示例 {sample}；MASK_ID={MASK_ID}")

    eval_recs = pick_eval_records(n_hg=args.n_eval_hg)
    log(f"预计算 eval teacher（{len(eval_recs)} 窗，层 {ALL_INJECT_LAYERS}）")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    eval_packs = build_eval_packs(evo, eval_recs, args.window, device, ALL_INJECT_LAYERS)

    log("构建 Markov k=5 地板（训练前缀）")
    table = MarkovTable.build([r.seq for r in train_recs], k=5, alpha=DEFAULT_ALPHA)
    markov_model = MarkovDraftModel(table)
    markov_metrics = eval_markov(markov_model, eval_packs, args.gamma, args.cap_eval_anchors)
    log(f"Markov 地板 τ̂={markov_metrics.get('tau_hat')} ᾱ={markov_metrics.get('alpha_bar')}")

    if args.schemes:
        cells = [(s.strip(), 512 if args.grid else args.d_model) for s in args.schemes.split(",") if s.strip()]
        if args.grid:
            cells = [(s, 512) for s, _ in cells]
    elif args.grid:
        cells = list(GRID_D512)
    else:
        cells = [(args.scheme, args.d_model)]

    results = {
        "env": {
            "torch": torch.__version__,
            "gpu": props.name,
            "gpu_mem_gb": round(props.total_memory / 1e9, 1),
            "visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "hf_home": os.environ.get("HF_HOME"),
            "mask_id": MASK_ID,
            "target_hidden": TARGET_HIDDEN,
            "vocab": VOCAB_SIZE,
            "seed": args.seed,
            "gamma": args.gamma,
            "window": args.window,
            "n_anchors": args.n_anchors,
            "ctx_window": args.ctx_window,
            "max_positions": args.max_positions,
            "lr": args.lr,
            "pt_transform": {"temperature": 1.0, "top_k": 4},
            "step6_lacz_markov_tau": STEP6_TAU_LACZ,
        },
        "eval_names": [p.name for p in eval_packs],
        "markov_floor": markov_metrics,
        "cells": [],
        "recommendation": {},
    }
    planned = {(s, d) for s, d in cells}
    old_path = Path(args.results)
    if args.merge_results and old_path.exists():
        try:
            old = json.loads(old_path.read_text())
            for c in old.get("cells", []):
                key = (c.get("scheme"), c.get("d_model"))
                c.setdefault("ctx_window", 0)
                if key in planned:
                    if c.get("scheme") == "S1" and int(c.get("ctx_window") or 0) == 0:
                        c = dict(c)
                        c["scheme"] = "S1_fullctx"
                        results["cells"].append(c)
                    continue
                results["cells"].append(c)
            log(f"合并已有格子 {[ (c['scheme'], c['d_model'], c.get('ctx_window')) for c in results['cells'] ]}")
        except (OSError, json.JSONDecodeError) as exc:
            log(f"合并已有结果失败：{exc}")
    save_json(Path(args.results), results)

    for scheme, d_model in cells:
        cell = train_one_cell(
            evo=evo,
            train_recs=train_recs,
            eval_packs=eval_packs,
            markov_metrics=markov_metrics,
            scheme=scheme,
            d_model=d_model,
            gamma=args.gamma,
            window=args.window,
            n_anchors=args.n_anchors,
            min_ctx=args.min_ctx,
            max_positions=args.max_positions,
            lr=args.lr,
            eval_every=args.eval_every,
            seed=args.seed,
            device=device,
            ckpt_dir=Path(args.ckpt_dir),
            cap_eval_anchors=args.cap_eval_anchors,
            ctx_window=args.ctx_window,
            compile_drafter=args.compile,
            amp=args.amp,
        )
        results["cells"].append(cell)
        save_json(Path(args.results), results)

    if args.grid:
        d512 = [c for c in results["cells"] if c["d_model"] == 512]
        by_s = {c["scheme"]: c for c in d512}
        s0 = by_s.get("S0")
        s1 = by_s.get("S1")
        inj_ok = True
        if s0 and s1:
            t0 = float(s0["eval"]["tau_hat"])
            t1 = float(s1["eval"]["tau_hat"])
            inj_ok = t1 > t0 + 0.05
            if not inj_ok:
                log(f"S0 τ̂={t0:.3f} vs S1 τ̂={t1:.3f} 差过小，怀疑注入通路而非「注入无效」")
        ranked = sorted(
            [c for c in d512 if c["scheme"] in ("S1", "S2", "S3")],
            key=lambda c: float(c["eval"].get("tau_hat") or -1),
            reverse=True,
        )
        best = ranked[0] if ranked else (d512[-1] if d512 else None)
        results["recommendation"]["s0_vs_s1_injection_ok"] = inj_ok
        results["recommendation"]["best_injection"] = None if best is None else best["scheme"]
        if best is not None and not args.skip_width:
            for d in (1024, 2048):
                cell = train_one_cell(
                    evo=evo,
                    train_recs=train_recs,
                    eval_packs=eval_packs,
                    markov_metrics=markov_metrics,
                    scheme=best["scheme"],
                    d_model=d,
                    gamma=args.gamma,
                    window=args.window,
                    n_anchors=args.n_anchors,
                    min_ctx=args.min_ctx,
                    max_positions=args.max_positions,
                    lr=args.lr,
                    eval_every=args.eval_every,
                    seed=args.seed,
                    device=device,
                    ckpt_dir=Path(args.ckpt_dir),
                    cap_eval_anchors=args.cap_eval_anchors,
                    ctx_window=args.ctx_window,
                    compile_drafter=args.compile,
                    amp=args.amp,
                )
                results["cells"].append(cell)
                save_json(Path(args.results), results)
            width_cells = [c for c in results["cells"] if c["scheme"] == best["scheme"]]
            best_w = max(width_cells, key=lambda c: float(c["eval"].get("tau_hat") or -1))
            results["recommendation"]["best_d_model"] = best_w["d_model"]
            results["recommendation"]["best_tau_hat"] = best_w["eval"].get("tau_hat")
        elif best is not None:
            results["recommendation"]["best_d_model"] = best["d_model"]
            results["recommendation"]["best_tau_hat"] = best["eval"].get("tau_hat")

    tok_s = None
    if results["cells"]:
        tok_s = results["cells"][0]["throughput"].get("target_tok_s")
    results["formal_estimate"] = estimate_formal_hours(float(tok_s or 0.0))
    results["recommendation"]["markov_floor_tau_hat"] = markov_metrics.get("tau_hat")
    save_json(Path(args.results), results)
    log(f"已写入 {args.results}")
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase C 冒烟期在线蒸馏")
    p.add_argument("--scheme", default="S1", choices=sorted(SCHEME_LAYERS))
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--gamma", type=int, default=DEFAULT_GAMMA)
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--n-anchors", type=int, default=64)
    p.add_argument("--min-ctx", type=int, default=128)
    p.add_argument("--max-positions", type=int, default=500_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-every", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=20260821)
    p.add_argument("--n-eval-hg", type=int, default=16)
    p.add_argument("--cap-eval-anchors", type=int, default=256)
    p.add_argument("--grid", action="store_true", help="S1→S0→S2→S3（d512）再最优注入×d1024/2048")
    p.add_argument("--schemes", type=str, default=None, help="逗号分隔，覆盖默认格子，如 S1,S2,S3")
    p.add_argument("--merge-results", action="store_true", help="保留 results 里未重跑的格子")
    p.add_argument("--ctx-window", type=int, default=64, help="H_ctx 近锚点窗口；0=全前缀")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile 包装 drafter 训练前向（评估仍 eager；数值不逐位等）")
    p.add_argument("--amp", action="store_true",
                   help="drafter 训练前向 bf16 autocast（损失保持 fp32；参数/优化器 fp32）")
    p.add_argument("--skip-width", action="store_true")
    p.add_argument("--quick", action="store_true", help="2 万位置、单格 S1，管线打通")
    p.add_argument("--no-kernels", action="store_true")
    p.add_argument("--fasta-dir", type=str, default=None, help="预留；默认不扫 OpenGenome2")
    p.add_argument("--results", type=str, default=str(RESULTS_PATH))
    p.add_argument("--ckpt-dir", type=str, default=str(CKPT_DIR))
    args = p.parse_args(argv)
    if args.quick:
        args.max_positions = min(args.max_positions, 20_000)
        args.eval_every = min(args.eval_every, 10_000)
        args.grid = False
        args.scheme = "S1"
        args.d_model = 512
    return args


if __name__ == "__main__":
    run(parse_args())
