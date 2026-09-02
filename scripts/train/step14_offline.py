"""Step 14 离线训练器：从 dump_c1 分片练最终 drafter（plans/16 Stage C 的离线替代）。

与在线管线（train_one_cell）的关系：同一 drafter 结构/损失/超参（判定 scheme、d=1024、
γ=7、ctx_window=64、AdamW lr=3e-4 cosine、clip 1.0、--amp --compile），唯一差别是
数据来源——teacher 前向替换为 dump_c1 分片读取：

- ``p_t``：shard 存的是**原始 softmax** top-8 + 残差（fp16），本训练器重建稀疏分布后
  重新施加 top_k=4 截断归一（softmax 保序，top-4 概率精确等于在线口径；fp16 存储
  误差 ~1e-3 相对值，对拍门槛覆盖）。fp16 舍入并列时 top-4 边界选择可能与在线
  stable-sort 差一个等概率 token——概率质量相同、位置罕见，对拍按 L1 容差计。
- ``hidden``：int8 + per-token absmax scale（fp16）反量化，相对误差 ≲2%。
- 位置计数口径与在线一致（监督位置 = n_anchors×γ/步）。默认 n_anchors=256
  （4k 窗监督密度 44%，是在线 2k 窗的两倍——offline 无 teacher 成本，摊薄落盘 IO）。
  一整遍全量落盘（~264M 落盘位置）≈ 116M 监督位置；默认目标 150M ≈ 1.3 遍。

对拍（--max-positions 1000000 冒烟）三重门：
1. 重建对拍：取 shard 首窗跑 teacher_forward，top_k=4 p_t 的 L1 差、hidden 相对误差
   须在门槛内（fp16/int8 存储噪声级）；
2. 训练健全性：loss 首尾 20 步均值须下降、冻结 Δ=0、末次 eval τ̂ 落在合理带；
3. 全量前必须由 1M 冒烟绿灯（管线串联，失败即停）。

用法::

    CUDA_VISIBLE_DEVICES=1 python -u scripts/train/step14_offline.py \
      --max-positions 1000000 --results /tmp/step14_offline_smoke.json \
      --ckpt-name step14_offline_smoke.pt          # 1M 对拍
    CUDA_VISIBLE_DEVICES=0 python -u scripts/train/step14_offline.py \
      --results benchmarks/step14_offline.json      # 全量
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[2] / "hf_home"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import numpy as np
import torch

from train.c1_pack import arrays_crc, dequantize_hidden
from train.distill import (
    EvalPack,
    assert_layers_exist,
    build_eval_packs,
    eval_drafter,
    gather_batch,
    load_evo2,
    log,
    pick_eval_records,
    probe_max_abs_diff,
    probe_target,
    save_json,
    seed_all,
    teacher_forward,
    unpack_logits,
)
from train.drafter import (
    MASK_ID,
    SCHEME_LAYERS,
    Drafter,
    analytic_accept,
    apply_transform_torch,
    distill_losses,
)
from train.data import sample_anchors
from train.step11_retrain import build_holdout_eval_records
from train.step12_data import gtdb_holdout_eval_records, load_gtdb
from train.step12_retrain import load_pool_pin, parse_holdout_pin
from train.step15_mix import FINAL_MIX, build_epoch_plan, load_mix

REPO = Path(__file__).resolve().parents[2]
DUMP_DIR = REPO / "dump_c1"
PIN_PATH = REPO / "benchmarks" / "step13_pool_pin.json"
DECISION_PATH = REPO / "benchmarks" / "step13b_layer_rescan.json"
SHARD_KEYS = ("ids", "top8_idx", "top8_prob", "resid_mass", "hidden", "hidden_scale")

# 对拍门槛（fp16/int8 存储噪声 + 罕见等概率并列角的容差）
PARITY_PT_L1_MEAN_MAX = 3e-3
PARITY_PT_L1_FRAC_MAX = 1e-2  # L1>0.01 的位置占比上限
PARITY_HID_REL_MAX = 0.02
SMOKE_TAU_BAND = (2.3, 3.6)  # 1M 冒烟 τ̂ 合理带（在线 L27@3M=3.19 参考）


# ---------------------------------------------------------------------------
# 分片读取
# ---------------------------------------------------------------------------


def list_shards(dump_dir: Path, sources: tuple[str, ...] | None) -> list[dict]:
    """可读本 = meta.complete==True（meta 在 npz 写完之后落盘，故此时 npz 必然完整）。"""
    out = []
    for p in sorted(dump_dir.glob("*_*.json")):
        if p.name.startswith("manifest") or p.name == "qc_probe.json":
            continue
        try:
            meta = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not meta.get("complete"):
            continue
        if sources and meta.get("source") not in sources:
            continue
        out.append(meta)
    return out


def load_shard(meta: dict, dump_dir: Path, layers: tuple[str, ...]) -> dict:
    """读 shard 并校验 CRC 与层一致性。"""
    npz = dump_dir / meta["npz"]
    data = np.load(npz)
    crc = arrays_crc(*(data[k] for k in SHARD_KEYS))
    if crc != meta.get("crc32"):
        raise RuntimeError(f"{npz} CRC 不符：meta {meta.get('crc32')} 重算 {crc}")
    if tuple(meta.get("layers") or ()) != layers:
        raise RuntimeError(f"{npz} 层 {meta.get('layers')} ≠ 训练层 {layers}")
    return {k: data[k] for k in SHARD_KEYS}


def reconstruct_pt_torch(tidx: torch.Tensor, tval: torch.Tensor, vocab: int = 512) -> torch.Tensor:
    """top-8 (idx, prob) → top_k=4 截断归一的稀疏 p_t [N, vocab]（在线口径）。"""
    n = tidx.shape[0]
    pt = torch.zeros(n, vocab, dtype=torch.float32, device=tval.device)
    rows = torch.arange(n, device=tval.device)[:, None]
    pt[rows, tidx.long()] = tval.float()
    top4 = tval.float().argsort(dim=-1, descending=True, stable=True)[:, :4]
    keep = torch.zeros(n, vocab, dtype=torch.bool, device=tval.device)
    keep[rows, tidx.long().gather(1, top4)] = True
    pt = pt * keep
    return pt / pt.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def load_window_arrays(shard: dict, w: int, window: int, n_layers: int, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """shard 第 w 窗 → (ids[T], p_t[T,512] top_k=4, h_raw[T,4096·K])。"""
    s, e = w * window, (w + 1) * window
    ids = torch.as_tensor(shard["ids"][s:e].astype(np.int64), device=device)
    tidx = torch.as_tensor(shard["top8_idx"][s:e].astype(np.int64), device=device)
    tval = torch.as_tensor(shard["top8_prob"][s:e].astype(np.float32), device=device)
    pt = reconstruct_pt_torch(tidx, tval)
    hs = [dequantize_hidden(shard["hidden"][k, s:e], shard["hidden_scale"][k, s:e]) for k in range(n_layers)]
    h_raw = torch.as_tensor(np.concatenate(hs, axis=-1), dtype=torch.float32, device=device)
    return ids, pt, h_raw


# ---------------------------------------------------------------------------
# 对拍（重建精度）
# ---------------------------------------------------------------------------


@torch.no_grad()
def parity_check(
    evo,
    shard: dict,
    layers: tuple[str, ...],
    window: int,
    device,
    n_windows: int = 2,
    base_window: int | None = None,
) -> dict:
    """落盘重建 vs 在线前向的数值门槛判定。

    基线 = **与落盘同长度**的无状态 base_window（默认=window）前向；window <
    base_window（4k 落盘平切 2k 训练窗）时取该前向的窗首对齐 [0:window] 切片比较
    ——因果上等价、kernel 路径同长度，门槛沿用 Step 14 口径。直接对 2k 切片做
    无状态前向是不对的：奇数半窗缺前半上下文（2026-08-24 w2048 格实测 47.6%
    位置超阈）；即便偶数半窗，跨长度并行扫描的 bf16 分解差（实测 p_t L1≈3.6e-3）
    也会糊掉存储噪声级的门槛。训练监督用落盘真值，不受本检查口径影响。
    """
    base = int(base_window or window)
    stride = max(1, base // int(window))
    pt_l1_mean: list[float] = []
    pt_l1_frac = 0.0
    hid_rel: list[float] = []
    for i in range(n_windows):
        w = i * stride  # 对齐到 base 块首的 slice 窗下标
        blk_start = (w * int(window) // base) * base
        ids_blk = torch.as_tensor(
            shard["ids"][blk_start : blk_start + base].astype(np.int64), device=device
        )[None]
        logits, emb = teacher_forward(evo, ids_blk, layers)
        pt_on = apply_transform_torch(unpack_logits(logits)[0][: int(window)], temperature=1.0, top_k=4)
        h_on = torch.cat(
            [emb[ln][0][: int(window)] if emb[ln].ndim == 3 else emb[ln][: int(window)] for ln in layers],
            dim=-1,
        ).float()
        del logits, emb
        _ids, pt_off, h_off = load_window_arrays(shard, w, int(window), len(layers), device)
        l1 = (pt_off - pt_on).abs().sum(dim=-1)
        pt_l1_mean.append(float(l1.mean().item()))
        pt_l1_frac += float((l1 > 1e-2).float().mean().item())
        denom = h_on.abs().max().clamp_min(1e-6)
        hid_rel.append(float(((h_off - h_on).abs().max() / denom).item()))
    pt_l1_frac /= n_windows
    ok = (
        float(np.mean(pt_l1_mean)) <= PARITY_PT_L1_MEAN_MAX
        and pt_l1_frac <= PARITY_PT_L1_FRAC_MAX
        and float(np.mean(hid_rel)) <= PARITY_HID_REL_MAX
    )
    return {
        "pt_l1_mean": float(np.mean(pt_l1_mean)),
        "pt_l1_frac_gt_1e-2": pt_l1_frac,
        "hidden_rel_max": float(np.mean(hid_rel)),
        "ok": ok,
        "gates": {
            "pt_l1_mean<=": PARITY_PT_L1_MEAN_MAX,
            "pt_l1_frac<=": PARITY_PT_L1_FRAC_MAX,
            "hidden_rel<=": PARITY_HID_REL_MAX,
        },
    }


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------


def resolve_scheme(args) -> tuple[str, tuple[str, ...]]:
    if args.scheme:
        sch = args.scheme
    else:
        dec = json.loads(DECISION_PATH.read_text()).get("decision") or {}
        if dec.get("upgrade"):
            raise RuntimeError("复扫判定 upgrade=true，需人工裁决，拒绝自动训练")
        sch = dec.get("scheme") or "L27"
    if sch not in SCHEME_LAYERS or not SCHEME_LAYERS[sch]:
        raise RuntimeError(f"scheme={sch} 无注入层")
    return sch, SCHEME_LAYERS[sch]


def run(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA")
    # DDP（可选）：启动器注入 RANK/WORLD_SIZE/LOCAL_RANK，每 rank 独占一卡
    # （drafter-only 训练显存 ~3GB，24G 的 GPU1 也可参战）；单卡直跑全取默认
    # 0/1/0，行为与无 DDP 版一致。teacher/对拍/eval/存盘/落 json 仅 rank0。
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        import datetime

        import torch.distributed as dist

        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30))
    visible = torch.cuda.device_count()
    dev_idx = local_rank if visible > local_rank else 0
    device = torch.device(f"cuda:{dev_idx}")
    torch.cuda.set_device(device)
    scheme, layers = resolve_scheme(args)
    sources = tuple(s.strip() for s in args.sources.split(",") if s.strip()) if args.sources else None
    dump_dir = Path(args.dump_dir)
    metas = list_shards(dump_dir, sources)
    if not metas:
        raise RuntimeError(f"{dump_dir} 没有可读 shard（sources={sources}）")
    total_dumped = sum(int(m["n_pos"]) for m in metas)
    mix_weights = load_mix(args.mix)
    if mix_weights is not None and rank == 0:
        on_disk = {str(m["source"]) for m in metas}
        zeroed = sorted(on_disk - set(mix_weights))
        absent = sorted(set(mix_weights) - on_disk)
        if zeroed:
            log(f"警告：源 {zeroed} 在配比表中权重为 0，本run 不消费其 shard")
        if absent:
            log(f"警告：配比表含盘上不存在的源 {absent}（权重按剩余源归一）")
        log(f"配比采样 mix={args.mix} cap_epochs={args.max_source_epochs} weights={mix_weights}")
    log(
        f"离线训练 scheme={scheme} layers={layers} shards={len(metas)} "
        f"落盘位置={total_dumped:,} 目标监督位置={args.max_positions:,} "
        f"world={world} rank={rank}"
    )
    # 每 rank 交错分片子集并逐 epoch 循环使用：各 rank 永远有窗可载，
    # 全局步数对称（每步 n_anchors×γ×world），allreduce 不会因步数分叉挂死。
    # epoch 计划按 --mix 加权（uniform=全量一遍，Step 14 口径）。
    my_metas = [metas[i] for i in range(len(metas)) if i % world == rank]
    visit_pos: dict[str, int] = {}
    first_plan_info = None

    seed_all(args.seed)
    evo = load_evo2(use_kernels=not args.no_kernels)
    assert_layers_exist(evo, layers)

    # eval packs / 重建对拍仅 rank0（各 rank 都载 teacher 仅为取冻结 embedding；
    # rank≠0 不消费 packs）。gtdb 留出用 Step 13 pin（DPLL01/JARLHP + chunk6/7）
    eval_packs: list[EvalPack] = []
    parity = None
    if rank == 0:
        pin = load_pool_pin(PIN_PATH)
        gtdb_meta = load_gtdb(
            seed=args.seed,
            only=set(pin["chunks"]["gtdb_v220_imgpr"]),
            holdout_pin=parse_holdout_pin(",".join(pin["gtdb_holdout_pin"])),
        )[1]
        eval_recs = (
            list(pick_eval_records(n_hg=args.n_eval_hg))
            + list(build_holdout_eval_records())
            + list(gtdb_holdout_eval_records(gtdb_meta))
        )
        log(f"构建 eval packs（{len(eval_recs)} 窗，层 {layers}）…")
        eval_packs = build_eval_packs(evo, eval_recs, args.eval_window, device, layers)

        if not args.skip_parity:
            first = load_shard(metas[0], dump_dir, layers)
            base_window = None
            try:
                _dm = json.loads((dump_dir / "manifest.json").read_text())
                base_window = int(_dm.get("window") or 0) or None
            except (OSError, json.JSONDecodeError, ValueError):
                pass
            if base_window is not None and args.window > base_window:
                raise RuntimeError(f"--window {args.window} 大于落盘窗 {base_window}，无法平切")
            parity = parity_check(evo, first, layers, args.window, device, base_window=base_window)
            log(f"重建对拍(base_window={base_window}): {parity}")
            if not parity["ok"]:
                raise RuntimeError(f"重建对拍未过门槛: {parity}")
            del first

    drafter = Drafter.from_scheme(scheme, d_model=args.d_model, gamma=args.gamma, ctx_window=args.ctx_window)
    w_emb = evo.model.embedding_layer.weight.detach().float().cpu()
    drafter.load_frozen_embedding(w_emb)
    drafter.to(device)
    ddp = None
    if world > 1:
        ddp = torch.nn.parallel.DistributedDataParallel(
            drafter,
            device_ids=[torch.cuda.current_device()],
            find_unused_parameters=False,  # 固定图（同在线 DDP 冒烟验证）
            broadcast_buffers=False,
        )
    fwd_model = ddp if ddp is not None else drafter
    if args.compile:
        fwd_model = torch.compile(fwd_model)
    frozen_before = {k: v.detach().cpu().clone() for k, v in drafter.frozen_named_tensors().items()}
    target_before = probe_target(evo)
    if not args.keep_teacher:
        # teacher 使命完成（eval packs / 对拍 / 冻结快照）——释放 13.5GB，
        # 否则 24G 卡上 compile autotune 工作区都塞不下（冒烟实测 OOM 踩过）。
        # 释放后 target 探针末检结构上无意义（无写路径），记 "freed" 而非数值。
        del evo
        torch.cuda.empty_cache()
        log("teacher 已释放（eval packs/对拍/冻结快照均在手），进入纯 drafter 训练")
    opt = torch.optim.AdamW(drafter.trainable_parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))
    pos_per_step = args.n_anchors * args.gamma * world  # 全局监督位置/步（lr 随位置轨迹不变）
    n_steps_est = max(1, int(np.ceil(args.max_positions / pos_per_step)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps_est, eta_min=args.lr * 0.1)

    rng = np.random.default_rng(args.seed + 10_000 * rank)  # 各 rank 采样流错开
    curve: list[dict] = []
    evals: list[dict] = []
    n_pos = 0
    step = 0
    t_dr_sum = 0.0
    n_timed = 0
    t0_all = time.perf_counter()
    last_eval = None
    done = False
    epoch = 0
    t_load = 0.0

    with ThreadPoolExecutor(max_workers=1) as ex:
        while not done:
            epoch += 1
            plan, plan_info = build_epoch_plan(
                my_metas, mix_weights, args.window, rng, cap_epochs=args.max_source_epochs
            )
            if not plan:
                raise RuntimeError("epoch 计划为空（shard 均不足一个窗？）——拒绝空转")
            if first_plan_info is None:
                first_plan_info = plan_info
                if rank == 0:
                    log(f"epoch 计划（首个）: {plan_info}")
            fut = ex.submit(load_shard, my_metas[plan[0][0]], dump_dir, layers) if plan else None
            for j, (mi, wins) in enumerate(plan):
                if done:
                    break
                tw0 = time.perf_counter()
                shard = fut.result()
                if j + 1 < len(plan):
                    fut = ex.submit(load_shard, my_metas[plan[j + 1][0]], dump_dir, layers)
                t_load += time.perf_counter() - tw0
                src = str(my_metas[mi].get("source", "?"))
                visit_pos[src] = visit_pos.get(src, 0) + len(wins) * args.window
                for wi in wins:
                    if n_pos >= args.max_positions:
                        done = True
                        break
                    ids, pt_full, h_raw = load_window_arrays(shard, int(wi), args.window, len(layers), device)
                    anchors_np = sample_anchors(args.window, args.gamma, args.n_anchors, rng, min_ctx=args.min_ctx)
                    if anchors_np.size == 0:
                        continue
                    a_t = torch.as_tensor(anchors_np, dtype=torch.long, device=device)
                    anchor_ids, prev, target, pt, ctx_lens = gather_batch(ids, pt_full, a_t, args.gamma)
                    pt = pt.detach()

                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    opt.zero_grad(set_to_none=True)
                    amp_ctx = (
                        torch.autocast("cuda", dtype=torch.bfloat16)
                        if args.amp and device.type == "cuda"
                        else contextlib.nullcontext()
                    )
                    with amp_ctx:
                        logits, probs, conf, _ = fwd_model(anchor_ids, prev, ctx_lens, h_raw=h_raw)
                    br = distill_losses(logits, pt, target, conf, p_d=probs)
                    br.total.backward()
                    torch.nn.utils.clip_grad_norm_(drafter.trainable_parameters(), 1.0)
                    opt.step()
                    sched.step()
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_dr_sum += time.perf_counter() - t1
                    n_timed += 1

                    n_this = int(anchors_np.size) * args.gamma * world  # 全局监督位置
                    n_pos += n_this
                    step += 1
                    with torch.no_grad():
                        a_mean = float(analytic_accept(probs, pt).mean().item())
                    if step <= 3 or step % 20 == 0:
                        rec = {
                            "step": step, "n_pos": n_pos, "loss": float(br.total.item()),
                            "ce": float(br.ce.item()), "tv": float(br.tv.item()),
                            "conf": float(br.conf.item()), "train_cstar": a_mean,
                            "lr": float(sched.get_last_lr()[0]),
                        }
                        curve.append(rec)
                        log(f"step={step} pos={n_pos} loss={rec['loss']:.4f} c*={a_mean:.3f} "
                            f"draft={t_dr_sum / max(n_timed, 1):.3f}s")
                    do_eval = (n_pos >= args.max_positions) or (
                        args.eval_every > 0
                        and n_pos // args.eval_every > (n_pos - n_this) // args.eval_every
                        and n_pos >= args.eval_every
                    )
                    if do_eval and rank == 0:
                        last_eval = eval_drafter(drafter, eval_packs, scheme, args.gamma, device, args.cap_eval_anchors)
                        last_eval = {**last_eval, "n_pos": n_pos}
                        evals.append(last_eval)
                        log(f"eval pos={n_pos} τ̂={last_eval.get('tau_hat'):.4f}")
                    if (
                        rank == 0
                        and args.save_every > 0
                        and n_pos // args.save_every > (n_pos - n_this) // args.save_every
                    ):
                        save_ckpt(drafter, args, scheme, n_pos)
                del shard
    if world > 1:
        import torch.distributed as dist

        dist.barrier()
        dist.destroy_process_group()
    if rank != 0:
        return {"status": "done", "rank": rank, "note": "非 rank0 无落盘职责"}
    if last_eval is None:
        last_eval = eval_drafter(drafter, eval_packs, scheme, args.gamma, device, args.cap_eval_anchors)
        last_eval = {**last_eval, "n_pos": n_pos}
        evals.append(last_eval)

    ckpt_path = save_ckpt(drafter, args, scheme, n_pos, final=True)
    frozen_after = {k: v.detach().cpu() for k, v in drafter.frozen_named_tensors().items()}
    emb_delta = float((frozen_before["embed.weight"] - frozen_after["embed.weight"]).abs().max().item())
    if args.keep_teacher:
        target_delta = probe_max_abs_diff(target_before, probe_target(evo))
        target_note: str | float = target_delta
    else:
        target_delta = 0.0  # teacher 已释放，无写路径（结构上不可能被训练）
        target_note = "teacher_freed_structural"

    sanity = {"loss_first20": None, "loss_last20": None, "loss_dropped": None}
    if len(curve) >= 40:
        losses = [r["loss"] for r in curve]
        sanity = {
            "loss_first20": float(np.mean(losses[:20])),
            "loss_last20": float(np.mean(losses[-20:])),
            "loss_dropped": bool(np.mean(losses[-20:]) < np.mean(losses[:20])),
        }
    tau = float(last_eval.get("tau_hat") or float("nan"))
    sanity["tau_hat_final"] = tau
    sanity["tau_in_band"] = bool(SMOKE_TAU_BAND[0] <= tau <= SMOKE_TAU_BAND[1])
    sanity["freeze_ok"] = bool(emb_delta == 0.0 and target_delta == 0.0)
    sanity["ok"] = bool(sanity["freeze_ok"] and sanity["tau_in_band"] and (sanity["loss_dropped"] in (True, None)))

    out = {
        "tag": args.tag,
        "scheme": scheme,
        "layers": list(layers),
        "d_model": args.d_model,
        "gamma": args.gamma,
        "ctx_window": args.ctx_window,
        "n_anchors": args.n_anchors,
        "n_positions": n_pos,
        "epochs": epoch,
        "seed": args.seed,
        "lr": args.lr,
        "amp": args.amp,
        "compile": args.compile,
        "dump_dir": str(dump_dir),
        "dump_positions_available": total_dumped,
        "mix": {
            "spec": args.mix,
            "weights": mix_weights,
            "max_source_epochs": args.max_source_epochs,
            "first_epoch_plan": first_plan_info,
            "visit_positions_by_source": visit_pos,
            "note": None if mix_weights is None else (
                "visit = 本 rank 实际访问窗×窗长（不含 world 倍增）；盘面构成见 per_source"
            ),
        },
        "parity": parity,
        "freeze": {"embed_max_abs_delta": emb_delta, "target_max_abs_delta": target_note},
        "throughput": {
            "positions_per_s": n_pos / max(t_dr_sum, 1e-9),
            "draft_s_per_step": t_dr_sum / max(n_timed, 1),
            "load_s_total": t_load,
        },
        "train_curve": curve,
        "evals": evals,
        "eval": last_eval,
        "sanity": sanity,
        "ckpt": str(ckpt_path),
        "elapsed_s": time.perf_counter() - t0_all,
        "status": "done",
    }
    save_json(Path(args.results), out)
    log(
        f"完成 {scheme} 离线 τ̂={tau:.4f} pos={n_pos:,} sanity={sanity['ok']} "
        f"{out['throughput']['positions_per_s']:.0f} pos/s（监督） 已写入 {args.results}"
    )
    return out


def save_ckpt(drafter, args, scheme: str, n_pos: int, final: bool = False) -> Path:
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    name = args.ckpt_name or f"{args.tag}.pt"
    if not final and name.endswith(".pt"):
        name = name[:-3] + f"_pos{n_pos}.pt"
    path = ckpt_dir / name
    torch.save(
        {
            "scheme": scheme,
            "d_model": args.d_model,
            "gamma": args.gamma,
            "ctx_window": args.ctx_window,
            "state_dict": drafter.state_dict(),
            "n_pos": n_pos,
            "mask_id": MASK_ID,
            "offline": True,
            "dump_dir": args.dump_dir,
        },
        path,
    )
    log(f"ckpt 落盘 {path}（pos={n_pos:,}）")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scheme", default=None, help="默认读 benchmarks/step13b_layer_rescan.json 判定")
    p.add_argument("--d-model", type=int, default=1024)
    p.add_argument("--gamma", type=int, default=7)
    p.add_argument("--ctx-window", type=int, default=64)
    p.add_argument("--n-anchors", type=int, default=256)
    p.add_argument("--min-ctx", type=int, default=128)
    p.add_argument("--max-positions", type=int, default=150_000_000, help="监督位置（1M=对拍冒烟）")
    p.add_argument("--window", type=int, default=4096, help="须与落盘窗一致")
    p.add_argument("--eval-window", type=int, default=2048, help="eval packs 窗长（Step 8–13 协议）")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-every", type=int, default=5_000_000)
    p.add_argument("--save-every", type=int, default=50_000_000)
    p.add_argument("--cap-eval-anchors", type=int, default=256)
    p.add_argument("--n-eval-hg", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260821)
    p.add_argument("--sources", type=str, default=None, help="逗号分隔源子集；默认全部可读 shard")
    p.add_argument("--mix", type=str, default="uniform",
                   help="源采样配比：uniform=盘面均匀（Step 14 口径）；final15=终版配比"
                        "（train.step15_mix.FINAL_MIX）；或 JSON 字符串/文件路径")
    p.add_argument("--max-source-epochs", type=float, default=2.0,
                   help="单源每 epoch 访问窗数上限（×自有窗数）；防小池静默过采样")
    p.add_argument("--dump-dir", type=str, default=str(DUMP_DIR))
    p.add_argument("--results", type=str, default="benchmarks/step14_offline.json")
    p.add_argument("--ckpt-dir", type=str, default=str(REPO / "benchmarks" / "phasec_ckpts"))
    p.add_argument("--ckpt-name", type=str, default=None)
    p.add_argument("--tag", type=str, default="L27_d1024_g7_offline_step14")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--no-kernels", action="store_true")
    p.add_argument("--keep-teacher", action="store_true",
                   help="训练期间常驻 teacher（默认释放：对拍/eval packs 后无用，24G 卡必须释放）")
    p.add_argument("--skip-parity", action="store_true")
    args = p.parse_args(argv)
    args.amp = not args.no_amp
    args.compile = not args.no_compile
    return args


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
