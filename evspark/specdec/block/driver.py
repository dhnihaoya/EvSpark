"""块验证前向驱动 + 参照步进 + 状态快照/比对（Plan 02 A.2 / Step 4 对拍工具）。

语义红线（off-by-one，plans/06 §3.1）：prefill 后缓存状态对应长度 L 的前缀。
``block_forward`` 输入 γ 个 token ``x_1..x_γ``，输出 logits [1, γ, V] 的第 i 位
（0 基）= ``p(· | prefix, x_1..x_{i+1})``，即验证器契约里的 **p₂..p_{γ+1}**
（p_{γ+1} 是 bonus 位）。**p₁ = p(·|prefix) 不在块输出里**，它来自 prefill /
上一轮验证的末位 logits，由投机循环拼接成 [γ+1, V]。

``seqlen_offset`` 簿记约定：每次前向**入口**时 offset = 已缓存 token 数
（= 本次写入起点），前向完成后按实际消耗 token 数步进；mha/hcl/hcm/hcs 四键同步。
"""

from __future__ import annotations

import copy

import torch

from evspark.specdec.block.chunk import _CHUNK_FLAG, _STASH_ATTR, install_chunk_forward

#: 推理参数四键（簿记须同步，notes/vortex_state_flow.md §6.2）
PARAM_KEYS = ("mha", "hcl", "hcm", "hcs")
_HYENA_KEYS = ("hcl", "hcm", "hcs")


def set_seqlen_offsets(inference_params_dict: dict, offset: int) -> None:
    for key in PARAM_KEYS:
        inference_params_dict[key].seqlen_offset = offset


def step_seqlen_offsets(inference_params_dict: dict, n: int = 1) -> None:
    for key in PARAM_KEYS:
        inference_params_dict[key].seqlen_offset += n


def get_seqlen_offset(inference_params_dict: dict) -> int:
    return int(inference_params_dict["mha"].seqlen_offset)


def clone_inference_params(inference_params_dict: dict) -> dict:
    """全量深拷贝快照（KV cache 与张量状态一并 clone）。

    Hyena 全状态 <10MB/序列，1k 上下文 KV ~84MB，深拷贝代价可接受；
    这正是 notes §5 的 fallback 快照基线。
    """
    return copy.deepcopy(inference_params_dict)


def prefill(model, input_ids: torch.Tensor, inference_params_dict: dict) -> torch.Tensor:
    """并行预填（vortex 原生 parallel 路径）。要求四键 offset 均为 0；
    返回后 offset 置为 L。返回 [B, L, V]，末位即 p(·|prefix)。"""
    assert all(inference_params_dict[k].seqlen_offset == 0 for k in PARAM_KEYS), (
        "prefill 要求 seqlen_offset 全为 0"
    )
    logits, _ = model(input_ids, inference_params_dict=inference_params_dict)
    L = int(input_ids.shape[1])
    B = int(input_ids.shape[0])
    set_seqlen_offsets(inference_params_dict, L)
    _bump_lengths(inference_params_dict, B, L, replace=True)
    return logits


def _bump_lengths(inference_params_dict: dict, batch: int, n: int, *, replace: bool) -> None:
    """同步 ``mha.lengths_per_sample[:batch]``（未分配则跳过）。

    ``replace=True``：写成 ``n``（prefill）；否则 ``+= n``（块前向）。
    flash_attn 快路径用它做 per-sample ``cache_seqlens``；SDPA 慢路径忽略。
    """
    mha = inference_params_dict["mha"]
    lp = getattr(mha, "lengths_per_sample", None)
    if lp is None:
        return
    b = int(batch)
    if replace:
        lp[:b] = int(n)
    else:
        lp[:b] += int(n)


def _flash_kvcache_enabled(model) -> bool:
    """flash_attn kvcache 快路径是否真实生效（config 开 + 内核可 import）。

    慢路径（SDPA）只认标量 seqlen_offset，变长切片会静默错 KV；
    写进 ChunkStash 供 slice 拒绝非一致 k。
    """
    config = getattr(model, "config", None)
    if config is None or not bool(config.get("use_flash_attn", False)):
        return False
    try:
        from vortex.model.attention import local_flash_attn_with_kvcache
    except ImportError:
        return False
    return local_flash_attn_with_kvcache is not None


def expand_inference_params(inference_params_dict: dict, batch: int) -> None:
    """把 B=1 的 Hyena / KV 前缀复制成 ``batch`` 份（多候选块验证用）。

    要求 ``mha.max_batch_size >= batch``（KV 在首次 MHA 前向按该值预分配）。
    ``batch==1`` 为 no-op。
    """
    B = int(batch)
    if B < 1:
        raise ValueError(f"batch 须 ≥ 1，得到 {B}")
    if B == 1:
        return
    mha = inference_params_dict["mha"]
    if int(mha.max_batch_size) < B:
        raise ValueError(
            f"mha.max_batch_size={mha.max_batch_size} < B={B}："
            "须在 initialize_inference_params 前把 model.config.max_batch_size 设到 ≥ B"
        )
    L = int(mha.seqlen_offset)
    for cache in mha.key_value_memory_dict.values():
        if int(cache.shape[0]) < B:
            raise ValueError(f"KV cache batch={cache.shape[0]} < B={B}")
        cache[1:B, :L].copy_(cache[0, :L])
    device = None
    if mha.key_value_memory_dict:
        device = next(iter(mha.key_value_memory_dict.values())).device
    if mha.lengths_per_sample is None:
        if device is None:
            raise RuntimeError("expand 需要 KV cache 以确定 lengths_per_sample 的 device")
        mha.lengths_per_sample = torch.zeros(
            int(mha.max_batch_size), dtype=torch.int32, device=device
        )
    mha.lengths_per_sample[:B] = L

    for key in _HYENA_KEYS:
        params = inference_params_dict[key]
        for attr in ("fir_state_dict", "fir_inner_state_dict", "state_dict"):
            d = getattr(params, attr, None)
            if not d:
                continue
            for layer_idx, tensor in list(d.items()):
                if int(tensor.shape[0]) == B:
                    continue
                if int(tensor.shape[0]) != 1:
                    raise ValueError(
                        f"{key}.{attr}[{layer_idx}] batch={tensor.shape[0]}，无法扩到 {B}"
                    )
                d[layer_idx] = tensor.expand(B, *tensor.shape[1:]).contiguous()


def collapse_inference_params(inference_params_dict: dict, idx: int) -> None:
    """多候选块验证后收回 B=1：把 ``idx`` 的 Hyena / KV 前缀写到 batch 0。

    Hyena 张量收成 ``[1, ...]``。KV 只拷胜出路径已消费前缀。
    变长切片后 ``seqlen_offset`` 可能停在 ``max(k)``，此处收到
    ``lengths_per_sample[idx]``（否则下一轮会按过长前缀读）。
    收回 B=1 后 ``lengths_per_sample`` 置 None 回标量 offset 模式：
    vortex rotary 要求该张量形状恰好 ``(batch,)``（attention.py 全量传入），
    B=1 前向遇到 ``[max_batch_size]`` 会断言；下轮 expand 会重新分配。
    """
    idx = int(idx)
    mha = inference_params_dict["mha"]
    L = int(mha.seqlen_offset)
    if mha.lengths_per_sample is not None:
        win_L = int(mha.lengths_per_sample[idx].item())
        if win_L > 0:
            L = win_L
    if idx != 0:
        for cache in mha.key_value_memory_dict.values():
            cache[0, :L].copy_(cache[idx, :L])
    mha.lengths_per_sample = None
    for key in _HYENA_KEYS:
        params = inference_params_dict[key]
        for attr in ("fir_state_dict", "fir_inner_state_dict", "state_dict"):
            d = getattr(params, attr, None)
            if not d:
                continue
            for layer_idx, tensor in list(d.items()):
                if int(tensor.shape[0]) == 1:
                    continue
                d[layer_idx] = tensor[idx : idx + 1].contiguous()
    set_seqlen_offsets(inference_params_dict, L)


def block_forward(
    model,
    chunk_ids: torch.Tensor,
    inference_params_dict: dict,
    retain: bool = False,
):
    """带初始状态的块验证前向。

    ``chunk_ids``: [B, γ]；入口时四键 offset 须等于前缀长度 L（同长 batch）。
    返回 [B, γ, V]：第 i 位 = p(·|prefix, x_1..x_{i+1})。
    返回后四键 offset 同步 += γ；若已分配 ``lengths_per_sample`` 则 ``[:B] += γ``。

    ``retain=False``（默认）：行为与 Step 4 逐位一致，返回 logits。
    ``retain=True``：额外保留逐位置中间量，返回 ``(logits, ChunkStash)``；
    未切片前的末态写回与 ``retain=False`` 相同。
    """
    install_chunk_forward()
    stash = None
    if retain:
        from evspark.specdec.block.slice import ChunkStash

        stash = ChunkStash(
            L0=get_seqlen_offset(inference_params_dict),
            chunk_len=int(chunk_ids.shape[1]),
            flash_attn=_flash_kvcache_enabled(model),
            lengths_before=(None if inference_params_dict['mha'].lengths_per_sample is None
                            else inference_params_dict['mha'].lengths_per_sample[:chunk_ids.shape[0]].clone()),
        )
    for key in _HYENA_KEYS:
        setattr(inference_params_dict[key], _CHUNK_FLAG, True)
        if stash is not None:
            setattr(inference_params_dict[key], _STASH_ATTR, stash)
    try:
        logits, _ = model(chunk_ids, inference_params_dict=inference_params_dict)
    finally:
        for key in _HYENA_KEYS:
            params = inference_params_dict[key]
            if hasattr(params, _CHUNK_FLAG):
                delattr(params, _CHUNK_FLAG)
            if hasattr(params, _STASH_ATTR):
                delattr(params, _STASH_ATTR)
    step_seqlen_offsets(inference_params_dict, chunk_ids.shape[1])
    _bump_lengths(
        inference_params_dict, int(chunk_ids.shape[0]), int(chunk_ids.shape[1]), replace=False
    )
    if retain:
        return logits, stash
    return logits


def step_forward_reference(model, chunk_ids: torch.Tensor, inference_params_dict: dict) -> torch.Tensor:
    """原生逐 token 参照路径（sequential/step；offset 语义与块前向一致）。

    返回 [B, γ, V]，第 i 位 = 喂入 x_{i+1} 后的末位 logits，
    与 :func:`block_forward` 逐位对应。
    """
    outs = []
    for i in range(chunk_ids.shape[1]):
        logits, _ = model(chunk_ids[:, i : i + 1], inference_params_dict=inference_params_dict)
        outs.append(logits[:, -1])
        step_seqlen_offsets(inference_params_dict, 1)
    return torch.stack(outs, dim=1)


def snapshot_states(
    inference_params_dict: dict,
    kv_len: int | None = None,
    *,
    batch_size: int = 1,
) -> dict:
    """把当前推理状态收集为 CPU 张量树。

    KV cache 只取前 ``kv_len`` 个已写位置（之后是 ``torch.empty`` 未初始化垃圾），
    且只取 ``[:batch_size]``（默认 1：多候选 expand 后的闲置槽不进对拍）。
    """
    bsz = int(batch_size)
    snap: dict = {
        "seqlen_offsets": {k: int(inference_params_dict[k].seqlen_offset) for k in PARAM_KEYS}
    }
    mha = inference_params_dict["mha"]
    snap["mha"] = {
        "key_value_memory_dict": {
            int(layer): (
                (t[:bsz, :kv_len] if kv_len is not None else t[:bsz]).detach().cpu()
            )
            for layer, t in mha.key_value_memory_dict.items()
        }
    }
    for key in _HYENA_KEYS:
        params = inference_params_dict[key]
        entry = {}
        for attr in ("fir_state_dict", "fir_inner_state_dict", "state_dict"):
            d = getattr(params, attr, None)
            if d:
                entry[attr] = {
                    int(layer): (t[:bsz] if t.shape[0] > bsz else t).detach().cpu()
                    for layer, t in d.items()
                }
        snap[key] = entry
    return snap


def diff_states(a: dict, b: dict) -> dict:
    """逐状态张量比对：max 绝对差 / max 相对差（分母为该张量 |b| 最大值，clamp 1e-12）。

    ``a``/``b`` 为 :func:`snapshot_states` 产物。返回 ``{路径: {max_abs, max_rel}}``。
    键集合不一致会抛错（结构性分歧必须显式暴露）。
    """
    if a["seqlen_offsets"] != b["seqlen_offsets"]:
        raise ValueError(f"seqlen_offsets 不一致: {a['seqlen_offsets']} vs {b['seqlen_offsets']}")
    out: dict = {}

    def _visit(va, vb, path):
        if isinstance(va, dict):
            if set(va.keys()) != set(vb.keys()):
                raise ValueError(f"状态键不一致 @{path}: {sorted(va.keys())} vs {sorted(vb.keys())}")
            for k in va:
                _visit(va[k], vb[k], f"{path}/{k}")
        else:
            d = (va.float() - vb.float()).abs()
            max_abs = float(d.max()) if d.numel() else 0.0
            denom = float(vb.float().abs().max()) if vb.numel() else 0.0
            out[path] = {
                "max_abs": max_abs,
                "max_rel": max_abs / max(denom, 1e-12),
            }

    for key in PARAM_KEYS:
        _visit(a[key], b[key], key)
    return out
