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

from specdec.block.chunk import _CHUNK_FLAG, _STASH_ATTR, install_chunk_forward

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
    返回后 offset 置为 L。返回 [1, L, V]，末位即 p(·|prefix)。"""
    assert all(inference_params_dict[k].seqlen_offset == 0 for k in PARAM_KEYS), (
        "prefill 要求 seqlen_offset 全为 0"
    )
    logits, _ = model(input_ids, inference_params_dict=inference_params_dict)
    set_seqlen_offsets(inference_params_dict, input_ids.shape[1])
    return logits


def block_forward(
    model,
    chunk_ids: torch.Tensor,
    inference_params_dict: dict,
    retain: bool = False,
):
    """带初始状态的块验证前向。

    ``chunk_ids``: [1, γ]；入口时四键 offset 须等于前缀长度 L。
    返回 [1, γ, V]：第 i 位 = p(·|prefix, x_1..x_{i+1})。
    返回后四键 offset 同步 += γ。

    ``retain=False``（默认）：行为与 Step 4 逐位一致，返回 logits。
    ``retain=True``：额外保留逐位置中间量，返回 ``(logits, ChunkStash)``；
    未切片前的末态写回与 ``retain=False`` 相同。
    """
    install_chunk_forward()
    stash = None
    if retain:
        from specdec.block.slice import ChunkStash

        stash = ChunkStash(
            L0=get_seqlen_offset(inference_params_dict),
            chunk_len=int(chunk_ids.shape[1]),
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
    if retain:
        return logits, stash
    return logits


def step_forward_reference(model, chunk_ids: torch.Tensor, inference_params_dict: dict) -> torch.Tensor:
    """原生逐 token 参照路径（sequential/step；offset 语义与块前向一致）。

    返回 [1, γ, V]，第 i 位 = 喂入 x_{i+1} 后的末位 logits，
    与 :func:`block_forward` 逐位对应。
    """
    outs = []
    for i in range(chunk_ids.shape[1]):
        logits, _ = model(chunk_ids[:, i : i + 1], inference_params_dict=inference_params_dict)
        outs.append(logits[:, -1])
        step_seqlen_offsets(inference_params_dict, 1)
    return torch.stack(outs, dim=1)


def snapshot_states(inference_params_dict: dict, kv_len: int | None = None) -> dict:
    """把当前推理状态收集为 CPU 张量树。

    KV cache 只取前 ``kv_len`` 个已写位置（之后是 ``torch.empty`` 未初始化垃圾）。
    """
    snap: dict = {
        "seqlen_offsets": {k: int(inference_params_dict[k].seqlen_offset) for k in PARAM_KEYS}
    }
    mha = inference_params_dict["mha"]
    snap["mha"] = {
        "key_value_memory_dict": {
            int(layer): (t[:, :kv_len] if kv_len is not None else t).detach().cpu()
            for layer, t in mha.key_value_memory_dict.items()
        }
    }
    for key in _HYENA_KEYS:
        params = inference_params_dict[key]
        entry = {}
        for attr in ("fir_state_dict", "fir_inner_state_dict", "state_dict"):
            d = getattr(params, attr, None)
            if d:
                entry[attr] = {int(layer): t.detach().cpu() for layer, t in d.items()}
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
