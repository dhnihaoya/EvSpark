"""状态切片回滚（Plan 02 A.3 / Step 7）：用块前向暂存的逐位置中间量切到接受位。

循环块输入 ``[y, x₁..x_γ]``（下标 0..γ）。接受 ``x₁..x_k`` 后目标状态 =
**消费到块内下标 k**（即 x_k）；新锚点 y′ 不消费。切片位置 ``j* = k``。
截断轮：提交前缀长度为 ``take``，等价 ``j* = take - 1``。

逐层规则（``cat`` = ``[缓存窗口 | 块输入]``，K = 滤波器长）：

- 外层短 FIR（K=3）：``cat_u[..., k+1 : k+K]``
- HCS 内层（K=7）/ HCM 内层（K=128）：``x1v_cat[..., k+1 : k+K]``
- HCL IIR：``s_all[..., k]``（块前向已算出的逐位置状态，fp32）
- MHA KV：零数据移动，四键 ``seqlen_offset = L0 + k + 1``

k=0：只消费锚点 y → IIR 取 ``s_all[..., 0]``（消费 y **之后**，不是轮初 s0）。
k=γ：切出末态，与块前向已写入的 ``[..., -(K-1):]`` / ``s_all[..., -1]`` 一致；
循环在全接受且未截断时跳过本函数（零开销）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from evspark.specdec.block.driver import PARAM_KEYS, set_seqlen_offsets

_HYENA_KEYS = ("hcl", "hcm", "hcs")


@dataclass
class ChunkStash:
    """一块前向的保留中间量。每轮覆盖，不累积。

    下标：``chunk_len`` = 块输入长度（循环里 γ+1）；位置 0 = 锚点 y。
    ``s_all[..., j]`` 与 FIR 窗 ``[..., j+1 : j+K]`` 对应消费到下标 j 之后。
    """

    L0: int
    chunk_len: int
    cat_u: dict[int, torch.Tensor] = field(default_factory=dict)
    k_outer: dict[int, int] = field(default_factory=dict)
    x1v_cat: dict[int, torch.Tensor] = field(default_factory=dict)
    k_inner: dict[int, int] = field(default_factory=dict)
    s_all: dict[int, torch.Tensor] = field(default_factory=dict)


def _hyena_params(ip_dict: dict, layer_idx: int):
    for key in _HYENA_KEYS:
        if layer_idx in ip_dict[key].fir_state_dict:
            return ip_dict[key]
    raise KeyError(f"layer_idx {layer_idx} 不在任何 Hyena 参数组的 fir_state_dict")


def slice_states_to_accept(ip_dict: dict, stash: ChunkStash, k: int) -> None:
    """按块内下标 ``k`` 写回全部 Hyena 状态 + 四键 offset。

    ``k`` ∈ ``[0, chunk_len-1]``：最后被消费 token 的块内下标（接受 k 个草稿
    且未截断时即验证器 ``accepted_len``；只消费 y 时为 0）。
    写回用 ``clone()``，与 stash 存储解耦（stash 下轮覆盖）。
    """
    if not isinstance(stash, ChunkStash):
        raise TypeError(f"stash 须为 ChunkStash，得到 {type(stash)!r}")
    if k < 0 or k >= stash.chunk_len:
        raise ValueError(f"k={k} 超出块内下标 [0, {stash.chunk_len - 1}]")

    for layer_idx, cat in stash.cat_u.items():
        params = _hyena_params(ip_dict, layer_idx)
        filt_k = stash.k_outer[layer_idx]
        expected = (filt_k - 1) + stash.chunk_len
        if cat.shape[-1] != expected:
            raise ValueError(
                f"层 {layer_idx} cat_u 末维 {cat.shape[-1]} != (K-1)+chunk_len={expected}"
            )
        params.fir_state_dict[layer_idx] = cat[..., k + 1 : k + filt_k].clone()

    for layer_idx, cat in stash.x1v_cat.items():
        params = _hyena_params(ip_dict, layer_idx)
        filt_k = stash.k_inner[layer_idx]
        expected = (filt_k - 1) + stash.chunk_len
        if cat.shape[-1] != expected:
            raise ValueError(
                f"层 {layer_idx} x1v_cat 末维 {cat.shape[-1]} != (K-1)+chunk_len={expected}"
            )
        params.fir_inner_state_dict[layer_idx] = cat[..., k + 1 : k + filt_k].clone()

    for layer_idx, s_all in stash.s_all.items():
        params = _hyena_params(ip_dict, layer_idx)
        if s_all.shape[-1] != stash.chunk_len:
            raise ValueError(
                f"层 {layer_idx} s_all 末维 {s_all.shape[-1]} != chunk_len={stash.chunk_len}"
            )
        params.state_dict[layer_idx] = s_all[..., k].clone()

    set_seqlen_offsets(ip_dict, stash.L0 + k + 1)
    for key in PARAM_KEYS:
        if int(ip_dict[key].seqlen_offset) != stash.L0 + k + 1:
            raise RuntimeError(f"{key}.seqlen_offset 未同步到 L0+k+1")
