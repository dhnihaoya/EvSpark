"""状态切片回滚（Plan 02 A.3 / Step 7）：用块前向暂存的逐位置中间量切到接受位。

循环块输入 ``[y, x₁..x_γ]``（下标 0..γ）。接受 ``x₁..x_k`` 后目标状态 =
**消费到块内下标 k**（即 x_k）；新锚点 y′ 不消费。切片位置 ``j* = k``。
截断轮：提交前缀长度为 ``take``，等价 ``j* = take - 1``。

逐层规则（``cat`` = ``[缓存 S 位 | 块输入]``，S≤K−1，K = 滤波器长）：

- FIR：消费到下标 k 后取 ``cat[..., max(0, S+k+1-(K-1)) : S+k+1]``。
  满窗 S=K−1 时退化为 ``[..., k+1 : k+K]``。
- HCL IIR：``s_all[..., k]``（块前向已算出的逐位置状态，fp32）
- MHA KV：零数据移动，四键 ``seqlen_offset = L0 + k + 1``；变长 k 时取
  ``L0 + max(k) + 1`` 并写 ``lengths_per_sample``（仅 flash_attn 快路径按 sample
  生效；SDPA/慢路径只认标量 offset，变长切片会被本模块拒绝）

k=0：只消费锚点 y → IIR 取 ``s_all[..., 0]``（消费 y **之后**，不是轮初 s0）。
k=γ：切出末态，与块前向已写入的增长/滑窗状态 / ``s_all[..., -1]`` 一致；
循环在全接受且未截断时跳过本函数（零开销）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from evspark.specdec.block.driver import PARAM_KEYS, set_seqlen_offsets

_HYENA_KEYS = ("hcl", "hcm", "hcs")


@dataclass
class ChunkStash:
    """一块前向的保留中间量。每轮覆盖，不累积。

    下标：``chunk_len`` = 块输入长度（循环里 γ+1）；位置 0 = 锚点 y。
    ``s_all[..., j]`` 与 FIR 窗 ``[..., max(0, S+j+1-(K-1)) : S+j+1]``
    对应消费到下标 j 之后（满窗 S=K−1 即 ``[..., j+1 : j+K]``）。
    """

    L0: int
    chunk_len: int
    cat_u: dict[int, torch.Tensor] = field(default_factory=dict)
    k_outer: dict[int, int] = field(default_factory=dict)
    x1v_cat: dict[int, torch.Tensor] = field(default_factory=dict)
    k_inner: dict[int, int] = field(default_factory=dict)
    s_all: dict[int, torch.Tensor] = field(default_factory=dict)
    #: attention 是否走 flash_attn kvcache 快路径（driver 建 stash 时按模型记录）。
    #: 慢路径（SDPA）只认标量 seqlen_offset，变长切片会被 slice_states_to_accept 拒绝
    flash_attn: bool = True


def _hyena_params(ip_dict: dict, layer_idx: int):
    for key in _HYENA_KEYS:
        if layer_idx in ip_dict[key].fir_state_dict:
            return ip_dict[key]
    raise KeyError(f"layer_idx {layer_idx} 不在任何 Hyena 参数组的 fir_state_dict")


def _stash_batch(stash: ChunkStash) -> int:
    for src in (stash.cat_u, stash.x1v_cat, stash.s_all):
        if src:
            return int(next(iter(src.values())).shape[0])
    return 1


def _as_k_array(k, chunk_len: int, batch: int) -> np.ndarray:
    arr = np.asarray(k, dtype=np.int64).ravel()
    if arr.size == 1:
        arr = np.full(batch, int(arr[0]), dtype=np.int64)
    if arr.size != batch:
        raise ValueError(f"k 长度 {arr.size} 与 stash batch={batch} 不一致")
    if np.any(arr < 0) or np.any(arr >= chunk_len):
        raise ValueError(f"k={arr.tolist()} 超出块内下标 [0, {chunk_len - 1}]")
    return arr


def _fir_prefill_len(cat: torch.Tensor, chunk_len: int, filt_k: int) -> int:
    s = int(cat.shape[-1]) - int(chunk_len)
    if s < 0 or s > filt_k - 1:
        raise ValueError(
            f"FIR cat 末维 {cat.shape[-1]} 与 chunk_len={chunk_len}、K={filt_k} 不兼容（S={s}）"
        )
    return s


def _fir_window(cat: torch.Tensor, kk: int, s: int, filt_k: int) -> torch.Tensor:
    """消费到块内下标 ``kk`` 之后的 FIR 状态（短窗增长或满窗滑窗）。"""
    end = s + int(kk) + 1
    start = max(0, end - (filt_k - 1))
    return cat[..., start:end]


def _gather_fir_windows(cat: torch.Tensor, ks: np.ndarray, s: int, filt_k: int) -> torch.Tensor:
    if bool(np.all(ks == ks[0])):
        return _fir_window(cat, int(ks[0]), s, filt_k)
    windows = [_fir_window(cat[b], int(ks[b]), s, filt_k) for b in range(int(cat.shape[0]))]
    widths = {int(w.shape[-1]) for w in windows}
    if len(widths) == 1:
        return torch.stack(windows, dim=0)
    # HCM（flip）左零填到对齐宽度与短 tap 选取等价；HCS/外层不 flip，禁止。
    if filt_k < 128:
        raise ValueError(
            "短于 HCS/外层窗且变长切片会得到不同 FIR 状态长度 "
            f"{sorted(int(w.shape[-1]) for w in windows)}；请先 collapse 或加长前缀"
        )
    width = max(widths)
    out = cat.new_zeros(*cat.shape[:-1], width)
    for b, w in enumerate(windows):
        out[b, ..., -w.shape[-1] :] = w
    return out


def slice_states_to_accept(ip_dict: dict, stash: ChunkStash, k) -> None:
    """按块内下标 ``k`` 写回全部 Hyena 状态 + 四键 offset。

    ``k`` ∈ ``[0, chunk_len-1]``：最后被消费 token 的块内下标（接受 k 个草稿
    且未截断时即验证器 ``accepted_len``；只消费 y 时为 0）。
    ``k`` 可为标量（所有 batch 同切）或长度为 B 的数组（每序列变长）。
    写回用 ``clone()``，与 stash 存储解耦（stash 下轮覆盖）。

    变长时 ``seqlen_offset = L0 + max(k) + 1``，并写 ``lengths_per_sample[:B]``
    （flash_attn 快路径按 sample 读 cache；SDPA 慢路径仍只认标量 offset）。
    """
    if not isinstance(stash, ChunkStash):
        raise TypeError(f"stash 须为 ChunkStash，得到 {type(stash)!r}")
    batch = _stash_batch(stash)
    ks = _as_k_array(k, stash.chunk_len, batch)
    uniform = bool(np.all(ks == ks[0]))
    if not uniform and not stash.flash_attn:
        raise ValueError(
            "变长切片要求 flash_attn kvcache 快路径（lengths_per_sample 按 sample 读 "
            "cache）；当前为 SDPA/慢路径，只认标量 seqlen_offset=max(k)，短序列会读到 "
            "stale KV。请先 collapse 收回单序列，或改用 flash_attn"
        )

    for layer_idx, cat in stash.cat_u.items():
        params = _hyena_params(ip_dict, layer_idx)
        filt_k = stash.k_outer[layer_idx]
        s = _fir_prefill_len(cat, stash.chunk_len, filt_k)
        params.fir_state_dict[layer_idx] = _gather_fir_windows(cat, ks, s, filt_k).clone()

    for layer_idx, cat in stash.x1v_cat.items():
        params = _hyena_params(ip_dict, layer_idx)
        filt_k = stash.k_inner[layer_idx]
        s = _fir_prefill_len(cat, stash.chunk_len, filt_k)
        params.fir_inner_state_dict[layer_idx] = _gather_fir_windows(cat, ks, s, filt_k).clone()

    for layer_idx, s_all in stash.s_all.items():
        params = _hyena_params(ip_dict, layer_idx)
        if s_all.shape[-1] != stash.chunk_len:
            raise ValueError(
                f"层 {layer_idx} s_all 末维 {s_all.shape[-1]} != chunk_len={stash.chunk_len}"
            )
        if uniform:
            params.state_dict[layer_idx] = s_all[..., int(ks[0])].clone()
        else:
            gathered = s_all.new_empty(*s_all.shape[:-1])
            for b in range(batch):
                gathered[b] = s_all[b, ..., int(ks[b])]
            params.state_dict[layer_idx] = gathered

    lengths = stash.L0 + ks + 1
    off = int(lengths.max())
    set_seqlen_offsets(ip_dict, off)
    for key in PARAM_KEYS:
        if int(ip_dict[key].seqlen_offset) != off:
            raise RuntimeError(f"{key}.seqlen_offset 未同步到 L0+max(k)+1")
    mha = ip_dict["mha"]
    # 变长必须写 lengths_per_sample，collapse 才能把 offset 收到胜出路径。
    if (not uniform) or mha.lengths_per_sample is not None:
        if mha.lengths_per_sample is None:
            cache = next(iter(mha.key_value_memory_dict.values()), None)
            if cache is None:
                raise RuntimeError("变长切片需要 KV cache 以分配 lengths_per_sample")
            mha.lengths_per_sample = torch.zeros(
                int(mha.max_batch_size), dtype=torch.int32, device=cache.device
            )
        dev = mha.lengths_per_sample.device
        mha.lengths_per_sample[:batch] = torch.as_tensor(
            lengths, device=dev, dtype=torch.int32
        )
