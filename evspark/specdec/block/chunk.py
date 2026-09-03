"""Hyena 层块前向（带初始状态的 chunk forward，Plan 02 A.2 / Step 4 核心）。

动机：vortex 的 ``HyenaCascade.forward`` 在 prefill 之后收到多 token 输入会走
``sequential_forward`` 并被 ``u[:, -1]`` 静默截断（vortex ``model/model.py:332-333``，
错了不报错）。本模块以**运行时 monkeypatch** 给 ``HyenaCascade`` 增加
「带初始状态的块前向」分支：

- patch 只在本包 import 路径内完成，**不改动 site-packages 任何文件**；
- patch 后的 dispatch 只有在 driver 显式设置 flag（``_CHUNK_FLAG``）且该层
  状态已存在（prefill 完成）时才走块前向，否则原样转交原始
  ``HyenaCascade.forward``——原生逐 token 解码路径行为零改变。

数学（与 ``engine.py`` 的 ``step_fir`` / ``step_iir`` / ``prefill_via_modal_fft``
逐步等价；推导见 ``notes/vortex_state_flow.md`` §4–5）：

- **外层短 FIR / HCS-7 / HCM-128**：状态 = 最近 K−1 个卷积输入。块前向 = 把缓存
  窗口拼到 chunk 前面做因果 conv（``F.conv1d``，fp32，padding=0，输出恰好 γ 位）。
  注意 HCM（K=128）的 step/parallel 路径用真卷积取向（``step_fir(flip_filter=True)``
  与 FFT conv），故 conv1d（互相关）前要翻转权重；外层 FIR 与 HCS（K<128）不翻。
- **HCL IIR**（无限记忆）：s_j = p^{j+1} ⊙ s0 + zis_j（j=0..γ−1 为 chunk 局部下标），
  其中零初态响应 zis 由单次 FFT（与 ``prefill_via_modal_fft`` 同型）一次性算出全部
  位置；输出 y_j = x2_j ⊙ (Σ_s residues·s_j + D·x1v_j)；末位置状态 s_{γ−1} 写回。
- 全部内部累加保持 **fp32**（与 vortex step 路径一致），输出 cast 回 ``data_dtype``；
  写回的状态一律 fp32（与 step 路径跑过一步后的状态 dtype 一致）。

保留模式：driver 在推理参数上挂 ``_STASH_ATTR``（ChunkStash）时，把 ``cat_u`` /
``x1v_cat`` / ``s_all`` 记入 stash，供切片回滚；未挂时不额外分配、写回与 Step 4
逐位一致。

Attention 层无需改造：``flash_attn_with_kvcache`` 原生支持 q_len=γ（原地写
cache + causal 右下对齐 + 内核内 rope），由模型正常 forward 走到。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vortex.model.model import HyenaCascade
from vortex.model.utils import column_split, interleave

#: driver 在块前向期间挂到各 Hyena 推理参数对象上的触发标记
_CHUNK_FLAG = "_specdec_chunk_forward"
#: 保留模式：同一对象上挂 ChunkStash，层前向把 cat_u / x1v_cat / s_all 记进去
_STASH_ATTR = "_specdec_chunk_stash"

_ORIGINAL_FORWARD = HyenaCascade.forward
_installed = False


def install_chunk_forward() -> None:
    """安装 HyenaCascade 块前向 dispatch（幂等）。

    安装后仅在推理参数对象携带 ``_CHUNK_FLAG`` 时改变该次前向行为；
    未带标记的调用（原生 decode、prefill）完全走 vortex 原路径。
    """
    global _installed
    if _installed:
        return
    HyenaCascade.forward = _patched_forward
    _installed = True


def _patched_forward(self, u, inference_params=None, padding_mask=None, *args, **kwargs):
    if (
        inference_params is not None
        and getattr(inference_params, _CHUNK_FLAG, False)
        and self.layer_idx in inference_params.fir_state_dict
    ):
        return hyena_chunk_forward(self, u, inference_params)
    return _ORIGINAL_FORWARD(self, u, inference_params, padding_mask, *args, **kwargs)


def hyena_chunk_forward(cascade: HyenaCascade, u: torch.Tensor, inference_params):
    """单个 Hyena 层的块前向。

    输入 ``u``: [B, γ, 3H]（projections 输出，data dtype）；返回
    ``(y, inference_params)``，``y``: [B, γ, H]。逐位置语义与连续调用 γ 次
    ``sequential_forward`` 完全一致，状态同步推进 γ 步。
    """
    if cascade.data_dtype is None:
        cascade.data_dtype = u.dtype
    dtype = cascade.data_dtype
    config = cascade.config
    layer_idx = cascade.layer_idx
    hidden = cascade.hidden_size

    # --- 外层短 FIR（K=3）：拼缓存窗 + 因果 conv（fp32） ---
    z_pre = u.permute(0, 2, 1)  # [B, 3H, γ]
    k_outer = cascade.short_filter_length
    fir_state = inference_params.fir_state_dict[layer_idx]  # [B, 3H, K−1]
    cat = torch.cat([fir_state, z_pre], dim=-1).to(torch.float32)
    w = cascade.short_filter_weight.to(torch.float32)  # [3H, 1, K]
    z = F.conv1d(cat, w, bias=None, stride=1, padding=0, groups=cat.shape[1])
    if cascade.short_filter_bias is not None:
        z = z + cascade.short_filter_bias.to(torch.float32)[None, :, None]
    # 末 K−1 个卷积输入即新滑窗状态（γ≥1 时与 step_fir 的 roll+append 逐项一致）
    inference_params.fir_state_dict[layer_idx] = cat[..., -(k_outer - 1) :]
    stash = getattr(inference_params, _STASH_ATTR, None)
    if stash is not None:
        stash.cat_u[layer_idx] = cat
        stash.k_outer[layer_idx] = int(k_outer)
    z_pre = z.to(dtype)

    if config.interleave:
        z_pre = interleave(z_pre)
    if cascade.column_split_hyena:
        x2, x1, v = column_split(
            z_pre, cascade.num_attention_heads, cascade.hidden_size_per_attention_head
        )
    else:
        x2, x1, v = z_pre.split([hidden, hidden, hidden], dim=1)
    if cascade.hyena_flip_x1x2:
        x1, x2 = x2, x1

    if cascade.fir_inner_filter_length is not None:
        y = _inner_fir_chunk(cascade, x1, x2, v, inference_params, dtype, stash)
    else:
        y = _inner_iir_chunk(cascade, x1, x2, v, inference_params, dtype, stash)

    return y.permute(0, 2, 1), inference_params


def _inner_fir_chunk(cascade, x1, x2, v, inference_params, dtype, stash=None):
    """HCS（K=7）/ HCM（K=128）内层 FIR 块前向；输入 x1/x2/v: [B, H, γ]。"""
    layer_idx = cascade.layer_idx
    k_inner = cascade.fir_inner_filter_length
    h = cascade.h
    if cascade.hyena_filter_groups > 1:
        h = h.repeat_interleave(cascade.hidden_size // cascade.hyena_filter_groups, 0)

    x1v = x1 * v  # data dtype（与 sequential_forward 的乘积精度一致）
    state = inference_params.fir_inner_state_dict[layer_idx]  # [B, H, K−1]
    if state.shape[-1] != k_inner - 1:
        # 前缀短于 HCM 窗口（K−1=127）时 prefill 只能存截断状态，vortex step 路径
        # 此时语义本身有 quirk（见 notes §6）；本项目 prompt ≥1k，直接拒绝。
        raise ValueError(
            f"层 {layer_idx} fir_inner_state 长度 {state.shape[-1]} != {k_inner - 1}："
            "前缀短于 HCM 滤波窗，块前向不支持"
        )
    cat = torch.cat([state, x1v], dim=-1).to(torch.float32)
    w = h.to(torch.float32)  # [H, 1, K]
    if k_inner >= 128:
        # HCM：step_fir(flip_filter=True) / parallel FFT conv 均为真卷积取向，
        # conv1d 是互相关，翻转权重后等价
        w = w.flip(-1)
    z = F.conv1d(cat, w, bias=None, stride=1, padding=0, groups=x1v.shape[1])
    if k_inner >= 128:
        # gated_bias（parallel_fir / step_fir 一致）：z += D ⊙ x1v（仅 chunk 的 γ 位）
        z = z + cascade.D.to(torch.float32)[None, :, None] * x1v.to(torch.float32)
    inference_params.fir_inner_state_dict[layer_idx] = cat[..., -(k_inner - 1) :]
    if stash is not None:
        stash.x1v_cat[layer_idx] = cat
        stash.k_inner[layer_idx] = int(k_inner)

    # step 路径先把 FIR 输出 cast 回 data dtype 再做 postgate
    return x2 * z.to(dtype)  # [B, H, γ]


def _inner_iir_chunk(cascade, x1, x2, v, inference_params, dtype, stash=None):
    """HCL 内层 IIR 块前向；输入 x1/x2/v: [B, H, γ]。

    s_j = p^{j+1} ⊙ s0 + zis_j；zis 由单次 FFT 卷积（零初态）给出全部位置，
    初态衰减项 p^{j+1}⊙s0 闭式相加。末位置状态写回 ``state_dict``（fp32）。
    """
    layer_idx = cascade.layer_idx
    gamma = x1.shape[-1]
    device = x1.device
    hidden = cascade.hidden_size

    s0 = inference_params.state_dict[layer_idx].to(torch.float32)  # [B, H, S]
    log_poles = cascade.log_poles.to(torch.float32)  # [G, S, 1]
    residues = cascade.residues.to(torch.float32)  # [G, S]
    groups = cascade.hyena_filter_groups
    if groups > 1 and groups != hidden:
        log_poles = log_poles.repeat_interleave(hidden // groups, 0)
        residues = residues.repeat_interleave(hidden // groups, 0)

    x1v = (x1 * v).to(torch.float32)  # [B, H, γ]
    fft_size = 2 * gamma
    t_zis = torch.arange(gamma, device=device, dtype=torch.float32)  # 0..γ−1
    t_zir = torch.arange(1, gamma + 1, device=device, dtype=torch.float32)  # 1..γ
    pows_zis = (log_poles * t_zis).exp()  # p^t, [H, S, γ]
    pows_zir = (log_poles * t_zir).exp()  # p^{j+1}, [H, S, γ]

    # 零初态响应（与 prefill_via_modal_fft 同型的单次 FFT 卷积，取全部 γ 位）
    x_s = torch.fft.fft(x1v, n=fft_size)  # [B, H, 2γ]
    state_s = torch.fft.fft(pows_zis, n=fft_size)  # [H, S, 2γ]
    zis = torch.fft.ifft(x_s[:, :, None, :] * state_s[None], n=fft_size)[..., :gamma]

    s_all = zis + s0[..., None] * pows_zir[None]  # [B, H, S, γ]，fp32
    inference_params.state_dict[layer_idx] = s_all[..., -1]
    if stash is not None:
        stash.s_all[layer_idx] = s_all

    res_state = (residues[None, :, :, None] * s_all).sum(dim=2)  # [B, H, γ]
    d = cascade.D.to(torch.float32)
    # step_iir: y = x2 * (res_state + D * x1v)（fp32 提升后由外层 cast 回 data dtype）
    y = x2.to(torch.float32) * (res_state + d[None, :, None] * x1v)
    return y.to(dtype)  # [B, H, γ]
