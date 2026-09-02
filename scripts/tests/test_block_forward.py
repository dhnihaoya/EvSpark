"""Step 4（Plan 02 A.2，plans/06 §4.1）迷你模型对拍：块前向 vs 原生逐 token。

fp32 迷你 StripedHyena（hcs/hcm/hcl/mha 各 2 层，hidden 256，随机初始化 + 稳定
IIR 极点），同一随机 prompt prefill 后喂 γ ∈ {1,2,4,8} 个随机 token：
(a) 逐 token 原生 step 路径；(b) 块前向。断言逐位置 logits 与全部末状态
（KV、fir_state、fir_inner_state、IIR state）max|diff| < 1e-4。

运行环境：evo2 conda 环境 + GPU0（CUDA_VISIBLE_DEVICES=0）；fp32 秒级。
CPU 套件（无 torch/vortex）下本文件整体 skip，不影响既有 28 项测试。
"""

from __future__ import annotations

import os

# 必须在 torch import 前限定 GPU0（迷你模型只准落在 GPU0）
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="需要 evo2 环境（torch + CUDA）")
if not torch.cuda.is_available():
    # vortex.model 在 import 时初始化 CUDA，无卡机器会在 collection 阶段直接抛
    # RuntimeError，这里转成干净的 skip（本文件全部用例都需要 GPU/编译内核）。
    pytest.skip("需要 CUDA", allow_module_level=True)
pytest.importorskip("vortex", reason="需要 vortex(vtx) 包")

from vortex.model.model import ParallelGatedConvBlock, StripedHyena
from vortex.model.utils import dotdict

from specdec.block import (
    block_forward,
    clone_inference_params,
    diff_states,
    prefill,
    snapshot_states,
    step_forward_reference,
)

# ACGTN 的字节级 token id（CharLevelTokenizer 约定）
ACGTN_IDS = np.array([65, 67, 71, 84, 78], dtype=np.int64)
TOL = 1e-4
PREFIX_LEN = 192  # 须 ≥ HCM 滤波窗 127（否则 prefill 状态截断，step 语义有 quirk）
MODEL_SEED = 20260820
TOKEN_SEED = 20260821


def _mini_config() -> dotdict:
    """覆盖全部层型的迷你配置（排布镜像 evo2_7b：HCS→HCM→HCL→MHA 一循环×2）。"""
    return dotdict(
        {
            "model_name": "mini-specdec",
            "vocab_size": 512,
            "hidden_size": 256,
            "num_filters": 256,
            "hcl_layer_idxs": [2, 6],
            "hcm_layer_idxs": [1, 5],
            "hcs_layer_idxs": [0, 4],
            "attn_layer_idxs": [3, 7],
            "hcm_filter_length": 128,
            "hcm_filter_groups": 64,
            "hcs_filter_groups": 64,
            "hcs_filter_length": 7,
            "num_layers": 8,
            "short_filter_length": 3,
            "short_filter_bias": False,
            "num_attention_heads": 4,
            "state_size": 16,
            "rotary_emb_base": 10000,
            "eps": 1e-6,
            "proj_groups": 1,
            "hyena_filter_groups": 1,
            "column_split_hyena": False,
            "column_split": False,
            "interleave": True,
            "evo2_style_activations": True,
            "model_parallel_size": 1,
            "pipe_parallel_size": 1,
            "tie_embeddings": True,
            "mha_out_proj_bias": True,
            "hyena_out_proj_bias": True,
            "hyena_flip_x1x2": False,
            "qkv_proj_bias": False,
            "use_fp8_input_projections": False,
            "max_seqlen": 4096,
            "max_batch_size": 1,
            "final_norm": True,
            # fp32 慢路径注意力（CrossAttention + KV cache），保证 1e-4 紧对拍
            "use_flash_attn": False,
            "use_flash_rmsnorm": False,
            "use_flash_depthwise": False,
            "use_flashfft": False,
            "inference_mode": True,
            "prefill_style": "fft",
            "mlp_activation": "gelu",
            "print_activations": False,
            "inner_size_multiple_of": 16,
            "make_vocab_size_divisible_by": 8,
            "params_dtype": torch.float32,
            "hyena_block_dtype": torch.float32,
            "attn_block_dtype": torch.float32,
            "mlp_dtype": torch.float32,
        }
    )


def _build_mini_model() -> StripedHyena:
    torch.manual_seed(MODEL_SEED)
    model = StripedHyena(_mini_config())
    # 随机初始化下 log_poles ~ N(0,1) → 极点 e^{±1} 可能 >1，IIR 指数发散无法对拍。
    # 把极点压到 |p| ∈ (e^{-1.05}, e^{-0.05})（独立种子，保持随机性但稳定）。
    gen = torch.Generator().manual_seed(MODEL_SEED + 1)
    with torch.no_grad():
        for block in model.blocks:
            if isinstance(block, ParallelGatedConvBlock) and block.filter.h is None:
                shape = block.filter.log_poles.shape
                stable = -(0.05 + torch.rand(shape, generator=gen))
                block.filter.log_poles.data = stable.to(block.filter.log_poles.device)
    model = model.to(torch.float32)
    model.eval()
    return model


@pytest.fixture(scope="module")
def mini_model():
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA（GPU0）")
    return _build_mini_model()


def _rand_ids(rng: np.random.Generator, n: int) -> torch.Tensor:
    return torch.tensor(rng.choice(ACGTN_IDS, size=n), dtype=torch.long)[None].to("cuda:0")


@pytest.mark.parametrize("gamma", [1, 2, 4, 8])
def test_block_vs_step(mini_model, gamma: int):
    rng = np.random.default_rng(TOKEN_SEED + gamma)
    prompt = _rand_ids(rng, PREFIX_LEN)
    chunk = _rand_ids(rng, gamma)

    with torch.inference_mode():
        ip = mini_model.initialize_inference_params(max_seqlen=PREFIX_LEN + 64)
        pre_logits = prefill(mini_model, prompt, ip)
        p1 = pre_logits[0, -1].float().cpu()  # p(·|prefix)，验证器契约的 p₁

        ip_step = clone_inference_params(ip)
        step_logits = step_forward_reference(mini_model, chunk, ip_step)
        block_logits = block_forward(mini_model, chunk, ip)

        # --- 逐位置 logits：块前向 vs 逐 token ---
        diff = (block_logits[0].float() - step_logits[0].float()).abs()
        assert float(diff.max()) < TOL, f"γ={gamma} logits max|diff|={float(diff.max()):.3e}"

        # --- p₁..p_{γ+1} 拼接语义（off-by-one 红线） ---
        # 块输出第 0 位必须是 p(·|prefix, x_1)（step 参照的第 0 位），不是 p₁；
        # [p₁; 块输出] 拼成 γ+1 行即验证器契约形状。
        full = torch.cat([p1[None], block_logits[0].float().cpu()], dim=0)
        assert full.shape == (gamma + 1, p1.shape[0])
        assert float((block_logits[0, 0].float() - step_logits[0, 0].float()).abs().max()) < TOL

        # --- 末状态逐项比对（KV 按已写位置、FIR 滑窗、IIR 态） ---
        snap_block = snapshot_states(ip, kv_len=PREFIX_LEN + gamma)
        snap_step = snapshot_states(ip_step, kv_len=PREFIX_LEN + gamma)
        diffs = diff_states(snap_block, snap_step)
        worst = max(diffs.items(), key=lambda kv: kv[1]["max_abs"])
        assert worst[1]["max_abs"] < TOL, f"γ={gamma} 状态最大差 {worst[0]}: {worst[1]['max_abs']:.3e}"

        # --- 簿记：四键 offset 同步推进 γ ---
        for k in ("mha", "hcl", "hcm", "hcs"):
            assert ip[k].seqlen_offset == PREFIX_LEN + gamma
            assert ip_step[k].seqlen_offset == PREFIX_LEN + gamma
