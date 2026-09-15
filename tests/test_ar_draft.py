"""Phase 1 D1：First-N 截断自草稿 + 独立 AR drafter。

CPU：快照/恢复结构、n_blocks 守卫（无 torch 也可 import 失败则 skip）。
GPU（迷你 StripedHyena，fp32）：
1. first_n_forward N=全部层 ≡ 原生 step logits（1e-4）。
2. First-N propose 不污染深层状态 / seqlen_offset。
3. First-N N=全部层贪心全接受 + 与 native 逐位一致。
4. First-N N=半层贪心无损（允许拒绝）。
5. IndependentAR 双副本贪心全接受 + 无损；commit 后 offset = L + n。
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="需要 evo2 环境（torch）")
if not torch.cuda.is_available():
    # vortex.model 在 import 时初始化 CUDA，无卡机器会在 collection 阶段直接抛
    # RuntimeError，这里转成干净的 skip（本文件 GPU 用例需要 CUDA）。
    pytest.skip("需要 CUDA", allow_module_level=True)
pytest.importorskip("vortex", reason="需要 vortex(vtx)")

from evspark.specdec.block.ar_draft import (  # noqa: E402
    FirstNDraftModel,
    IndependentARDraftModel,
    first_n_forward,
    restore_prefix_states,
    snapshot_prefix_states,
)
from evspark.specdec.block.driver import (  # noqa: E402
    get_seqlen_offset,
    prefill,
    snapshot_states,
    step_forward_reference,
    step_seqlen_offsets,
)

SEED = 20260914
GAMMA = 4
N_TOK = 32


@pytest.fixture(scope="module")
def mini_model():
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA（GPU0）")
    from test_block_forward import _build_mini_model

    return _build_mini_model()


def _prompt(rng, n: int):
    from test_block_forward import ACGTN_IDS

    ids = rng.choice(ACGTN_IDS, size=n)
    return torch.tensor(ids, dtype=torch.long)[None].to("cuda:0")


def test_first_n_nblocks_guard(mini_model):
    with pytest.raises(ValueError, match="越界"):
        FirstNDraftModel(mini_model, n_blocks=0)
    with pytest.raises(ValueError, match="越界"):
        FirstNDraftModel(mini_model, n_blocks=9)
    FirstNDraftModel(mini_model, n_blocks=4)
    FirstNDraftModel(mini_model, n_blocks=8)


def test_first_n_forward_eq_full_step(mini_model):
    """N=全部层的 first_n_forward ≡ 原生逐 token。"""
    from test_block_forward import PREFIX_LEN, TOL

    rng = np.random.default_rng(SEED)
    prompt = _prompt(rng, PREFIX_LEN)
    tok = _prompt(rng, 1)
    n_layers = int(len(mini_model.blocks))
    with torch.inference_mode():
        ip = mini_model.initialize_inference_params(max_seqlen=PREFIX_LEN + 8)
        prefill(mini_model, prompt, ip)
        ip_n = snapshot_prefix_states(ip, tuple(range(n_layers)))
        # 用一份 ip 跑 first_n，另一份 clone 跑 step——先 clone 再各自步进
        from evspark.specdec.block.driver import clone_inference_params

        ip_step = clone_inference_params(ip)
        logits_n = first_n_forward(mini_model, tok, ip, n_layers)
        step_seqlen_offsets(ip, 1)
        logits_s = step_forward_reference(mini_model, tok, ip_step)
        diff = float((logits_n[0].float() - logits_s[0].float()).abs().max())
        assert diff < TOL, f"N=all first_n vs step max|Δ|={diff:.3e}"
        restore_prefix_states(ip, ip_n)  # 不污染后续
        assert get_seqlen_offset(ip) == PREFIX_LEN


def test_first_n_propose_does_not_pollute(mini_model):
    from test_block_forward import PREFIX_LEN, TOL

    rng = np.random.default_rng(SEED + 1)
    prompt = _prompt(rng, PREFIX_LEN)
    n_half = 4
    draft = FirstNDraftModel(mini_model, n_half)
    with torch.inference_mode():
        ip = mini_model.initialize_inference_params(max_seqlen=PREFIX_LEN + 64)
        prefill(mini_model, prompt[:, :-1], ip)
        off0 = get_seqlen_offset(ip)
        later = tuple(range(n_half, int(len(mini_model.blocks))))
        snap_later = snapshot_prefix_states(ip, later)
        kv_len = off0
        full_before = snapshot_states(ip, kv_len=kv_len)
        draft.propose_block(
            int(prompt[0, -1].item()),
            GAMMA,
            True,
            np.random.default_rng(0),
            inference_params_dict=ip,
        )
        assert get_seqlen_offset(ip) == off0
        full_after = snapshot_states(ip, kv_len=kv_len)
        from evspark.specdec.block.driver import diff_states

        diffs = diff_states(full_before, full_after)
        worst = max(diffs.items(), key=lambda kv: kv[1]["max_abs"])
        assert worst[1]["max_abs"] < TOL, f"propose 污染 {worst[0]}: {worst[1]['max_abs']:.3e}"
        restore_prefix_states(ip, snap_later)  # 形状 sanity（later 本不应变）


def test_first_n_full_layers_greedy_full_accept(mini_model):
    from evspark.specdec.block.loop import native_greedy_reference, speculative_generate
    from test_block_forward import PREFIX_LEN

    rng = np.random.default_rng(SEED + 2)
    prompt = _prompt(rng, PREFIX_LEN)
    n_layers = int(len(mini_model.blocks))
    draft = FirstNDraftModel(mini_model, n_layers)
    spec = speculative_generate(
        mini_model, draft, prompt, N_TOK, GAMMA, greedy=True, rng=np.random.default_rng(0)
    )
    native, _, _ = native_greedy_reference(mini_model, prompt, N_TOK)
    np.testing.assert_array_equal(spec.emitted_ids, native)
    # 未截断轮应全接受（截断最后一轮 take 可能 < γ+1，k 仍可以是 γ）
    assert all(r.k == GAMMA for r in spec.rounds_log[:-1])


def test_first_n_half_layers_greedy_lossless(mini_model):
    from evspark.specdec.block.loop import native_greedy_reference, speculative_generate
    from test_block_forward import PREFIX_LEN

    rng = np.random.default_rng(SEED + 3)
    prompt = _prompt(rng, PREFIX_LEN)
    draft = FirstNDraftModel(mini_model, 4)
    spec = speculative_generate(
        mini_model, draft, prompt, N_TOK, GAMMA, greedy=True, rng=np.random.default_rng(0)
    )
    native, _, _ = native_greedy_reference(mini_model, prompt, N_TOK)
    np.testing.assert_array_equal(spec.emitted_ids, native)
    assert any(r.k < GAMMA for r in spec.rounds_log), "半层截断应覆盖拒绝"


def test_independent_ar_twin_greedy(mini_model):
    from evspark.specdec.block.loop import native_greedy_reference, speculative_generate
    from test_block_forward import PREFIX_LEN, _build_mini_model

    rng = np.random.default_rng(SEED + 4)
    prompt = _prompt(rng, PREFIX_LEN)
    twin = _build_mini_model()
    draft = IndependentARDraftModel(twin, max_context=4096)
    spec = speculative_generate(
        mini_model, draft, prompt, N_TOK, GAMMA, greedy=True, rng=np.random.default_rng(0)
    )
    native, _, _ = native_greedy_reference(mini_model, prompt, N_TOK)
    np.testing.assert_array_equal(spec.emitted_ids, native)
    assert all(r.k == GAMMA for r in spec.rounds_log[:-1])
    # 1B 侧消费了完整 prompt + 全部 emitted（含末锚点）
    assert get_seqlen_offset(draft.ip) == PREFIX_LEN + N_TOK
    draft.reset()
    assert draft.ip is None


def test_independent_ar_over_context_ineligible():
    from test_block_forward import _build_mini_model

    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA")
    m = _build_mini_model()
    draft = IndependentARDraftModel(m, max_context=200)
    prompt = torch.zeros(1, 192, dtype=torch.long, device="cuda:0")
    prompt[0] = 65
    with pytest.raises(ValueError, match="无资格"):
        draft.prefill_prompt(prompt, max_seqlen=256)
