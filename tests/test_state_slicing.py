"""Step 7（Plan 02 A.3，plans/09 §4.1）状态切片回滚：迷你模型 fp32。

1. 切片 vs 重放等价（k ∈ {0,1,2,γ−1,γ}）
2. 循环级状态不变量（slice 模式，恒等 drafter，必有拒绝）
3. 贪心逐 token 全等（slice，γ ∈ {2,4,8}）
4. snapshot 模式回归（与逐步参照全等）

无 torch/vortex/CUDA 时 skip。复用 Step 4 迷你模型 fixture。
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

SEED_PROMPT = 20260823
SEED_CHUNK = 20260822
N_GREEDY = 64
TOL = 1e-4
GAMMA = 8
GREEDY_GAMMAS = (2, 4, 8)
SLICE_KS = (0, 1, 2, GAMMA - 1, GAMMA)


@pytest.fixture(scope="module")
def mini_model():
    pytest.importorskip("torch", reason="需要 evo2 环境（torch + CUDA）")
    pytest.importorskip("vortex", reason="需要 vortex(vtx) 包")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA（GPU0）")
    from test_block_forward import _build_mini_model

    return _build_mini_model()


def _prompt_ids(torch, rng, n: int):
    from test_block_forward import ACGTN_IDS

    ids = rng.choice(ACGTN_IDS, size=n)
    return torch.tensor(ids, dtype=torch.long)[None].to("cuda:0")


def _worst(diffs: dict) -> tuple[str, float]:
    item = max(diffs.items(), key=lambda kv: kv[1]["max_abs"])
    return item[0], float(item[1]["max_abs"])


def _assert_states_close(snap_a, snap_b, what: str):
    from evspark.specdec.block.driver import diff_states

    diffs = diff_states(snap_a, snap_b)
    path, mag = _worst(diffs)
    assert mag < TOL, f"{what} 状态最大差 {path}: {mag:.3e}"


class _OracleGreedyDraft:
    """把已知贪心续延做成 one-hot，覆盖全接受 / 截断轮。"""

    def __init__(self, prompt_len: int, continuation: np.ndarray):
        self.prompt_len = int(prompt_len)
        self.continuation = np.asarray(continuation, dtype=np.int64).ravel()

    def probs(self, prefix: np.ndarray) -> np.ndarray:
        from evspark.specdec.drafts import MASS_IDS, VOCAB_SIZE

        prefix = np.asarray(prefix, dtype=np.int64).ravel()
        idx = int(prefix.size) - self.prompt_len
        out = np.zeros(VOCAB_SIZE, dtype=np.float64)
        if 0 <= idx < self.continuation.size:
            out[int(self.continuation[idx])] = 1.0
        elif prefix.size:
            last = int(prefix[-1])
            if 0 <= last < VOCAB_SIZE:
                out[last] = 1.0
            else:
                out[np.array(MASS_IDS)] = 0.2
        else:
            out[np.array(MASS_IDS)] = 0.2
        return out


def test_retain_false_matches_retain_true_end_state(mini_model):
    """未切片前，retain 开关不得改变 logits 与末态。"""
    import torch

    from evspark.specdec.block.driver import (
        block_forward,
        clone_inference_params,
        prefill,
        snapshot_states,
    )
    from test_block_forward import PREFIX_LEN

    rng = np.random.default_rng(SEED_CHUNK)
    prompt = _prompt_ids(torch, rng, PREFIX_LEN)
    chunk = _prompt_ids(torch, rng, GAMMA + 1)
    with torch.inference_mode():
        ip0 = mini_model.initialize_inference_params(max_seqlen=PREFIX_LEN + 64)
        prefill(mini_model, prompt, ip0)
        ip_off = clone_inference_params(ip0)
        ip_on = clone_inference_params(ip0)
        logits_off = block_forward(mini_model, chunk, ip_off, retain=False)
        logits_on, stash = block_forward(mini_model, chunk, ip_on, retain=True)
        d = (logits_off.float() - logits_on.float()).abs()
        assert float(d.max()) < TOL, f"retain logits max|diff|={float(d.max()):.3e}"
        kv_len = PREFIX_LEN + GAMMA + 1
        _assert_states_close(
            snapshot_states(ip_off, kv_len=kv_len),
            snapshot_states(ip_on, kv_len=kv_len),
            "retain 末态",
        )
        assert stash.L0 == PREFIX_LEN
        assert stash.chunk_len == GAMMA + 1
        assert stash.cat_u and stash.s_all and stash.x1v_cat


def test_slice_vs_replay_equivalent(mini_model):
    """同一 prompt+草稿块，k ∈ {0,1,2,γ−1,γ} 切片态 vs 快照重放态。"""
    import torch

    from evspark.specdec.block.driver import (
        block_forward,
        clone_inference_params,
        get_seqlen_offset,
        prefill,
        snapshot_states,
    )
    from evspark.specdec.block.slice import slice_states_to_accept
    from test_block_forward import PREFIX_LEN

    rng = np.random.default_rng(SEED_CHUNK + 1)
    prompt = _prompt_ids(torch, rng, PREFIX_LEN)
    chunk = _prompt_ids(torch, rng, GAMMA + 1)
    with torch.inference_mode():
        ip0 = mini_model.initialize_inference_params(max_seqlen=PREFIX_LEN + 64)
        prefill(mini_model, prompt, ip0)
        L0 = get_seqlen_offset(ip0)
        assert L0 == PREFIX_LEN
        snap = clone_inference_params(ip0)
        ip_full = clone_inference_params(ip0)
        _logits, stash = block_forward(mini_model, chunk, ip_full, retain=True)
        assert stash.chunk_len == GAMMA + 1

        # k=γ 不变量：切片末态 = 块前向已写入末态
        ip_end = clone_inference_params(ip_full)
        slice_states_to_accept(ip_end, stash, GAMMA)
        kv_full = L0 + GAMMA + 1
        _assert_states_close(
            snapshot_states(ip_full, kv_len=kv_full),
            snapshot_states(ip_end, kv_len=kv_full),
            "k=γ 切片 vs 块前向末态",
        )

        for k in SLICE_KS:
            ip_slice = clone_inference_params(ip_full)
            slice_states_to_accept(ip_slice, stash, k)
            ip_replay = clone_inference_params(snap)
            replay = chunk[:, : k + 1]
            block_forward(mini_model, replay, ip_replay)
            kv_len = L0 + k + 1
            assert get_seqlen_offset(ip_slice) == kv_len
            assert get_seqlen_offset(ip_replay) == kv_len
            _assert_states_close(
                snapshot_states(ip_slice, kv_len=kv_len),
                snapshot_states(ip_replay, kv_len=kv_len),
                f"k={k} 切片 vs 重放",
            )


@pytest.mark.parametrize("gamma", GREEDY_GAMMAS)
def test_greedy_tokenwise_equal_slice(mini_model, gamma: int):
    import torch

    from evspark.specdec.block.loop import native_greedy_reference, speculative_generate
    from evspark.specdec.drafts import IdentityDraftModel
    from test_block_forward import PREFIX_LEN

    rng = np.random.default_rng(SEED_PROMPT + gamma)
    prompt = _prompt_ids(torch, rng, PREFIX_LEN)
    native, _ip, _ = native_greedy_reference(mini_model, prompt, N_GREEDY)
    spec = speculative_generate(
        mini_model,
        IdentityDraftModel(),
        prompt,
        N_GREEDY,
        gamma,
        greedy=True,
        rng=np.random.default_rng(0),
        rollback="slice",
    )
    np.testing.assert_array_equal(spec.emitted_ids, native)
    assert spec.rollback == "slice"
    assert any(r.k < gamma for r in spec.rounds_log), "恒等 drafter 应覆盖拒绝+切片"
    assert all(r.replay_len == 0 for r in spec.rounds_log)


def test_state_invariant_slice_mode(mini_model):
    """slice 模式跑 64 token（γ=8，必有拒绝），末态 vs teacher-force < 1e-4。"""
    import torch

    from evspark.specdec.block.driver import get_seqlen_offset, snapshot_states
    from evspark.specdec.block.loop import (
        spec_end_kv_len,
        speculative_generate,
        teacher_force_to_spec_state,
    )
    from evspark.specdec.drafts import IdentityDraftModel
    from test_block_forward import PREFIX_LEN

    gamma = 8
    rng = np.random.default_rng(SEED_PROMPT + 99)
    prompt = _prompt_ids(torch, rng, PREFIX_LEN)
    spec = speculative_generate(
        mini_model,
        IdentityDraftModel(),
        prompt,
        N_GREEDY,
        gamma,
        greedy=True,
        rng=np.random.default_rng(0),
        rollback="slice",
    )
    assert any(r.k < gamma for r in spec.rounds_log)
    kv_len = spec_end_kv_len(PREFIX_LEN, N_GREEDY)
    assert get_seqlen_offset(spec.inference_params_dict) == kv_len
    snap_spec = snapshot_states(spec.inference_params_dict, kv_len=kv_len)
    ip_ref = teacher_force_to_spec_state(mini_model, prompt, spec.emitted_ids)
    assert get_seqlen_offset(ip_ref) == kv_len
    _assert_states_close(
        snap_spec,
        snapshot_states(ip_ref, kv_len=kv_len),
        "slice 循环末态 vs teacher-force",
    )


def test_slice_vs_snapshot_tokens_and_state(mini_model):
    """同种子 slice 与 snapshot 贪心输出全等，末态一致。"""
    import torch

    from evspark.specdec.block.driver import snapshot_states
    from evspark.specdec.block.loop import spec_end_kv_len, speculative_generate
    from evspark.specdec.drafts import IdentityDraftModel
    from test_block_forward import PREFIX_LEN

    gamma = 8
    rng = np.random.default_rng(SEED_PROMPT + 5)
    prompt = _prompt_ids(torch, rng, PREFIX_LEN)
    spec_s = speculative_generate(
        mini_model,
        IdentityDraftModel(),
        prompt,
        N_GREEDY,
        gamma,
        greedy=True,
        rng=np.random.default_rng(0),
        rollback="slice",
    )
    spec_p = speculative_generate(
        mini_model,
        IdentityDraftModel(),
        prompt,
        N_GREEDY,
        gamma,
        greedy=True,
        rng=np.random.default_rng(0),
        rollback="snapshot",
    )
    np.testing.assert_array_equal(spec_s.emitted_ids, spec_p.emitted_ids)
    kv_len = spec_end_kv_len(PREFIX_LEN, N_GREEDY)
    _assert_states_close(
        snapshot_states(spec_s.inference_params_dict, kv_len=kv_len),
        snapshot_states(spec_p.inference_params_dict, kv_len=kv_len),
        "slice vs snapshot 末态",
    )


def test_snapshot_mode_regression(mini_model):
    """snapshot 路径：贪心全等 + 拒绝回放仍发生。"""
    import torch

    from evspark.specdec.block.loop import native_greedy_reference, speculative_generate
    from evspark.specdec.drafts import IdentityDraftModel
    from test_block_forward import PREFIX_LEN

    gamma = 8
    rng = np.random.default_rng(SEED_PROMPT + 8)
    prompt = _prompt_ids(torch, rng, PREFIX_LEN)
    native, _ip, _ = native_greedy_reference(mini_model, prompt, N_GREEDY)
    spec = speculative_generate(
        mini_model,
        IdentityDraftModel(),
        prompt,
        N_GREEDY,
        gamma,
        greedy=True,
        rng=np.random.default_rng(0),
        rollback="snapshot",
    )
    np.testing.assert_array_equal(spec.emitted_ids, native)
    assert spec.rollback == "snapshot"
    assert any(r.k < gamma and r.replay_len > 0 for r in spec.rounds_log)


def test_truncation_round_slices(mini_model):
    """截断轮：oracle 全接受但 n_tokens 不整除 γ+1，切片到 take−1。"""
    import torch

    from evspark.specdec.block.driver import get_seqlen_offset, snapshot_states
    from evspark.specdec.block.loop import (
        native_greedy_reference,
        spec_end_kv_len,
        speculative_generate,
        teacher_force_to_spec_state,
    )
    from test_block_forward import PREFIX_LEN

    n_tokens = 10
    gamma = 8
    rng = np.random.default_rng(SEED_PROMPT + 7)
    prompt = _prompt_ids(torch, rng, PREFIX_LEN)
    native, _ip, _ = native_greedy_reference(mini_model, prompt, n_tokens)
    draft = _OracleGreedyDraft(PREFIX_LEN, native)
    spec = speculative_generate(
        mini_model,
        draft,
        prompt,
        n_tokens,
        gamma,
        greedy=True,
        rng=np.random.default_rng(0),
        rollback="slice",
    )
    np.testing.assert_array_equal(spec.emitted_ids, native)
    assert any(r.n_emitted < r.k + 1 for r in spec.rounds_log), "应覆盖截断轮"
    kv_len = spec_end_kv_len(PREFIX_LEN, n_tokens)
    assert get_seqlen_offset(spec.inference_params_dict) == kv_len
    ip_ref = teacher_force_to_spec_state(mini_model, prompt, spec.emitted_ids)
    _assert_states_close(
        snapshot_states(spec.inference_params_dict, kv_len=kv_len),
        snapshot_states(ip_ref, kv_len=kv_len),
        "截断轮后状态",
    )
