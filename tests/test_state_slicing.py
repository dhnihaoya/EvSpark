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


def test_slice_per_sequence_hyena(mini_model):
    """D4a：B=2 块前向后按不同 k 切片，各序列 Hyena 态对拍独立重放。"""
    import torch

    from evspark.specdec.block.driver import (
        block_forward,
        clone_inference_params,
        collapse_inference_params,
        expand_inference_params,
        get_seqlen_offset,
        prefill,
        snapshot_states,
    )
    from evspark.specdec.block.slice import slice_states_to_accept
    from test_block_forward import ACGTN_IDS, PREFIX_LEN

    prev = mini_model.config.max_batch_size
    mini_model.config.max_batch_size = 2
    gamma = 4
    ks = np.array([0, 3], dtype=np.int64)
    rng = np.random.default_rng(SEED_CHUNK + 9)
    prompt_1 = torch.tensor(
        rng.choice(ACGTN_IDS, size=PREFIX_LEN), dtype=torch.long, device="cuda:0"
    )[None]
    drafts = torch.tensor(
        rng.choice(ACGTN_IDS, size=(2, gamma + 1)), dtype=torch.long, device="cuda:0"
    )
    drafts[:, 0] = prompt_1[0, -1]
    try:
        with torch.inference_mode():
            ip1 = mini_model.initialize_inference_params(max_seqlen=PREFIX_LEN + 64)
            prefill(mini_model, prompt_1[:, :-1], ip1)
            ip_b = clone_inference_params(ip1)
            expand_inference_params(ip_b, 2)
            _, stash = block_forward(mini_model, drafts, ip_b, retain=True)
            # 迷你模型是慢路径（use_flash_attn=False），变长切片会触发守卫；
            # 本测试只拍 Hyena 态、随后即 collapse，不变长读 KV，显式旁路
            stash.flash_attn = True
            slice_states_to_accept(ip_b, stash, ks)
            snap_b = snapshot_states(
                ip_b, kv_len=PREFIX_LEN - 1 + int(ks.max()) + 1, batch_size=2
            )
            for b, kb in enumerate(ks.tolist()):
                ip_ref = clone_inference_params(ip1)
                block_forward(mini_model, drafts[b : b + 1, : kb + 1], ip_ref)
                snap_r = snapshot_states(ip_ref, kv_len=PREFIX_LEN - 1 + kb + 1, batch_size=1)

                def _cmp(a, bdict, path):
                    if isinstance(a, dict):
                        for kk in a:
                            _cmp(a[kk], bdict[kk], f"{path}/{kk}")
                    else:
                        ta = a[b] if a.shape[0] == 2 else a
                        d = float((ta.float() - bdict.float()).abs().max())
                        assert d < TOL, f"b={b} k={kb} {path} max|diff|={d:.3e}"

                for key in ("hcl", "hcm", "hcs"):
                    _cmp(snap_b[key], snap_r[key], key)
            # 收回 k 较短的胜出路径 0：offset 须收到 L0+k+1，不是 max(k)
            collapse_inference_params(ip_b, 0)
            want = PREFIX_LEN - 1 + int(ks[0]) + 1
            assert get_seqlen_offset(ip_b) == want, (
                f"collapse 后 offset={get_seqlen_offset(ip_b)} ≠ 胜出长度 {want}"
            )
            ip_ref0 = clone_inference_params(ip1)
            block_forward(mini_model, drafts[0:1, : int(ks[0]) + 1], ip_ref0)
            snap_c = snapshot_states(ip_b, kv_len=want, batch_size=1)
            snap_r0 = snapshot_states(ip_ref0, kv_len=want, batch_size=1)

            def _cmp1(a, bdict, path):
                if isinstance(a, dict):
                    for kk in a:
                        _cmp1(a[kk], bdict[kk], f"{path}/{kk}")
                else:
                    d = float((a.float() - bdict.float()).abs().max())
                    assert d < TOL, f"collapse b=0 {path} max|diff|={d:.3e}"

            for key in ("hcl", "hcm", "hcs"):
                _cmp1(snap_c[key], snap_r0[key], key)
    finally:
        mini_model.config.max_batch_size = prev


def test_greedy_short_prefix_slice(mini_model):
    """64-token 前缀（HCM 短窗）slice 循环贪心 vs 原生逐步。"""
    import torch

    from evspark.specdec.block.loop import native_greedy_reference, speculative_generate
    from evspark.specdec.drafts import IdentityDraftModel

    prefix = 64
    gamma = 4
    n_tokens = 32
    rng = np.random.default_rng(SEED_PROMPT + 64)
    prompt = _prompt_ids(torch, rng, prefix)
    native, _ip, _ = native_greedy_reference(mini_model, prompt, n_tokens)
    spec = speculative_generate(
        mini_model,
        IdentityDraftModel(),
        prompt,
        n_tokens,
        gamma,
        greedy=True,
        rng=np.random.default_rng(0),
        rollback="slice",
    )
    np.testing.assert_array_equal(spec.emitted_ids, native)
    assert any(r.k < gamma for r in spec.rounds_log)


def test_slice_vs_replay_short_prefix(mini_model):
    """前缀 64 < HCM 窗：短 cat 切片 vs 重放。"""
    import torch

    from evspark.specdec.block.driver import (
        block_forward,
        clone_inference_params,
        get_seqlen_offset,
        prefill,
        snapshot_states,
    )
    from evspark.specdec.block.slice import slice_states_to_accept

    prefix = 64
    gamma = 4
    rng = np.random.default_rng(SEED_CHUNK + 11)
    prompt = _prompt_ids(torch, rng, prefix)
    chunk = _prompt_ids(torch, rng, gamma + 1)
    with torch.inference_mode():
        ip0 = mini_model.initialize_inference_params(max_seqlen=prefix + 64)
        prefill(mini_model, prompt, ip0)
        snap = clone_inference_params(ip0)
        ip_full = clone_inference_params(ip0)
        _logits, stash = block_forward(mini_model, chunk, ip_full, retain=True)
        assert stash.chunk_len == gamma + 1
        hcm_cat = next(c for i, c in stash.x1v_cat.items() if stash.k_inner[i] >= 128)
        assert hcm_cat.shape[-1] == prefix + gamma + 1  # S=64 < 127，cat 未补零
        for k in (0, 1, 2, gamma):
            ip_slice = clone_inference_params(ip_full)
            slice_states_to_accept(ip_slice, stash, k)
            ip_replay = clone_inference_params(snap)
            block_forward(mini_model, chunk[:, : k + 1], ip_replay)
            kv_len = prefix + k + 1
            assert get_seqlen_offset(ip_slice) == kv_len
            _assert_states_close(
                snapshot_states(ip_slice, kv_len=kv_len),
                snapshot_states(ip_replay, kv_len=kv_len),
                f"短前缀 k={k} 切片 vs 重放",
            )


def test_slice_per_sequence_slow_attention_guard(mini_model):
    """慢路径（use_flash_attn=False）下变长 k 切片必须被拒绝（stale KV 风险）。"""
    import torch

    from evspark.specdec.block.driver import (
        block_forward,
        clone_inference_params,
        expand_inference_params,
        prefill,
    )
    from evspark.specdec.block.slice import slice_states_to_accept
    from test_block_forward import ACGTN_IDS, PREFIX_LEN

    prev = mini_model.config.max_batch_size
    mini_model.config.max_batch_size = 2
    try:
        rng = np.random.default_rng(SEED_CHUNK + 21)
        prompt = torch.tensor(
            rng.choice(ACGTN_IDS, size=(1, PREFIX_LEN)), dtype=torch.long, device="cuda:0"
        )
        chunk = torch.tensor(
            rng.choice(ACGTN_IDS, size=(2, GAMMA + 1)), dtype=torch.long, device="cuda:0"
        )
        with torch.inference_mode():
            ip = mini_model.initialize_inference_params(max_seqlen=PREFIX_LEN + 64)
            prefill(mini_model, prompt[:, :-1], ip)
            expand_inference_params(ip, 2)
            _, stash = block_forward(mini_model, chunk, ip, retain=True)
            assert stash.flash_attn is False  # 迷你模型慢路径
            with pytest.raises(ValueError, match="flash_attn"):
                slice_states_to_accept(ip, stash, np.array([0, 3], dtype=np.int64))
            # 标量 k（一致）不受守卫限制
            slice_states_to_accept(ip, stash, 2)
    finally:
        mini_model.config.max_batch_size = prev


def test_slice_per_sequence_hcs_width_mismatch_raises(mini_model):
    """前缀短于 HCS 窗且变长切片宽度不一：非 flip 核禁止零填，须报错。"""
    import torch

    from evspark.specdec.block.driver import (
        block_forward,
        clone_inference_params,
        expand_inference_params,
        prefill,
    )
    from evspark.specdec.block.slice import slice_states_to_accept
    from test_block_forward import ACGTN_IDS

    prev = mini_model.config.max_batch_size
    mini_model.config.max_batch_size = 2
    prefix = 4  # HCS（K=7）S=3 < 6；外层满窗不受影响
    gamma = 4
    try:
        rng = np.random.default_rng(SEED_CHUNK + 22)
        prompt = torch.tensor(
            rng.choice(ACGTN_IDS, size=(1, prefix)), dtype=torch.long, device="cuda:0"
        )
        chunk = torch.tensor(
            rng.choice(ACGTN_IDS, size=(2, gamma + 1)), dtype=torch.long, device="cuda:0"
        )
        with torch.inference_mode():
            ip = mini_model.initialize_inference_params(max_seqlen=prefix + 64)
            prefill(mini_model, prompt[:, :-1], ip)
            expand_inference_params(ip, 2)
            _, stash = block_forward(mini_model, chunk, ip, retain=True)
            stash.flash_attn = True  # 绕过慢路径守卫，专测 HCS 变宽分支
            with pytest.raises(ValueError, match="不同 FIR 状态长度"):
                slice_states_to_accept(ip, stash, np.array([0, 2], dtype=np.int64))
    finally:
        mini_model.config.max_batch_size = prev


def test_slice_per_sequence_hcm_zeropad(mini_model):
    """HCM（flip）变长切片宽度不一左零填：前导零 + 尾部=独立重放 + 下游 ≡ 真短态。"""
    import torch

    from evspark.specdec.block.driver import (
        block_forward,
        clone_inference_params,
        collapse_inference_params,
        expand_inference_params,
        get_seqlen_offset,
        prefill,
        step_forward_reference,
    )
    from evspark.specdec.block.slice import slice_states_to_accept
    from test_block_forward import ACGTN_IDS

    prev = mini_model.config.max_batch_size
    mini_model.config.max_batch_size = 2
    prefix = 64  # prefill 63 位 → HCM S=63 < 127；HCS/外层满窗
    gamma = 4
    ks = np.array([0, 3], dtype=np.int64)
    rng = np.random.default_rng(SEED_CHUNK + 31)
    prompt = torch.tensor(
        rng.choice(ACGTN_IDS, size=(1, prefix)), dtype=torch.long, device="cuda:0"
    )
    drafts = torch.tensor(
        rng.choice(ACGTN_IDS, size=(2, gamma + 1)), dtype=torch.long, device="cuda:0"
    )
    drafts[:, 0] = prompt[0, -1]
    try:
        with torch.inference_mode():
            ip1 = mini_model.initialize_inference_params(max_seqlen=prefix + 64)
            prefill(mini_model, prompt[:, :-1], ip1)  # L0 = 63
            ip_b = clone_inference_params(ip1)
            expand_inference_params(ip_b, 2)
            _, stash = block_forward(mini_model, drafts, ip_b, retain=True)
            stash.flash_attn = True  # 迷你模型慢路径；只拍 Hyena 态 + collapse 后单序列下游
            slice_states_to_accept(ip_b, stash, ks)

            layer_idx = next(i for i, kk in stash.k_inner.items() if kk >= 128)
            state = ip_b["hcm"].fir_inner_state_dict[layer_idx]
            width0 = 63 + int(ks[0]) + 1  # 64
            width1 = 63 + int(ks[1]) + 1  # 67
            assert int(state.shape[-1]) == width1, "变长须零填到对齐宽度"
            pad = width1 - width0
            assert float(state[0, ..., :pad].abs().max()) == 0.0, "前导须为零填"
            # 尾部窗口与独立重放（真短态）逐项一致
            for b, kb in enumerate(ks.tolist()):
                ip_ref = clone_inference_params(ip1)
                block_forward(mini_model, drafts[b : b + 1, : kb + 1], ip_ref)
                ref = ip_ref["hcm"].fir_inner_state_dict[layer_idx]
                w = 63 + kb + 1
                assert int(ref.shape[-1]) == w
                d = float((state[b, ..., -w:].float() - ref[0].float()).abs().max())
                assert d < TOL, f"b={b} k={kb} HCM 零填窗口尾部 {w} 位 max|diff|={d:.3e}"
            # 下游等价：collapse 到被零填的胜出路径 0 再走一块，
            # 与「真短态 → 逐步」参照的 logits 全等 ⇒ 零填 ≡ 真短态
            collapse_inference_params(ip_b, 0)
            assert get_seqlen_offset(ip_b) == 63 + int(ks[0]) + 1
            chunk2 = torch.tensor(
                rng.choice(ACGTN_IDS, size=(1, gamma)), dtype=torch.long, device="cuda:0"
            )
            ip_pad_step = clone_inference_params(ip_b)
            logits_pad = block_forward(mini_model, chunk2, ip_b)
            logits_pad_step = step_forward_reference(mini_model, chunk2, ip_pad_step)
            d = float((logits_pad.float() - logits_pad_step.float()).abs().max())
            assert d < TOL, f"零填态块前向 vs 逐步 max|diff|={d:.3e}"
            ip_true = clone_inference_params(ip1)
            block_forward(mini_model, drafts[0:1, : int(ks[0]) + 1], ip_true)
            logits_true = step_forward_reference(mini_model, chunk2, ip_true)
            d2 = float((logits_pad.float() - logits_true.float()).abs().max())
            assert d2 < TOL, f"零填态下游 vs 真短态逐步 max|diff|={d2:.3e}"
    finally:
        mini_model.config.max_batch_size = prev


def test_collapse_nonzero_winner(mini_model):
    """D4b：多候选切片后 collapse 到 winner=1，KV 行拷贝与 Hyena 态对拍 B=1 参照。"""
    import torch

    from evspark.specdec.block.driver import (
        block_forward,
        clone_inference_params,
        collapse_inference_params,
        expand_inference_params,
        get_seqlen_offset,
        prefill,
        snapshot_states,
    )
    from evspark.specdec.block.slice import slice_states_to_accept
    from test_block_forward import ACGTN_IDS, PREFIX_LEN

    prev = mini_model.config.max_batch_size
    mini_model.config.max_batch_size = 2
    gamma = 4
    k = 2
    rng = np.random.default_rng(SEED_CHUNK + 41)
    prompt = torch.tensor(
        rng.choice(ACGTN_IDS, size=(1, PREFIX_LEN)), dtype=torch.long, device="cuda:0"
    )
    drafts = torch.tensor(
        rng.choice(ACGTN_IDS, size=(2, gamma + 1)), dtype=torch.long, device="cuda:0"
    )
    drafts[:, 0] = prompt[0, -1]
    try:
        with torch.inference_mode():
            ip1 = mini_model.initialize_inference_params(max_seqlen=PREFIX_LEN + 64)
            prefill(mini_model, prompt[:, :-1], ip1)
            ip_b = clone_inference_params(ip1)
            expand_inference_params(ip_b, 2)
            _, stash = block_forward(mini_model, drafts, ip_b, retain=True)
            slice_states_to_accept(ip_b, stash, k)  # 标量：慢路径无风险
            collapse_inference_params(ip_b, 1)
            kv_len = PREFIX_LEN - 1 + k + 1
            assert get_seqlen_offset(ip_b) == kv_len
            ip_ref = clone_inference_params(ip1)
            block_forward(mini_model, drafts[1:2, : k + 1], ip_ref)
            _assert_states_close(
                snapshot_states(ip_b, kv_len=kv_len),
                snapshot_states(ip_ref, kv_len=kv_len),
                "collapse winner=1 全状态（含 KV 行拷贝）",
            )
    finally:
        mini_model.config.max_batch_size = prev
