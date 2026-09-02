"""Step 5（Plan 02 A.6/A.5，plans/07 §5.1）投机循环：恒等 drafter + 快照回滚。

CPU：IdentityDraftModel 单测（无 torch 也可跑）。
GPU：复用 Step 4 迷你 StripedHyena（fp32，8 层）做贪心全等、状态不变量、采样 KL。
无 torch/vortex/CUDA 时 GPU 项 skip，不影响 CPU 套件。
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from specdec.drafts import MASS_IDS, VOCAB_SIZE, IdentityDraftModel

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

SEED_PROMPT = 20260823
SEED_SAMPLE_NATIVE = 7001
SEED_SAMPLE_SPEC = 7002
N_GREEDY = 64
N_SAMPLE = 20_000
KL_MAX = 1e-3
TOL = 1e-4
GREEDY_GAMMAS = (2, 4, 8)


# ---------------------------------------------------------------------------
# CPU：IdentityDraftModel
# ---------------------------------------------------------------------------


def test_identity_empty_prefix_uniform_acgtn():
    draft = IdentityDraftModel()
    q = draft.probs(np.empty(0, dtype=np.int64))
    assert q.shape == (VOCAB_SIZE,)
    np.testing.assert_allclose(q.sum(), 1.0, atol=1e-15)
    mass = np.array(MASS_IDS, dtype=np.int64)
    np.testing.assert_allclose(q[mass], 0.2)
    others = np.ones(VOCAB_SIZE, dtype=bool)
    others[mass] = False
    assert np.all(q[others] == 0.0)


def test_identity_one_hot_last_token():
    draft = IdentityDraftModel()
    for tok in (65, 67, 71, 84, 78, 0, 511):
        q = draft.probs(np.array([1, 2, tok], dtype=np.int64))
        expected = np.zeros(VOCAB_SIZE, dtype=np.float64)
        expected[tok] = 1.0
        np.testing.assert_array_equal(q, expected)
        assert int(np.argmax(q)) == tok


# ---------------------------------------------------------------------------
# GPU：迷你模型
# ---------------------------------------------------------------------------


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


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    p = p / p.sum()
    q = q / q.sum()
    kl = 0.0
    for pi, qi in zip(p, q):
        if pi > 0.0:
            kl += float(pi * np.log(pi / qi))
    return kl


def _unigram(seq: np.ndarray, vocab: int) -> np.ndarray:
    return np.bincount(seq, minlength=vocab).astype(np.float64)


def _bigram_conditionals(seq: np.ndarray, start: int, vocab: int) -> tuple[np.ndarray, np.ndarray]:
    joint = np.zeros((vocab, vocab), dtype=np.float64)
    prev = start
    for tok in seq:
        joint[prev, int(tok)] += 1.0
        prev = int(tok)
    cond = np.zeros_like(joint)
    row_sum = joint.sum(axis=1, keepdims=True)
    nz = row_sum.ravel() > 0
    cond[nz] = joint[nz] / row_sum[nz]
    return joint, cond


def _weighted_conditional_kl_smoothed(joint_p: np.ndarray, joint_q: np.ndarray) -> float:
    """行加权条件 KL；每行 add-1，避免未出现转移把 KL 打成 inf。"""
    w = joint_q.sum(axis=1).astype(np.float64)
    if w.sum() <= 0:
        return 0.0
    w = w / w.sum()
    total = 0.0
    for i, wi in enumerate(w):
        if wi <= 0.0:
            continue
        total += wi * _kl(_smoothed_counts(joint_p[i]), _smoothed_counts(joint_q[i]))
    return float(total)


def _smoothed_counts(counts: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    c = counts.astype(np.float64) + alpha
    return c / c.sum()


class _OracleGreedyDraft:
    """把已知贪心续延做成 one-hot，用于覆盖全接受（跳过重放）路径。"""

    def __init__(self, prompt_len: int, continuation: np.ndarray):
        self.prompt_len = int(prompt_len)
        self.continuation = np.asarray(continuation, dtype=np.int64).ravel()

    def probs(self, prefix: np.ndarray) -> np.ndarray:
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


@pytest.mark.parametrize("gamma", GREEDY_GAMMAS)
def test_greedy_tokenwise_equal(mini_model, gamma: int):
    import torch

    from specdec.block.loop import native_greedy_reference, speculative_generate
    from specdec.drafts import IdentityDraftModel
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
    )
    np.testing.assert_array_equal(spec.emitted_ids, native)
    assert any(r.k < gamma for r in spec.rounds_log), "恒等 drafter 应覆盖拒绝+回滚"


def test_state_invariant_after_rejects(mini_model):
    import torch

    from specdec.block.driver import diff_states, get_seqlen_offset, snapshot_states
    from specdec.block.loop import (
        spec_end_kv_len,
        speculative_generate,
        teacher_force_to_spec_state,
    )
    from specdec.drafts import IdentityDraftModel
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
    )
    assert any(r.k < gamma for r in spec.rounds_log)
    kv_len = spec_end_kv_len(PREFIX_LEN, N_GREEDY)
    assert get_seqlen_offset(spec.inference_params_dict) == kv_len

    snap_spec = snapshot_states(spec.inference_params_dict, kv_len=kv_len)
    ip_ref = teacher_force_to_spec_state(mini_model, prompt, spec.emitted_ids)
    assert get_seqlen_offset(ip_ref) == kv_len
    snap_ref = snapshot_states(ip_ref, kv_len=kv_len)
    diffs = diff_states(snap_spec, snap_ref)
    worst = max(diffs.items(), key=lambda kv: kv[1]["max_abs"])
    assert worst[1]["max_abs"] < TOL, f"状态最大差 {worst[0]}: {worst[1]['max_abs']:.3e}"


def test_full_accept_skips_replay(mini_model):
    """oracle greedy draft → 每轮 k==γ，replay_len==0，输出仍与原生全等。"""
    import torch

    from specdec.block.driver import diff_states, get_seqlen_offset, snapshot_states
    from specdec.block.loop import (
        native_greedy_reference,
        spec_end_kv_len,
        speculative_generate,
        teacher_force_to_spec_state,
    )
    from test_block_forward import PREFIX_LEN

    gamma = 4
    rng = np.random.default_rng(SEED_PROMPT + 7)
    prompt = _prompt_ids(torch, rng, PREFIX_LEN)
    native, _ip, _ = native_greedy_reference(mini_model, prompt, N_GREEDY)
    draft = _OracleGreedyDraft(PREFIX_LEN, native)
    spec = speculative_generate(
        mini_model,
        draft,
        prompt,
        N_GREEDY,
        gamma,
        greedy=True,
        rng=np.random.default_rng(0),
    )
    np.testing.assert_array_equal(spec.emitted_ids, native)
    n_full = sum(1 for r in spec.rounds_log if r.k == gamma)
    n_skip = sum(1 for r in spec.rounds_log if r.replay_len == 0)
    assert n_full >= 1 and n_skip >= 1, f"full={n_full} skip={n_skip}"
    kv_len = spec_end_kv_len(PREFIX_LEN, N_GREEDY)
    assert get_seqlen_offset(spec.inference_params_dict) == kv_len
    snap_spec = snapshot_states(spec.inference_params_dict, kv_len=kv_len)
    ip_ref = teacher_force_to_spec_state(mini_model, prompt, spec.emitted_ids)
    diffs = diff_states(snap_spec, snapshot_states(ip_ref, kv_len=kv_len))
    worst = max(diffs.items(), key=lambda kv: kv[1]["max_abs"])
    assert worst[1]["max_abs"] < TOL, f"状态最大差 {worst[0]}: {worst[1]['max_abs']:.3e}"


def test_sampling_lossless_kl(mini_model):
    import torch

    from specdec.block.loop import native_sample_reference, speculative_generate
    from specdec.drafts import IdentityDraftModel
    from test_block_forward import PREFIX_LEN

    n_chains, chain_len = 20, 1_000
    gamma = 4
    native_parts: list[np.ndarray] = []
    spec_parts: list[np.ndarray] = []
    starts: list[int] = []
    rng_prompt = np.random.default_rng(SEED_PROMPT)
    rng_native = np.random.default_rng(SEED_SAMPLE_NATIVE)
    rng_spec = np.random.default_rng(SEED_SAMPLE_SPEC)
    draft = IdentityDraftModel()
    for c in range(n_chains):
        prompt = _prompt_ids(torch, rng_prompt, PREFIX_LEN)
        starts.append(int(prompt[0, -1].item()))
        native_parts.append(
            native_sample_reference(
                mini_model,
                prompt,
                chain_len,
                rng_native,
                temperature=1.0,
                top_k=4,
            )
        )
        spec = speculative_generate(
            mini_model,
            draft,
            prompt,
            chain_len,
            gamma,
            greedy=False,
            rng=rng_spec,
            temperature=1.0,
            top_k=4,
        )
        spec_parts.append(spec.emitted_ids)
        del spec
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    native = np.concatenate(native_parts)
    spec_ids = np.concatenate(spec_parts)
    assert native.size == spec_ids.size == n_chains * chain_len
    vocab = VOCAB_SIZE

    def _pool_joint(parts: list[np.ndarray], idx=None) -> np.ndarray:
        joint = np.zeros((vocab, vocab), dtype=np.float64)
        it = range(len(parts)) if idx is None else idx
        for i in it:
            j, _ = _bigram_conditionals(parts[i], starts[i], vocab)
            joint += j
        return joint

    uni_n = _unigram(native, vocab)
    uni_s = _unigram(spec_ids, vocab)
    kl_uni = _kl(_smoothed_counts(uni_s), _smoothed_counts(uni_n))
    joint_n = _pool_joint(native_parts)
    joint_s = _pool_joint(spec_parts)
    kl_joint = _kl(_smoothed_counts(joint_s.ravel()), _smoothed_counts(joint_n.ravel()))
    kl_cond = _weighted_conditional_kl_smoothed(joint_s, joint_n)
    floor = (vocab - 1) / (2.0 * native.size)

    # 把 20 条链拆成奇偶两组，得到同 N 的原生–原生噪声地板（不再额外采样）
    even = list(range(0, n_chains, 2))
    odd = list(range(1, n_chains, 2))
    uni_ne = _unigram(np.concatenate([native_parts[i] for i in even]), vocab)
    uni_no = _unigram(np.concatenate([native_parts[i] for i in odd]), vocab)
    uni_se = _unigram(np.concatenate([spec_parts[i] for i in even]), vocab)
    kl_uni_nn = _kl(_smoothed_counts(uni_ne), _smoothed_counts(uni_no))
    kl_uni_sn = _kl(_smoothed_counts(uni_se), _smoothed_counts(uni_ne))
    jn_e, jn_o = _pool_joint(native_parts, even), _pool_joint(native_parts, odd)
    js_e = _pool_joint(spec_parts, even)
    kl_joint_nn = _kl(_smoothed_counts(jn_e.ravel()), _smoothed_counts(jn_o.ravel()))
    kl_joint_sn = _kl(_smoothed_counts(js_e.ravel()), _smoothed_counts(jn_e.ravel()))
    kl_cond_nn = _weighted_conditional_kl_smoothed(jn_e, jn_o)
    kl_cond_sn = _weighted_conditional_kl_smoothed(js_e, jn_e)

    print(
        f"[5.1.3] N={native.size} chains={n_chains} gamma={gamma} "
        f"KL_uni spec/nat={kl_uni:.6e} floor≈{floor:.6e} "
        f"split KL_uni spec/nat={kl_uni_sn:.6e} nat/nat={kl_uni_nn:.6e} "
        f"KL_joint spec/nat={kl_joint:.6e} split spec/nat={kl_joint_sn:.6e} nat/nat={kl_joint_nn:.6e} "
        f"KL_cond spec/nat={kl_cond:.6e} split spec/nat={kl_cond_sn:.6e} nat/nat={kl_cond_nn:.6e} "
        f"seed_native={SEED_SAMPLE_NATIVE} seed_spec={SEED_SAMPLE_SPEC}"
    )
    # V=512、N=20k 时 add-1 经验 KL 噪声地板 ≈ (V-1)/(2N) ~ 1.3e-2，
    # 计划 1e-3 是 Step 1（V=4、N=2e5）口径，此处达不到。
    # 断言：全量 KL 与地板同量级；奇偶拆分下投机–原生不超过原生–原生的 2 倍。
    slack = 1e-3
    assert kl_uni < 5.0 * floor + slack, f"unigram KL {kl_uni} ≫ 噪声地板 {floor}"
    assert kl_uni_sn <= 2.0 * kl_uni_nn + slack, (
        f"split unigram KL spec/nat={kl_uni_sn} vs nat/nat={kl_uni_nn}"
    )
    assert kl_joint_sn <= 2.0 * kl_joint_nn + slack, (
        f"split bigram joint KL spec/nat={kl_joint_sn} vs nat/nat={kl_joint_nn}"
    )
    assert kl_cond_sn <= 2.0 * kl_cond_nn + slack, (
        f"split bigram cond KL spec/nat={kl_cond_sn} vs nat/nat={kl_cond_nn}"
    )
