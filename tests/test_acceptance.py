"""Step 6（Plan 02 B.3，plans/08 §3）draft 侧 q′ 变换 + Markov 贪心回归。

无 torch 时 skip（本文件测试 loop.py，该模块引入 torch）；
无 CUDA 时再 skip GPU 项。不影响既有 30 项 CPU 套件。
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from evspark.specdec.drafts import VOCAB_SIZE
from evspark.specdec.markov import MarkovDraftModel, MarkovTable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

SEED_PROPOSE = 20260821
SEED_PROMPT = 20260823
N_GREEDY = 64
GAMMA_CPU = 8
TOP_K = 4

_TRAIN_SEQ = "ACGTACGTNACGTACGTNACGTACGT"


def _markov_draft() -> MarkovDraftModel:
    table = MarkovTable.build([_TRAIN_SEQ], k=3, alpha=0.1)
    return MarkovDraftModel(table)


def _assert_q_prime_row(row: np.ndarray, token: int) -> None:
    row = np.asarray(row, dtype=np.float64).ravel()
    assert row.shape == (VOCAB_SIZE,)
    nnz = int(np.count_nonzero(row > 0.0))
    assert nnz <= TOP_K, f"q′ 非零数 {nnz} > top_k={TOP_K}"
    np.testing.assert_allclose(row.sum(), 1.0, atol=1e-12)
    assert 0 <= int(token) < VOCAB_SIZE
    assert row[int(token)] > 0.0, "draft token 必须落在 q′ 支撑集内"


# ---------------------------------------------------------------------------
# CPU（需 torch 以 import loop.py，不需 CUDA）
# ---------------------------------------------------------------------------


def test_q_prime_sampling_support():
    """采样路径：q′ 每行 ≤4 非零、和为 1、提出的 token 在支撑集内。"""
    torch = pytest.importorskip("torch", reason="loop.py 引入 torch")
    if not torch.cuda.is_available():
        # loop.py 经 vortex.model 在 import 时初始化 CUDA，无卡机器直接 skip
        pytest.skip("loop.py 引入 vortex.model 需要 CUDA")
    from evspark.specdec.block.loop import _propose_draft

    draft = _markov_draft()
    prefix = np.array([65, 67, 71, 84], dtype=np.int64)  # ACGT
    tokens, q = _propose_draft(
        draft,
        prefix,
        GAMMA_CPU,
        greedy=False,
        rng=np.random.default_rng(SEED_PROPOSE),
        temperature=1.0,
        top_k=TOP_K,
    )
    assert tokens.shape == (GAMMA_CPU,)
    assert q.shape == (GAMMA_CPU, VOCAB_SIZE)
    for i in range(GAMMA_CPU):
        _assert_q_prime_row(q[i], int(tokens[i]))
        # add-α Markov 原始质量在 5 个 ACGTN 上；截断后恰好 4 个非零
        assert int(np.count_nonzero(q[i] > 0.0)) == TOP_K


def test_greedy_propose_keeps_raw_q():
    """贪心路径不施加 q′：原始 Markov 行仍为 5 非零，草稿 = argmax(原始 q)。"""
    torch = pytest.importorskip("torch", reason="loop.py 引入 torch")
    if not torch.cuda.is_available():
        # loop.py 经 vortex.model 在 import 时初始化 CUDA，无卡机器直接 skip
        pytest.skip("loop.py 引入 vortex.model 需要 CUDA")
    from evspark.specdec.block.loop import _propose_draft, _q_prime_from_raw

    draft = _markov_draft()
    prefix = np.array([65, 67, 71, 84], dtype=np.int64)
    tokens, q = _propose_draft(
        draft,
        prefix,
        GAMMA_CPU,
        greedy=True,
        rng=np.random.default_rng(0),
        temperature=1.0,
        top_k=TOP_K,
    )
    buf = prefix.copy()
    for i in range(GAMMA_CPU):
        raw = np.asarray(draft.probs(buf), dtype=np.float64).ravel()
        np.testing.assert_array_equal(q[i], raw)
        assert int(np.count_nonzero(q[i] > 0.0)) == 5
        assert int(tokens[i]) == int(np.argmax(raw))
        q_prime = _q_prime_from_raw(raw, temperature=1.0, top_k=TOP_K)
        assert int(np.count_nonzero(q_prime > 0.0)) == TOP_K
        buf = np.concatenate([buf, np.array([int(tokens[i])], dtype=np.int64)])


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


def test_greedy_markov_tokenwise_equal(mini_model):
    """贪心 + Markov drafter：与逐步贪心逐 token 全等（q′ 变换不得影响贪心路径）。"""
    import torch

    from evspark.specdec.block.loop import native_greedy_reference, speculative_generate
    from test_block_forward import PREFIX_LEN

    rng = np.random.default_rng(SEED_PROMPT + 11)
    prompt = _prompt_ids(torch, rng, PREFIX_LEN)
    native, _ip, _ = native_greedy_reference(mini_model, prompt, N_GREEDY)
    spec = speculative_generate(
        mini_model,
        _markov_draft(),
        prompt,
        N_GREEDY,
        8,
        greedy=True,
        rng=np.random.default_rng(0),
        temperature=1.0,
        top_k=TOP_K,
    )
    np.testing.assert_array_equal(spec.emitted_ids, native)


def test_verify_receives_q_prime(mini_model, monkeypatch):
    """采样路径下 verify_round 收到的 draft_probs 就是 q′。"""
    import torch

    import evspark.specdec.block.loop as loop
    from evspark.specdec.block.loop import speculative_generate
    from test_block_forward import PREFIX_LEN

    captured: list[tuple[np.ndarray, np.ndarray]] = []
    orig = loop.verify_round

    def _wrapped(draft_tokens, draft_probs, target_probs, rng):
        captured.append(
            (
                np.asarray(draft_tokens, dtype=np.int64).copy(),
                np.asarray(draft_probs, dtype=np.float64).copy(),
            )
        )
        return orig(draft_tokens, draft_probs, target_probs, rng)

    monkeypatch.setattr(loop, "verify_round", _wrapped)

    rng = np.random.default_rng(SEED_PROMPT + 13)
    prompt = _prompt_ids(torch, rng, PREFIX_LEN)
    spec = speculative_generate(
        mini_model,
        _markov_draft(),
        prompt,
        16,
        4,
        greedy=False,
        rng=np.random.default_rng(SEED_PROPOSE + 1),
        temperature=1.0,
        top_k=TOP_K,
    )
    assert spec.emitted_ids.size == 16
    assert captured, "verify_round 应至少被调用一次"
    for tokens, q in captured:
        assert q.ndim == 2 and q.shape[1] == VOCAB_SIZE
        assert tokens.shape[0] == q.shape[0]
        for i in range(q.shape[0]):
            _assert_q_prime_row(q[i], int(tokens[i]))
            assert int(np.count_nonzero(q[i] > 0.0)) == TOP_K
