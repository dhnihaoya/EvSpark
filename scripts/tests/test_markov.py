"""k 阶 Markov drafter 单测。对应 plans/04 §4。"""

from __future__ import annotations

import numpy as np

from specdec.genome import clean_seq
from specdec.harness import run_speculative
from specdec.markov import (
    BYTE_A,
    BYTE_C,
    BYTE_G,
    BYTE_N,
    BYTE_T,
    MASS_IDS,
    VOCAB_SIZE,
    MarkovDraftModel,
    MarkovTable,
    byte_id_to_sym,
    context_code,
    split_contiguous,
)
from specdec.mock_models import _normalize_row, _normalize_rows

SEED_HARNESS_SMOKE = 20260820

MASS = np.array(MASS_IDS, dtype=np.int64)


def _hand_counts_acgtac_k1() -> np.ndarray:
    """人工计数：ACGTAC、k=1。A→C×2, C→G, G→T, T→A。"""
    # 内部 id: A=0 C=1 G=2 T=3 N=4
    counts = np.zeros((5, 5), dtype=np.int64)
    counts[0, 1] = 2  # A→C
    counts[1, 2] = 1  # C→G
    counts[2, 3] = 1  # G→T
    counts[3, 0] = 1  # T→A
    return counts


def test_counts_handwritten_acgtac_k1():
    table = MarkovTable.build(["ACGTAC"], k=1, alpha=0.1)
    expected = _hand_counts_acgtac_k1()
    np.testing.assert_array_equal(table.counts, expected)
    # 清洗幂等：小写应得同一计数
    table_lc = MarkovTable.build(["acgtac"], k=1, alpha=0.1)
    np.testing.assert_array_equal(table_lc.counts, expected)


def test_normalization_and_unseen_uniform():
    alpha = 0.1
    table = MarkovTable.build(["AAA"], k=2, alpha=alpha)
    row_sums = table.probs.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, rtol=0.0, atol=1e-15)
    # AAA、k=2：仅 AA→A 出现一次；其余 24 个上下文未见 → 均匀 1/5
    aa = context_code(np.array([0, 0], dtype=np.int64))
    seen = table.probs[aa]
    expected_seen = (np.array([1.0, 0, 0, 0, 0]) + alpha) / (1.0 + alpha * 5)
    np.testing.assert_allclose(seen, expected_seen, rtol=0.0, atol=1e-15)
    unseen_mask = np.ones(table.probs.shape[0], dtype=bool)
    unseen_mask[aa] = False
    np.testing.assert_allclose(table.probs[unseen_mask], 0.2, rtol=0.0, atol=1e-15)
    np.testing.assert_array_equal(table.counts[unseen_mask], 0)


def test_build_deterministic():
    seqs = ["ACGTNACGT", "TTGCANttgc"]
    a = MarkovTable.build(seqs, k=3, alpha=0.1)
    b = MarkovTable.build(seqs, k=3, alpha=0.1)
    np.testing.assert_array_equal(a.counts, b.counts)
    np.testing.assert_array_equal(a.probs, b.probs)


def test_adapter_vocab_mass_and_table():
    table = MarkovTable.build(["ACGTACGTACGT"], k=3, alpha=0.1)
    model = MarkovDraftModel(table)

    def check_prefix(prefix: np.ndarray, ctx_ids: np.ndarray) -> np.ndarray:
        out = model.probs(prefix)
        assert out.shape == (VOCAB_SIZE,)
        np.testing.assert_allclose(out.sum(), 1.0, rtol=0.0, atol=1e-15)
        other = np.ones(VOCAB_SIZE, dtype=bool)
        other[MASS] = False
        np.testing.assert_array_equal(out[other], 0.0)
        code = context_code(ctx_ids)
        np.testing.assert_array_equal(out[MASS], table.probs[code])
        return out

    # ACGT → 上下文 CGT = (1,2,3) → 1*25+2*5+3 = 38
    p_acgt = check_prefix(
        np.array([BYTE_A, BYTE_C, BYTE_G, BYTE_T], dtype=np.int64),
        np.array([1, 2, 3], dtype=np.int64),
    )
    # 小写等价
    p_lc = model.probs(np.array([97, 99, 103, 116], dtype=np.int64))
    np.testing.assert_array_equal(p_lc, p_acgt)
    # A|C → A, N, C = (0,4,1)
    check_prefix(
        np.array([BYTE_A, ord("|"), BYTE_C], dtype=np.int64),
        np.array([0, 4, 1], dtype=np.int64),
    )
    # 含 N
    check_prefix(
        np.array([BYTE_N, BYTE_A, BYTE_C], dtype=np.int64),
        np.array([4, 0, 1], dtype=np.int64),
    )
    # 短前缀左填 N：仅 A → NNA = (4,4,0)
    check_prefix(
        np.array([BYTE_A], dtype=np.int64),
        np.array([4, 4, 0], dtype=np.int64),
    )
    # 空前缀 → NNN
    check_prefix(np.array([], dtype=np.int64), np.array([4, 4, 4], dtype=np.int64))
    # 越界 byte → N
    assert byte_id_to_sym(-1) == 4
    assert byte_id_to_sym(512) == 4
    assert byte_id_to_sym(ord("|")) == 4
    assert byte_id_to_sym(ord("a")) == 0


class _AcgtnByteBigram:
    """测试用 512 维 mock target：质量只放在 5 个 byte id 上的一阶转移。"""

    def __init__(self, transitions: np.ndarray, start: np.ndarray):
        trans = np.asarray(transitions, dtype=np.float64)
        if trans.shape != (5, 5):
            raise ValueError("transitions 须为 [5,5]")
        self.transitions = _normalize_rows(trans)
        self.start = _normalize_row(start)
        self._inv = {int(b): i for i, b in enumerate(MASS_IDS)}

    def probs(self, prefix: np.ndarray) -> np.ndarray:
        prefix = np.asarray(prefix, dtype=np.int64).ravel()
        out = np.zeros(VOCAB_SIZE, dtype=np.float64)
        if prefix.size == 0:
            row = self.start
        else:
            idx = self._inv.get(int(prefix[-1]), 4)
            row = self.transitions[idx]
        out[MASS] = row
        return out


def test_harness_integration_smoke():
    rng = np.random.default_rng(SEED_HARNESS_SMOKE)
    table = MarkovTable.build(["ACGTACGTN" * 30], k=3, alpha=0.1)
    draft = MarkovDraftModel(table)
    target = _AcgtnByteBigram(
        rng.dirichlet(np.ones(5), size=5),
        rng.dirichlet(np.ones(5)),
    )
    prompt = np.array([BYTE_A, BYTE_C, BYTE_G, BYTE_T], dtype=np.int64)
    n_tokens = 500
    gamma = 4
    tokens, results = run_speculative(
        target, draft, prompt, n_tokens, gamma, rng, greedy=False
    )
    assert tokens.shape == (n_tokens,)
    assert len(results) > 0
    allowed = set(int(x) for x in MASS_IDS)
    for tok in tokens:
        assert int(tok) in allowed
    for r in results:
        assert int(r.tokens.shape[0]) == r.accepted_len + 1
        assert 0 <= r.accepted_len <= gamma
        assert r.accepted_mask.shape == (gamma,)
        for tok in r.tokens:
            assert int(tok) in allowed
    print(
        f"[markov-harness] n_tokens={n_tokens} gamma={gamma} n_rounds={len(results)} "
        f"seed={SEED_HARNESS_SMOKE}"
    )


def test_clean_seq_and_contiguous_split():
    assert clean_seq("acgt\nRYN") == "ACGTNNN"
    assert clean_seq("A C G T") == "ACGT"
    train, test = split_contiguous("ACGTACGTAC", test_frac=0.1)
    assert train + test == "ACGTACGTAC"
    assert len(test) == 1  # int(10*0.9)=9
    # 确认不是 shuffle：切分点固定在尾部
    assert test == "C"
    assert train == "ACGTACGTA"


def test_clean_iupac_and_pipe():
    assert clean_seq("A|C") == "ANC"
    assert clean_seq("") == ""
