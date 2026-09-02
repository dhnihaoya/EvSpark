"""prompt-lookup drafter 单测。对应 plans/05 §5。"""

from __future__ import annotations

import math

import numpy as np

from specdec.harness import run_speculative
from specdec.lookup import (
    DEFAULT_LAMBDA,
    MASS,
    UNIFORM,
    VOCAB_SIZE,
    PromptLookupModel,
    prefix_to_acgtn_bytes,
    rfind_continuation,
)
from specdec.markov import (
    BYTE_A,
    BYTE_C,
    BYTE_G,
    BYTE_N,
    BYTE_T,
    MASS_IDS,
    MarkovDraftModel,
    MarkovTable,
)
from specdec.mock_models import _normalize_row, _normalize_rows
from specdec.verifier import sample_categorical

SEED_SAMPLE_Q = 20260821
SEED_HARNESS_SMOKE = 20260822
SEED_MECH_DEMO = 20260823

CHI2_PVALUE_MIN = 0.01
N_SAMPLE_Q = 10_000


def _ascii(s: str) -> np.ndarray:
    return np.frombuffer(s.encode("ascii"), dtype=np.uint8).astype(np.int64)


def _chi2_sf(chi2: float, df: int) -> float:
    if chi2 <= 0.0:
        return 1.0
    if df <= 0:
        raise ValueError("df 须为正")
    return _gammaincc(0.5 * df, 0.5 * chi2)


def _gammaincc(a: float, x: float) -> float:
    if x < 0.0 or a <= 0.0:
        return float("nan")
    if x == 0.0:
        return 1.0
    if x < a + 1.0:
        return max(0.0, min(1.0, 1.0 - _gamma_p_series(a, x)))
    return max(0.0, min(1.0, _gamma_q_cf(a, x)))


def _gamma_p_series(a: float, x: float, eps: float = 1e-15) -> float:
    term = 1.0 / a
    total = term
    ap = a
    for _ in range(10000):
        ap += 1.0
        term *= x / ap
        total += term
        if abs(term) < abs(total) * eps:
            break
    log_p = -x + a * math.log(x) - math.lgamma(a) + math.log(total)
    if log_p >= 0.0:
        return 1.0
    return math.exp(log_p)


def _gamma_q_cf(a: float, x: float, eps: float = 1e-15) -> float:
    fpmin = 1e-300
    b = x + 1.0 - a
    c = 1.0 / fpmin
    d = 1.0 / b if abs(b) > fpmin else 1.0 / fpmin
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < fpmin:
            d = fpmin
        c = b + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    log_q = -x + a * math.log(x) - math.lgamma(a) + math.log(abs(h))
    if log_q >= 0.0:
        return 1.0
    return math.exp(log_q)


def _pearson_chi2_pvalue(counts: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    counts = np.asarray(counts, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    p = p / p.sum()
    n = float(counts.sum())
    expected = n * p
    mask = expected >= 5.0
    if int(mask.sum()) < 2:
        mask = np.ones_like(expected, dtype=bool)
    obs = counts[mask]
    exp = expected[mask]
    if not mask.all():
        obs = np.append(obs, counts[~mask].sum())
        exp = np.append(exp, expected[~mask].sum())
    chi2 = float(np.sum((obs - exp) ** 2 / np.maximum(exp, 1e-18)))
    df = int(obs.size - 1)
    return _chi2_sf(chi2, df), chi2


def _assert_mass_only_acgtn(q: np.ndarray) -> None:
    assert q.shape == (VOCAB_SIZE,)
    np.testing.assert_allclose(q.sum(), 1.0, rtol=0.0, atol=1e-15)
    other = np.ones(VOCAB_SIZE, dtype=bool)
    other[MASS] = False
    np.testing.assert_array_equal(q[other], 0.0)


# --- §5.1 平凡自匹配排除 ---


def test_homopolymer_match_not_self():
    """10 个 A、k=4：haystack 前 9 位有匹配，x*='A'，不是平凡自匹配。"""
    model = PromptLookupModel(k_lookup=4, lam=0.9)
    prefix = _ascii("AAAAAAAAAA")
    hit = model.lookup(prefix)
    assert hit is not None
    m, xstar = hit
    assert xstar == BYTE_A
    # 最近匹配在 haystack[5:9]='AAAA'，x*=s[9]='A'
    assert m == 5
    q = model.probs(prefix)
    _assert_mass_only_acgtn(q)
    expected_xstar = 0.9 + (1.0 - 0.9) * UNIFORM
    np.testing.assert_allclose(q[BYTE_A], expected_xstar, rtol=0.0, atol=1e-15)


def test_n_equals_k_no_match_uniform():
    """n=k=4，搜索空间长度 3 < 4 → 均匀。"""
    model = PromptLookupModel(k_lookup=4, lam=0.9)
    prefix = _ascii("ACGT")
    assert model.lookup(prefix) is None
    assert rfind_continuation(b"ACGT", 4) is None
    q = model.probs(prefix)
    _assert_mass_only_acgtn(q)
    np.testing.assert_allclose(q[MASS], UNIFORM, rtol=0.0, atol=1e-15)


def test_query_only_at_end_uniform():
    """查询 TTGC 仅自身出现 → 均匀。"""
    model = PromptLookupModel(k_lookup=4, lam=0.9)
    prefix = _ascii("ACGTGCAATTGC")
    assert prefix_to_acgtn_bytes(prefix) == b"ACGTGCAATTGC"
    assert model.lookup(prefix) is None
    q = model.probs(prefix)
    _assert_mass_only_acgtn(q)
    np.testing.assert_allclose(q[MASS], UNIFORM, rtol=0.0, atol=1e-15)


# --- §5.2 重叠匹配 ---


def test_overlapping_match():
    """ATGCATGC、k=4，查询 ATGC → 匹配起点 0，x*=s[4]='A'。"""
    model = PromptLookupModel(k_lookup=4, lam=0.9)
    prefix = _ascii("ATGCATGC")
    hit = model.lookup(prefix)
    assert hit is not None
    m, xstar = hit
    assert m == 0
    assert xstar == BYTE_A
    q = model.probs(prefix)
    np.testing.assert_allclose(q[BYTE_A], 0.9 + 0.1 * UNIFORM, rtol=0.0, atol=1e-15)


# --- §5.3 q 构造 ---


def test_mixture_q_construction():
    model = PromptLookupModel(k_lookup=4, lam=0.9)
    q = model.probs(_ascii("ATGCATGC"))
    _assert_mass_only_acgtn(q)
    np.testing.assert_allclose(q[BYTE_A], 0.92, rtol=0.0, atol=1e-15)
    for b in (BYTE_C, BYTE_G, BYTE_T, BYTE_N):
        np.testing.assert_allclose(q[b], 0.02, rtol=0.0, atol=1e-15)
    assert DEFAULT_LAMBDA == 0.9


# --- §5.4 fallback ---


def test_fallback_uniform_and_markov():
    prefix = _ascii("ACGTGCAATTGC")  # k=4 无匹配
    uniform_model = PromptLookupModel(k_lookup=4, lam=0.9, fallback=None)
    q_u = uniform_model.probs(prefix)
    np.testing.assert_allclose(q_u[MASS], UNIFORM, rtol=0.0, atol=1e-15)

    table = MarkovTable.build(["ACGTACGTN" * 20], k=3, alpha=0.1)
    markov = MarkovDraftModel(table)
    mixed = PromptLookupModel(k_lookup=4, lam=0.9, fallback=markov)
    q_fb = mixed.probs(prefix)
    q_mk = markov.probs(prefix)
    np.testing.assert_array_equal(q_fb, q_mk)

    # 短前缀 n=k 同样委托
    short = _ascii("ACGT")
    np.testing.assert_array_equal(mixed.probs(short), markov.probs(short))


# --- §5.5 采样合法性 ---


def test_sample_from_mixture_chi2():
    rng = np.random.default_rng(SEED_SAMPLE_Q)
    model = PromptLookupModel(k_lookup=4, lam=0.9)
    q = model.probs(_ascii("AAAAAAAAAA"))
    samples = np.empty(N_SAMPLE_Q, dtype=np.int64)
    for i in range(N_SAMPLE_Q):
        samples[i] = sample_categorical(q, rng)
    allowed = set(int(x) for x in MASS_IDS)
    for tok in samples:
        assert int(tok) in allowed
    counts = np.array([(samples == b).sum() for b in MASS_IDS], dtype=np.float64)
    p_mass = q[MASS]
    pvalue, chi2 = _pearson_chi2_pvalue(counts, p_mass)
    print(
        f"[lookup-sample] n={N_SAMPLE_Q} chi2={chi2:.4f} p={pvalue:.4g} "
        f"seed={SEED_SAMPLE_Q} emp={counts / N_SAMPLE_Q}"
    )
    assert pvalue > CHI2_PVALUE_MIN, f"chi2 p={pvalue} <= {CHI2_PVALUE_MIN}"


# --- §5.6 harness 冒烟 ---


class _AcgtnByteBigram:
    """512 维 mock target：质量只放在 5 个 byte id 上的一阶转移。"""

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
    draft = PromptLookupModel(k_lookup=4, lam=0.9)
    target = _AcgtnByteBigram(
        rng.dirichlet(np.ones(5), size=5),
        rng.dirichlet(np.ones(5)),
    )
    prompt = _ascii("ACGTACGTACGT")
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
        f"[lookup-harness] n_tokens={n_tokens} gamma={gamma} n_rounds={len(results)} "
        f"seed={SEED_HARNESS_SMOKE}"
    )


# --- §5.7 机制演示 ---


def test_repeat_follower_greedy_accept_rate():
    """target=λ=1 的 lookup（重复跟随者）；含重复单元 prompt 上 greedy 接受率 > 0.8。"""
    prompt = _ascii("ATGC" * 32)
    draft = PromptLookupModel(k_lookup=4, lam=0.9)
    target = PromptLookupModel(k_lookup=4, lam=1.0)
    rng = np.random.default_rng(SEED_MECH_DEMO)
    n_tokens = 200
    gamma = 4
    _tokens, results = run_speculative(
        target, draft, prompt, n_tokens, gamma, rng, greedy=True
    )
    n_acc = sum(int(r.accepted_len) for r in results)
    n_prop = len(results) * gamma
    rate = n_acc / n_prop if n_prop else 0.0
    print(
        f"[lookup-mech] n_tokens={n_tokens} gamma={gamma} n_rounds={len(results)} "
        f"accept_rate={rate:.4f} seed={SEED_MECH_DEMO}"
    )
    assert rate > 0.8


def test_lowercase_and_pipe_cleaning():
    model = PromptLookupModel(k_lookup=4, lam=0.9)
    upper = model.probs(_ascii("ATGCATGC"))
    lower = model.probs(np.array([ord(c) for c in "atgcatgc"], dtype=np.int64))
    np.testing.assert_array_equal(upper, lower)
    # A|C 与 ANC 等价（| → N）
    pipe = model.probs(np.array([BYTE_A, ord("|"), BYTE_C, BYTE_G], dtype=np.int64))
    mapped = model.probs(_ascii("ANCG"))
    np.testing.assert_array_equal(pipe, mapped)
