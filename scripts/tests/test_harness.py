"""CPU 端到端 harness 单测。对应 plans/03 §5 条目 7–9。"""

from __future__ import annotations

import numpy as np

from specdec.harness import run_native, run_speculative
from specdec.mock_models import random_bigram
from specdec.stats import position_accept_rates, tau_from_mask
from specdec.verifier import verify_round

SEED_GREEDY_HARNESS = 42
SEED_SAMPLE_HARNESS = 7
SEED_TAU = 9

N_GREEDY_TABLES = 50
N_GREEDY_TOKENS = 1000
GREEDY_GAMMAS = (1, 2, 4, 8)
N_SAMPLE_TOKENS = 200_000
KL_MAX = 1e-3


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
    """返回 joint [V,V] 与按行归一化的条件分布 [V,V]。"""
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


def _weighted_conditional_kl(cond_p: np.ndarray, cond_q: np.ndarray, weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    total = 0.0
    for i, wi in enumerate(w):
        if wi <= 0.0:
            continue
        total += wi * _kl(cond_p[i], cond_q[i])
    return float(total)


def test_greedy_lossless_tokenwise():
    rng = np.random.default_rng(SEED_GREEDY_HARNESS)
    prompt = np.array([0], dtype=np.int64)
    n_ok = 0
    for i in range(N_GREEDY_TABLES):
        target = random_bigram(rng, vocab_size=4, alpha=1.0)
        draft = random_bigram(rng, vocab_size=4, alpha=1.0)
        native = run_native(target, prompt, N_GREEDY_TOKENS, rng, greedy=True)
        for gamma in GREEDY_GAMMAS:
            spec, _results = run_speculative(
                target, draft, prompt, N_GREEDY_TOKENS, gamma, rng, greedy=True
            )
            np.testing.assert_array_equal(spec, native, err_msg=f"table={i} gamma={gamma}")
            n_ok += 1
    print(
        f"[7] greedy lossless tables={N_GREEDY_TABLES} gammas={GREEDY_GAMMAS} "
        f"tokens={N_GREEDY_TOKENS} comparisons={n_ok} seed={SEED_GREEDY_HARNESS}"
    )


def test_sampling_lossless_kl():
    rng_model = np.random.default_rng(SEED_SAMPLE_HARNESS)
    target = random_bigram(rng_model, vocab_size=4, alpha=0.7)
    draft = random_bigram(rng_model, vocab_size=4, alpha=0.7)
    prompt = np.array([0], dtype=np.int64)
    native = run_native(
        target, prompt, N_SAMPLE_TOKENS, np.random.default_rng(7001), greedy=False
    )
    spec, _results = run_speculative(
        target,
        draft,
        prompt,
        N_SAMPLE_TOKENS,
        4,
        np.random.default_rng(7002),
        greedy=False,
    )
    vocab = 4
    start = int(prompt[-1])
    uni_n = _unigram(native, vocab)
    uni_s = _unigram(spec, vocab)
    kl_uni = _kl(uni_s, uni_n)

    joint_n, cond_n = _bigram_conditionals(native, start, vocab)
    joint_s, cond_s = _bigram_conditionals(spec, start, vocab)
    kl_joint = _kl(joint_s.ravel(), joint_n.ravel())
    weights = joint_n.sum(axis=1)
    kl_cond = _weighted_conditional_kl(cond_s, cond_n, weights)

    print(
        f"[8] N={N_SAMPLE_TOKENS} gamma=4 KL_unigram={kl_uni:.6e} "
        f"KL_bigram_joint={kl_joint:.6e} KL_bigram_cond={kl_cond:.6e} "
        f"seed_model={SEED_SAMPLE_HARNESS} seed_native=7001 seed_spec=7002"
    )
    assert kl_uni < KL_MAX, f"unigram KL {kl_uni} >= {KL_MAX}"
    assert kl_joint < KL_MAX, f"bigram joint KL {kl_joint} >= {KL_MAX}"
    assert kl_cond < KL_MAX, f"bigram cond KL {kl_cond} >= {KL_MAX}"


def _ref_accepted_len(draft_tokens: np.ndarray, q: np.ndarray, p: np.ndarray, rng: np.random.Generator) -> tuple[int, np.ndarray]:
    """测试内直写接受循环，不调用 verify_round。"""
    draft_tokens = np.asarray(draft_tokens, dtype=np.int64).ravel()
    gamma = int(draft_tokens.shape[0])
    mask = np.zeros(gamma, dtype=bool)
    accepted = 0
    for i in range(gamma):
        x = int(draft_tokens[i])
        p_x = float(p[i, x])
        q_x = float(q[i, x])
        accept_prob = min(1.0, p_x / q_x) if q_x > 0.0 else 0.0
        if float(rng.random()) < accept_prob:
            mask[i] = True
            accepted += 1
        else:
            break
    return accepted, mask


def test_tau_accounting_and_position_rates():
    rng = np.random.default_rng(SEED_TAU)
    target = random_bigram(rng, vocab_size=4, alpha=0.8)
    draft = random_bigram(rng, vocab_size=4, alpha=0.8)
    prompt = np.array([0], dtype=np.int64)
    _seq, results = run_speculative(
        target, draft, prompt, 8000, 8, rng, greedy=False
    )
    for r in results:
        tau_mask = tau_from_mask(r.accepted_mask)
        assert r.accepted_len == tau_mask
        assert int(r.tokens.shape[0]) == r.accepted_len + 1

    rates = position_accept_rates(results)
    suffix_mean = float(rates[1:].mean()) if rates.size > 1 else 0.0
    print(
        f"[9] n_rounds={len(results)} mean_tau={np.mean([r.accepted_len for r in results]):.4f} "
        f"pos_rates={rates} first={rates[0]:.4f} suffix_mean={suffix_mean:.4f} seed={SEED_TAU}"
    )
    assert rates[0] >= suffix_mean - 1e-15

    # 独立参考实现：随机 p/q 上与 verify_round 对拍 τ / mask
    data_rng = np.random.default_rng(99)
    n_match = 0
    for _ in range(500):
        gamma = 4
        vocab = 8
        p = data_rng.dirichlet(np.ones(vocab), size=gamma + 1)
        q = data_rng.dirichlet(np.ones(vocab), size=gamma)
        draft_tokens = np.array(
            [data_rng.choice(vocab, p=q[i]) for i in range(gamma)], dtype=np.int64
        )
        seed = int(data_rng.integers(0, 2**31 - 1))
        result = verify_round(draft_tokens, q, p, np.random.default_rng(seed))
        acc, mask = _ref_accepted_len(draft_tokens, q, p, np.random.default_rng(seed))
        assert result.accepted_len == acc == tau_from_mask(result.accepted_mask)
        np.testing.assert_array_equal(result.accepted_mask, mask)
        n_match += 1
    print(f"[9] independent-loop matches={n_match}/500")
