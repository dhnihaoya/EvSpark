"""Phase C drafter 单测（plans/10 §6）。无 torch 时 skip；不占 GPU、不加载 7B。"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="需要 torch（drafter 实现）")

from evspark.specdec.transforms import apply_transform
from evspark.train.data import (
    LabeledSeq,
    holdout_eval_window,
    sample_anchors,
    seq_to_ids,
    split_corpus,
)
from evspark.train.drafter import (
    MASK_ID,
    VOCAB_SIZE,
    Drafter,
    analytic_accept,
    apply_transform_torch,
    build_kv_prefix_mask,
    distill_losses,
    make_block_ids,
    position_weights,
    predicted_tau,
)

GAMMA = 4
D_SMALL = 64
SEED = 20260821


def _drafter(n_inject: int = 1, gamma: int = GAMMA, d_model: int = D_SMALL) -> Drafter:
    torch.manual_seed(SEED)
    return Drafter(d_model=d_model, n_layers=2, gamma=gamma, n_inject=n_inject, n_heads=2)


# ---------------------------------------------------------------------------
# 1. 形状 / 掩码 / KV 前缀方向
# ---------------------------------------------------------------------------


def test_block_shapes_and_prob_simplex():
    m = _drafter(n_inject=1)
    bsz, t, gamma = 3, 16, GAMMA
    anchor = torch.tensor([65, 67, 71])
    prev = torch.tensor([[65, 67, 71, 84]] * bsz)
    ctx_lens = torch.tensor([5, 9, 16])
    h_raw = torch.randn(t, 1 * 4096)
    logits, probs, conf, hidden = m(anchor, prev, ctx_lens, h_raw=h_raw)
    assert logits.shape == (bsz, gamma, VOCAB_SIZE)
    assert probs.shape == (bsz, gamma, VOCAB_SIZE)
    assert conf.shape == (bsz, gamma)
    assert hidden.shape == (bsz, gamma, D_SMALL)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(bsz, gamma), atol=1e-5)
    assert (probs >= 0).all()
    assert (conf >= 0).all() and (conf <= 1).all()
    ids = make_block_ids(anchor, gamma)
    assert ids.shape == (bsz, gamma)
    assert torch.equal(ids[:, 0], anchor)
    assert torch.equal(ids[:, 1:], torch.full((bsz, gamma - 1), MASK_ID))


def test_causal_and_prefix_mask_layout():
    ctx_lens = torch.tensor([5, 2])
    gamma, t = 4, 8
    mask = build_kv_prefix_mask(ctx_lens, gamma, t)
    assert mask.shape == (2, 1, gamma, t + gamma)
    # KV 前缀在序列维左侧：前 T 列是 H_ctx，后 γ 列是 draft
    for b, cl in enumerate(ctx_lens.tolist()):
        for k in range(gamma):
            ctx = mask[b, 0, k, :t]
            draft = mask[b, 0, k, t:]
            assert ctx[:cl].all()
            assert (~ctx[cl:]).all()
            expected = torch.tensor([True] * (k + 1) + [False] * (gamma - k - 1))
            assert torch.equal(draft, expected)


def test_ctx_window_hides_distant_prefix():
    ctx_lens = torch.tensor([8])
    mask = build_kv_prefix_mask(ctx_lens, gamma=4, ctx_max=12, ctx_window=3)
    ctx = mask[0, 0, 0, :12]
    # lo = 8-3 = 5 → 可见 5,6,7
    assert not ctx[:5].any()
    assert ctx[5:8].all()
    assert not ctx[8:].any()

    m = Drafter(d_model=D_SMALL, n_layers=2, gamma=GAMMA, n_inject=1, n_heads=2, ctx_window=3)
    m.eval()
    t = 12
    anchor = torch.tensor([65])
    prev = torch.tensor([[65, 67, 71, 84]])
    ctx_lens = torch.tensor([8])
    h = torch.randn(t, 4096)
    h_far = h.clone()
    h_far[1] = h_far[1] + 10.0  # t=1 < lo=5，应不可见
    h_near = h.clone()
    h_near[6] = h_near[6] + 10.0  # 在窗口内
    with torch.no_grad():
        log0, _, _, _ = m(anchor, prev, ctx_lens, h_raw=h)
        log_f, _, _, _ = m(anchor, prev, ctx_lens, h_raw=h_far)
        log_n, _, _, _ = m(anchor, prev, ctx_lens, h_raw=h_near)
    assert torch.allclose(log0, log_f, atol=1e-5, rtol=1e-4)
    assert (log0 - log_n).abs().max() > 1e-4


def test_causal_mask_hides_future_draft_tokens():
    m = _drafter(n_inject=0)
    m.eval()
    bsz, gamma = 2, GAMMA
    anchor = torch.tensor([65, 67])
    prev = torch.stack([torch.tensor([65, 67, 71, 84])] * bsz)
    ctx = torch.tensor([0, 0])
    ids_a = torch.tensor([[65, 67, 71, 84], [65, 67, 71, 84]])
    ids_b = ids_a.clone()
    ids_b[:, -1] = 78
    with torch.no_grad():
        log_a, _, _, _ = m(anchor, prev, ctx, h_raw=None, draft_ids=ids_a)
        log_b, _, _, _ = m(anchor, prev, ctx, h_raw=None, draft_ids=ids_b)
    assert torch.allclose(log_a[:, :-1], log_b[:, :-1], atol=1e-5, rtol=1e-4)
    assert log_a.shape == log_b.shape


def test_ctx_prefix_visibility():
    m = _drafter(n_inject=1)
    m.eval()
    t, gamma = 12, GAMMA
    anchor = torch.tensor([65])
    prev = torch.tensor([[65, 67, 71, 84]])
    ctx_lens = torch.tensor([6])
    h = torch.randn(t, 4096)
    h_future = h.clone()
    h_future[9] = h_future[9] + 10.0
    h_past = h.clone()
    h_past[2] = h_past[2] + 10.0
    with torch.no_grad():
        log0, _, _, _ = m(anchor, prev, ctx_lens, h_raw=h)
        log_f, _, _, _ = m(anchor, prev, ctx_lens, h_raw=h_future)
        log_p, _, _, _ = m(anchor, prev, ctx_lens, h_raw=h_past)
    assert torch.allclose(log0, log_f, atol=1e-5, rtol=1e-4)
    assert (log0 - log_p).abs().max() > 1e-4


# ---------------------------------------------------------------------------
# 2. 冻结纪律
# ---------------------------------------------------------------------------


def test_freeze_embedding_lmhead_and_mock_target():
    m = _drafter(n_inject=1)
    mock_target = torch.nn.Linear(4, 4)
    torch.nn.init.zeros_(mock_target.weight)
    torch.nn.init.zeros_(mock_target.bias)
    mock_target.requires_grad_(False)
    emb0 = m.embed.weight.detach().clone()
    tgt0 = mock_target.weight.detach().clone()
    opt = torch.optim.AdamW(m.trainable_parameters(), lr=0.05)
    assert all(not p.requires_grad for p in m.embed.parameters())
    assert not any(id(m.embed.weight) == id(p) for p in m.trainable_parameters())

    anchor = torch.tensor([65, 67])
    prev = torch.tensor([[65, 67, 71, 84], [67, 71, 84, 65]])
    ctx = torch.tensor([8, 8])
    h_raw = torch.randn(10, 4096)
    target = torch.tensor([[67, 71, 84, 65], [71, 84, 65, 67]])
    p_t = torch.zeros(2, GAMMA, VOCAB_SIZE)
    p_t.scatter_(-1, target.unsqueeze(-1), 1.0)
    opt.zero_grad(set_to_none=True)
    logits, probs, conf, _ = m(anchor, prev, ctx, h_raw=h_raw)
    loss = distill_losses(logits, p_t, target, conf, p_d=probs).total
    loss = loss + 0.0 * mock_target.weight.reshape(-1)[0]
    loss.backward()
    opt.step()
    assert torch.equal(m.embed.weight, emb0)
    assert torch.equal(mock_target.weight, tgt0)
    moved = False
    for p in m.trainable_parameters():
        if p.grad is not None and float(p.grad.abs().max()) > 0:
            moved = True
            break
    assert moved


# ---------------------------------------------------------------------------
# 3. 损失组件手算对照
# ---------------------------------------------------------------------------


def test_position_weights_formula():
    gamma = 7
    w = position_weights(gamma)
    for k in range(gamma):
        assert math.isclose(float(w[k]), math.exp(-k / gamma), rel_tol=0, abs_tol=1e-7)


def test_loss_components_hand_computed():
    bsz, gamma, v = 1, 2, VOCAB_SIZE
    p_d = torch.zeros(bsz, gamma, v)
    p_t = torch.zeros(bsz, gamma, v)
    p_d[0, 0, 0], p_d[0, 0, 1] = 0.5, 0.5
    p_d[0, 1, 0], p_d[0, 1, 1] = 0.25, 0.75
    p_t[0, 0, 0] = 1.0
    p_t[0, 1, 1] = 1.0
    tokens = torch.tensor([[0, 1]])
    conf_pred = torch.tensor([[0.5, 0.5]])
    logits = p_d.clamp_min(1e-12).log()
    br = distill_losses(logits, p_t, tokens, conf_pred, p_d=p_d)
    w = position_weights(2)
    ce0 = -math.log(0.5)
    ce1 = -math.log(0.75)
    ce = w[0] * ce0 + w[1] * ce1
    tv0 = abs(0.5 - 1.0) + abs(0.5 - 0.0)
    tv1 = abs(0.25 - 0.0) + abs(0.75 - 1.0)
    tv = w[0] * tv0 + w[1] * tv1
    c0 = 1.0 - 0.5 * tv0
    c1 = 1.0 - 0.5 * tv1

    def bce(p, t):
        p = min(max(p, 1e-6), 1.0 - 1e-6)
        return -(t * math.log(p) + (1.0 - t) * math.log(1.0 - p))

    conf = w[0] * bce(0.5, c0) + w[1] * bce(0.5, c1)
    total = 0.1 * ce + 0.9 * tv + 1.0 * conf
    assert math.isclose(float(br.ce), float(ce), rel_tol=0, abs_tol=1e-5)
    assert math.isclose(float(br.tv), float(tv), rel_tol=0, abs_tol=1e-5)
    assert math.isclose(float(br.conf), float(conf), rel_tol=0, abs_tol=1e-5)
    assert math.isclose(float(br.total), float(total), rel_tol=0, abs_tol=1e-5)


def test_apply_transform_torch_matches_numpy():
    rng = np.random.default_rng(SEED)
    logits = rng.normal(size=(6, VOCAB_SIZE)).astype(np.float64)
    for row in logits:
        np_p = apply_transform(row, temperature=1.0, top_k=4)
        torch_p = apply_transform_torch(torch.tensor(row), temperature=1.0, top_k=4).numpy()
        np.testing.assert_allclose(torch_p, np_p, atol=1e-6, rtol=1e-5)
        assert int(np.count_nonzero(np_p > 0)) <= 4


# ---------------------------------------------------------------------------
# 4. 过拟合冒烟
# ---------------------------------------------------------------------------


def test_overfit_loss_drops():
    torch.manual_seed(SEED)
    m = _drafter(n_inject=1, d_model=D_SMALL)
    opt = torch.optim.AdamW(m.trainable_parameters(), lr=3e-4, weight_decay=0.0)
    bsz, gamma, t = 4, GAMMA, 24
    anchor = torch.tensor([65, 67, 71, 84])
    prev = torch.tensor(
        [
            [65, 67, 71, 84],
            [67, 71, 84, 65],
            [71, 84, 65, 67],
            [84, 65, 67, 71],
        ]
    )
    target = torch.tensor(
        [
            [67, 71, 84, 65],
            [71, 84, 65, 67],
            [84, 65, 67, 71],
            [65, 67, 71, 84],
        ]
    )
    ctx = torch.tensor([10, 12, 14, 16])
    h_raw = torch.randn(t, 4096)
    p_t = torch.zeros(bsz, gamma, VOCAB_SIZE)
    p_t.scatter_(-1, target.unsqueeze(-1), 1.0)
    losses = []
    for _ in range(80):
        opt.zero_grad(set_to_none=True)
        logits, probs, conf, _ = m(anchor, prev, ctx, h_raw=h_raw)
        loss = distill_losses(logits, p_t, target, conf, p_d=probs).total
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.trainable_parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))
    assert losses[-1] < 0.70 * losses[0], f"loss {losses[0]:.3f} → {losses[-1]:.3f}"


# ---------------------------------------------------------------------------
# 5. 解析接受率 / τ̂
# ---------------------------------------------------------------------------


def test_analytic_accept_and_tau_hat():
    p_d = torch.zeros(2, VOCAB_SIZE)
    p_t = torch.zeros(2, VOCAB_SIZE)
    p_d[0, 0], p_d[0, 1] = 0.5, 0.5
    p_t[0, 0] = 1.0
    p_d[1, 1] = 1.0
    p_t[1, 1] = 1.0
    c = analytic_accept(p_d, p_t)
    assert math.isclose(float(c[0]), 0.5, abs_tol=1e-6)
    assert math.isclose(float(c[1]), 1.0, abs_tol=1e-6)
    alpha = torch.tensor([1.0, 1.0, 1.0])
    assert math.isclose(float(predicted_tau(alpha)), 4.0, abs_tol=1e-6)
    alpha = torch.tensor([0.0, 0.0])
    assert math.isclose(float(predicted_tau(alpha)), 1.0, abs_tol=1e-6)
    alpha = torch.tensor([0.5, 0.5])
    assert math.isclose(float(predicted_tau(alpha)), 1.75, abs_tol=1e-6)


def test_s0_forward_no_context_tensor():
    m = Drafter.from_scheme("S0", d_model=D_SMALL, gamma=GAMMA, n_heads=2)
    assert m.n_inject == 0 and m.ctx_proj is None
    anchor = torch.tensor([65])
    prev = torch.tensor([[65, 67, 71, 84]])
    ctx = torch.tensor([0])
    _, probs, conf, _ = m(anchor, prev, ctx, h_raw=None)
    assert probs.shape == (1, GAMMA, VOCAB_SIZE)
    assert conf.shape == (1, GAMMA)


def test_data_split_and_anchor_bounds():
    recs = [LabeledSeq(name="s", seq="A" * 100 + "C" * 100)]
    train, hold = split_corpus(recs, test_frac=0.1)
    assert len(train[0].seq) == 180
    assert len(hold[0].seq) == 20
    ids, hold_s, split = holdout_eval_window("ACGT" * 50, test_frac=0.1, window=80)
    assert ids.shape[0] == 80
    assert split == 180
    assert hold_s == 80 - (200 - 180)
    rng = np.random.default_rng(0)
    a = sample_anchors(64, gamma=7, n_anchors=8, rng=rng, min_ctx=10)
    assert a.min() >= 9
    assert a.max() + 7 < 64
    ids2 = seq_to_ids("ACGTN")
    np.testing.assert_array_equal(ids2, np.array([65, 67, 71, 84, 78]))
