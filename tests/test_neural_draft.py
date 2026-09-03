"""Step 10（Plan 02 C.5，plans/12 §4）神经 drafter 解码适配单测。

CPU（torch 即可，无需 CUDA）：
1. 草稿路径——并行 trunk + 串行 Markov 偏置，与训练侧 ``Drafter.forward``
   eval 路径在同输入（teacher-forced 前驱）上**逐位一致**；q′ 支撑集性质。
2. H_ctx 滞后口径——decode 式滚动缓冲喂法与 eval 式滞后掩码（ctx_lens=a）
   逐位一致（plans/12 §2 对齐结论的机械钉死）。
3. ``HiddenCapture`` 滚动缓冲：begin/commit/discard、窗口截断、拒绝剔除。

GPU（迷你 StripedHyena，fp32）：
4. 循环集成——神经 drafter 跑 slice 循环，H_ctx 缓冲与「整段重算」对拍；
   贪心输出与逐步参照全等；guard（snapshot/外部状态/γ 不符）报错。

无 torch/vortex 时本文件整体 skip（CPU 套件不受影响）；无 CUDA 时第 4 项 skip。
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="需要 evo2 环境（torch）")
if not torch.cuda.is_available():
    # vortex.model 在 import 时初始化 CUDA，无卡机器会在 collection 阶段直接抛
    # RuntimeError，这里转成干净的 skip（本文件全部用例都需要 GPU/编译内核）。
    pytest.skip("需要 CUDA", allow_module_level=True)
pytest.importorskip("vortex", reason="需要 vortex(vtx) 包（neural_draft 经 loop 引入）")

from evspark.specdec.block.neural_draft import (  # noqa: E402
    HiddenCapture,
    NeuralDraftModel,
    serial_sample,
    trunk_forward,
)
from evspark.train.drafter import VOCAB_SIZE, Drafter  # noqa: E402

SEED = 20260830
GAMMA = 4
N_INJECT = 2
TARGET_HIDDEN = 32
D_MODEL = 64


def _small_drafter(gamma: int = GAMMA, ctx_window: int = 64) -> Drafter:
    torch.manual_seed(SEED)
    d = Drafter(
        d_model=D_MODEL,
        n_layers=2,
        gamma=gamma,
        n_inject=N_INJECT,
        n_heads=2,
        target_hidden=TARGET_HIDDEN,
        ctx_window=ctx_window,
    )
    with torch.no_grad():
        # markov 零初始化会让偏置路径空转，随机化以真正 exercised
        d.markov.copy_(torch.randn_like(d.markov) * 0.5)
    d.eval()
    return d


def _rand_ctx(T: int, n_inject: int = N_INJECT, hidden: int = TARGET_HIDDEN) -> torch.Tensor:
    g = torch.Generator().manual_seed(SEED + 1)
    return torch.randn(T, n_inject * hidden, generator=g)


# ---------------------------------------------------------------------------
# 1. 草稿路径：与训练侧 eval 逐位一致 + q′ 性质
# ---------------------------------------------------------------------------


def test_trunk_serial_bitwise_vs_train_eval():
    d = _small_drafter()
    T = 10
    h_raw = _rand_ctx(T)
    anchor_tok = 65
    anchor = torch.tensor([anchor_tok])
    prev_tokens = [65, 67, 71, 84][:GAMMA]  # prev[k] = 位置 k 的前驱（teacher forcing）
    prev = torch.tensor([prev_tokens])
    ctx_lens = torch.tensor([T])

    with torch.no_grad():
        _, probs_ref, conf_ref, _ = d(anchor, prev, ctx_lens, h_raw=h_raw)
        U, h = trunk_forward(d, anchor, h_raw, ctx_lens)
        # forced[k-1] = prev[k]（k≥1）→ 串行头使用的前驱序列与 teacher forcing 相同
        forced = np.array(prev_tokens[1:] + [78], dtype=np.int64)
        tokens, q_rows, confs = serial_sample(
            d, U, h, anchor_tok, True, np.random.default_rng(0), forced_tokens=forced
        )

    np.testing.assert_array_equal(tokens, forced)
    np.testing.assert_array_equal(
        q_rows, probs_ref[0].detach().cpu().numpy().astype(np.float64)
    )
    np.testing.assert_array_equal(
        confs, conf_ref[0].detach().cpu().numpy().astype(np.float64)
    )


def test_serial_sampling_q_prime_properties():
    d = _small_drafter()
    h_raw = _rand_ctx(12)
    anchor_tok = 84
    with torch.no_grad():
        U, h = trunk_forward(d, torch.tensor([anchor_tok]), h_raw, torch.tensor([12]))
        tokens, q_rows, confs = serial_sample(
            d, U, h, anchor_tok, False, np.random.default_rng(42), temperature=1.0, top_k=4
        )
    assert tokens.shape == (GAMMA,) and q_rows.shape == (GAMMA, VOCAB_SIZE)
    for k in range(GAMMA):
        np.testing.assert_allclose(q_rows[k].sum(), 1.0, atol=1e-12)
        assert int((q_rows[k] > 0).sum()) <= 4
        assert q_rows[k, int(tokens[k])] > 0.0
    assert np.all(confs >= 0.0) and np.all(confs <= 1.0)
    # 同种子重放逐位一致（rng 消耗确定）
    with torch.no_grad():
        U2, h2 = trunk_forward(d, torch.tensor([anchor_tok]), h_raw, torch.tensor([12]))
        tokens2, q_rows2, _ = serial_sample(
            d, U2, h2, anchor_tok, False, np.random.default_rng(42), temperature=1.0, top_k=4
        )
    np.testing.assert_array_equal(tokens, tokens2)
    np.testing.assert_array_equal(q_rows, q_rows2)


def test_lagged_ctx_decode_feed_equals_eval_mask():
    """decode 式缓冲（近 64 个已消费 hidden，全可见）== eval 式滞后掩码（ctx_lens=a）。

    锚点下标 a=10、窗口 8：eval 可见 [a−8, a−1]；decode 喂同一切片、ctx_len=8。
    """
    d = _small_drafter(ctx_window=8)
    T, a, w = 12, 10, 8
    h_raw = _rand_ctx(T)
    anchor = torch.tensor([65])
    prev = torch.tensor([[65, 67, 71, 84][:GAMMA]])
    with torch.no_grad():
        # eval 式：整段 h_raw + 滞后掩码 ctx_lens=a（不含锚点自身 hidden）
        _, probs_eval, _, _ = d(anchor, prev, torch.tensor([a]), h_raw=h_raw)
        # decode 式：缓冲只含已消费位置 [a−w, a−1]，ctx_len = w 全可见
        buf = h_raw[a - w : a]
        U, h = trunk_forward(d, anchor, buf, torch.tensor([w]))
        forced = np.array([67, 71, 84, 78], dtype=np.int64)
        _, q_rows, _ = serial_sample(
            d, U, h, 65, True, np.random.default_rng(0), forced_tokens=forced
        )
    # 注意：两种喂法的注意力 K 维不同（16 vs 12），掩码列贡献严格 0，但 fp32
    # matmul 归约分组不同 → ulp 级差（实测 max|Δ|≈9e-10）。断言语义等价，
    # 容差 1e-7；同形状主路径（上一个测试）保持逐位一致。
    diff = np.abs(q_rows - probs_eval[0].detach().cpu().numpy().astype(np.float64))
    print(f"[lagged-eq] max|Δ|={diff.max():.3e}")
    np.testing.assert_allclose(
        q_rows,
        probs_eval[0].detach().cpu().numpy().astype(np.float64),
        atol=1e-7,
        rtol=0.0,
    )


# ---------------------------------------------------------------------------
# 2. HiddenCapture 滚动缓冲
# ---------------------------------------------------------------------------


class _ToyBlock(torch.nn.Module):
    def forward(self, x, scale):
        return x * scale  # 非 tuple 输出路径


class _ToyTupleBlock(torch.nn.Module):
    def forward(self, x, scale):
        return x * scale, None  # tuple 输出路径（取第 0 元）


class _ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_ToyBlock(), _ToyTupleBlock()])

    def forward(self, x):
        return x


def test_hidden_capture_commit_window_and_reject_trim():
    model = _ToyModel()
    layers = ("blocks.0", "blocks.1")
    cap = HiddenCapture(model, layers, maxlen=5)
    H = 3

    # 未 armed 的前向被忽略
    model.blocks[0](torch.ones(1, 4, H), 2.0)
    assert cap.buffer_len == 0

    # 第 1 轮：捕获 7 位，commit 3 位（拒绝后缀 4 位剔除）
    cap.begin()
    x1 = torch.arange(1, 8, dtype=torch.float32).view(1, 7, 1).expand(1, 7, H).contiguous()
    model.blocks[0](x1, 1.0)  # blocks.0 原样
    model.blocks[1](x1, 10.0)  # blocks.1 ×10
    cap.commit(3)
    assert cap.buffer_len == 3
    np.testing.assert_array_equal(cap._buffers["blocks.0"][:, 0].numpy(), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(cap._buffers["blocks.1"][:, 0].numpy(), [10.0, 20.0, 30.0])

    # 第 2 轮：commit 4 位 → 缓冲 7 > maxlen=5，截到近 5 位
    cap.begin()
    x2 = torch.arange(8, 12, dtype=torch.float32).view(1, 4, 1).expand(1, 4, H).contiguous()
    model.blocks[0](x2, 1.0)
    model.blocks[1](x2, 10.0)
    cap.commit(4)
    assert cap.buffer_len == 5
    np.testing.assert_array_equal(cap._buffers["blocks.0"][:, 0].numpy(), [3.0, 8.0, 9.0, 10.0, 11.0])

    # context_tensor：按 layer 序拼接、取近 window
    ctx = cap.context_tensor(window=2)
    assert ctx.shape == (2, 2 * H)
    np.testing.assert_array_equal(ctx[:, 0].numpy(), [10.0, 11.0])
    np.testing.assert_array_equal(ctx[:, H].numpy(), [100.0, 110.0])

    # discard：本轮槽位丢弃，缓冲不变
    cap.begin()
    model.blocks[0](x2, 1.0)
    model.blocks[1](x2, 10.0)
    cap.discard()
    assert cap.buffer_len == 5

    # 未 begin 直接 commit / n_keep 超长：报错
    with pytest.raises(RuntimeError):
        cap.commit(1)
    cap.begin()
    model.blocks[0](x2, 1.0)
    model.blocks[1](x2, 1.0)
    with pytest.raises(ValueError):
        cap.commit(5)
    cap.discard()
    cap.close()


# ---------------------------------------------------------------------------
# 3. GPU：迷你模型循环集成 + 缓冲 vs 整段重算
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mini_model():
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA（GPU0）")
    from test_block_forward import _build_mini_model

    return _build_mini_model()


def _mini_neural_draft(mini_model, gamma: int = GAMMA, ctx_window: int = 16) -> NeuralDraftModel:
    torch.manual_seed(SEED + 2)
    d = Drafter(
        d_model=D_MODEL,
        n_layers=2,
        gamma=gamma,
        n_inject=2,
        n_heads=2,
        target_hidden=int(mini_model.config.hidden_size),
        ctx_window=ctx_window,
    )
    with torch.no_grad():
        d.markov.copy_(torch.randn_like(d.markov) * 0.5)
    d = d.to("cuda:0")
    d.eval()
    for p in d.parameters():
        p.requires_grad_(False)
    return NeuralDraftModel(mini_model, d, ("blocks.2", "blocks.6"))


def test_mini_loop_neural_buffer_vs_recompute(mini_model):
    from evspark.specdec.block.loop import native_greedy_reference, speculative_generate
    from test_block_forward import ACGTN_IDS, PREFIX_LEN

    gamma, n = GAMMA, 32
    nd = _mini_neural_draft(mini_model, gamma=gamma)
    rng = np.random.default_rng(SEED + 3)
    prompt = torch.tensor(rng.choice(ACGTN_IDS, size=PREFIX_LEN), dtype=torch.long)[None].to("cuda:0")
    try:
        spec = speculative_generate(
            mini_model, nd, prompt, n, gamma, greedy=True, rng=np.random.default_rng(0)
        )
        # 无损性（贪心契约，任意 drafter 成立）+ 拒绝覆盖（随机 drafter 必有 k<γ）
        native, _ip, _ = native_greedy_reference(mini_model, prompt, n)
        np.testing.assert_array_equal(spec.emitted_ids, native)
        assert any(r.k < gamma for r in spec.rounds_log)
        assert all(r.conf is not None and len(r.conf) == gamma for r in spec.rounds_log)

        # 整段重算：已消费 = prompt + emitted[:-1]（无状态并行前向）
        consumed = np.concatenate(
            [prompt[0].detach().cpu().numpy().astype(np.int64), spec.emitted_ids[:-1]]
        )
        ids = torch.tensor(consumed, dtype=torch.long, device="cuda:0")[None]
        cap2 = HiddenCapture(mini_model, nd.layer_names, maxlen=nd.capture.maxlen)
        try:
            cap2.begin()
            with torch.inference_mode():
                mini_model(ids)
            cap2.commit(int(ids.shape[1]))
        finally:
            cap2.close()
        for name in nd.layer_names:
            a = nd.capture._buffers[name]
            b = cap2._buffers[name]
            assert a.shape == b.shape == (nd.capture.maxlen, int(mini_model.config.hidden_size))
            max_abs = float((a - b).abs().max())
            print(f"[buffer-vs-recompute] {name} max|Δ|={max_abs:.3e}")
            assert max_abs < 1e-3, f"层 {name} 缓冲 vs 整段重算 max|Δ|={max_abs:.3e}"
    finally:
        nd.close()


def test_mini_loop_neural_sampling_and_guards(mini_model):
    from evspark.specdec.block.driver import clone_inference_params
    from evspark.specdec.block.loop import speculative_generate
    from test_block_forward import ACGTN_IDS, PREFIX_LEN

    gamma, n = GAMMA, 16
    nd = _mini_neural_draft(mini_model, gamma=gamma)
    rng = np.random.default_rng(SEED + 4)
    prompt = torch.tensor(rng.choice(ACGTN_IDS, size=PREFIX_LEN), dtype=torch.long)[None].to("cuda:0")
    try:
        # 采样路径跑通：q′ 进 verify、conf 落 rounds_log
        spec = speculative_generate(
            mini_model,
            nd,
            prompt,
            n,
            gamma,
            greedy=False,
            rng=np.random.default_rng(7),
            temperature=1.0,
            top_k=4,
        )
        assert spec.emitted_ids.shape == (n,)
        assert all(r.conf is not None and len(r.conf) == gamma for r in spec.rounds_log)

        # guard：snapshot / 外部 inference_params / γ 不符
        with pytest.raises(ValueError, match="slice"):
            speculative_generate(
                mini_model, nd, prompt, 4, gamma, greedy=True,
                rng=np.random.default_rng(0), rollback="snapshot",
            )
        ip = mini_model.initialize_inference_params(max_seqlen=PREFIX_LEN + 32)
        ip = clone_inference_params(ip)
        # guard：外部状态但 H_ctx 缓冲未对齐（Step 15 起长 prompt 分块预填路径要求
        # 预先 capture_begin/commit 对齐；未对齐必须拒绝）
        with pytest.raises(ValueError, match="对齐"):
            speculative_generate(
                mini_model, nd, prompt, 4, gamma, greedy=True,
                rng=np.random.default_rng(0), inference_params_dict=ip,
            )
        with pytest.raises(ValueError, match="γ"):
            speculative_generate(
                mini_model, nd, prompt, 4, gamma + 1, greedy=True,
                rng=np.random.default_rng(0),
            )
    finally:
        nd.close()


def test_mini_loop_external_prefill_aligned(mini_model):
    """Step 15 长 prompt 分块预填路径：对齐的外部状态 == 循环内部 prefill。

    分块（2 块）预填期间保持 capture_begin，commit(L-1) 后以外部状态进循环；
    与内部路径同 rng 同 prompt 采样输出必须逐位一致。"""
    from evspark.specdec.block.driver import block_forward, prefill
    from evspark.specdec.block.loop import speculative_generate
    from test_block_forward import ACGTN_IDS, PREFIX_LEN

    gamma, n = GAMMA, 12
    nd = _mini_neural_draft(mini_model, gamma=gamma)
    rng = np.random.default_rng(SEED + 11)
    L = PREFIX_LEN
    prompt = torch.tensor(rng.choice(ACGTN_IDS, size=L), dtype=torch.long)[None].to("cuda:0")
    try:
        ref = speculative_generate(
            mini_model, nd, prompt, n, gamma, greedy=False,
            rng=np.random.default_rng(5), temperature=1.0, top_k=4,
        )
        nd.capture_reset()
        ip = mini_model.initialize_inference_params(max_seqlen=L + n + gamma + 2)
        nd.capture_begin()
        split = max(128, L // 2)  # HCM FIR 内窗要求首块前缀 ≥ k_inner−1
        with torch.inference_mode():
            prefill(mini_model, prompt[:, :split], ip)
            block_forward(mini_model, prompt[:, split : L - 1], ip)
        nd.capture_commit(L - 1)
        assert nd.capture_buffer_len() == L - 1
        ext = speculative_generate(
            mini_model, nd, prompt, n, gamma, greedy=False,
            rng=np.random.default_rng(5), temperature=1.0, top_k=4,
            inference_params_dict=ip,
        )
        np.testing.assert_array_equal(ref.emitted_ids, ext.emitted_ids)
        assert [r.k for r in ext.rounds_log] == [r.k for r in ref.rounds_log]
    finally:
        nd.close()
