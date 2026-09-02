"""神经 drafter 解码适配（Plan 02 C.5 / Step 10）：并行草稿 + 串行 Markov 头。

把 ``train.drafter.Drafter``（DSpark 改装：γ 并行 trunk + 目标隐层 KV 前缀 +
全矩阵 Markov 偏置 + 置信度头）接入 ``loop.speculative_generate`` 的锚点循环：

- **H_ctx 捕获**：``HiddenCapture`` 在 prefill / 块前向期间用 forward hook 抓
  注入层（S3 = blocks.6/16/27）输出，维护近 ``ctx_window`` 个**已消费**位置的
  滚动缓冲。循环在每轮块前向前 ``capture_begin()``、验证后 ``capture_commit(j*+1)``
  ——与状态切片同下标 j*（``slice.py``），被拒后缀位置的 hidden 不进上下文。
- **一位滞后**（plans/12 §2）：训练时 ctx_lens = a+1，掩码 ``t < a+1`` 使 H_ctx
  **含锚点自身 hidden**；decode 时锚点 y 已发射但未被 target 消费，其 hidden 不
  存在，缓冲只能到最后被消费位置（训练口径的 ctx_lens = a）。本模块按 decode
  可得口径喂滞后一位的 H_ctx；对 α̂ 的影响由 ``bench_neural_c5.py --align-check``
  量化（结论见 notes/step10_c5_report.md §2）。
- **并行草稿 + 串行头**（plans/12 §3）：每轮一次 drafter 前向
  （``[emb(anchor), mask×(γ−1)]`` + H_ctx KV 前缀）得并行 logits U_1..U_γ
  （``trunk_forward``，**不含** Markov 偏置）；随后串行采样
  ``q_k = softmax(U_k + B[x_{k−1}])``，经与 target 相同的 top_k 变换（复用
  ``loop._q_prime_from_raw``）采 x_k。置信度头逐位记录（本轮默认不截断）。
- 数值约定：trunk 全部 fp32（与训练一致）；``serial_sample`` 的 q′/采样与
  ``loop._propose_draft`` 同契约（采样返回 q′ 行、贪心返回原始 q 行，
  rng 逐位消耗顺序一致）。``forced_tokens`` 供单测做 teacher-forced 对拍：
  与训练侧 ``Drafter.forward`` 在同输入上逐位一致。

本模块引入 torch（经 ``train.drafter`` 与 ``specdec.block.loop``），CPU 套件请勿 import。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from specdec.block.loop import _q_prime_from_raw
from specdec.verifier import sample_categorical
from train.drafter import (
    SCHEME_LAYERS,
    VOCAB_SIZE,
    Drafter,
    build_kv_prefix_mask,
    make_block_ids,
)


class HiddenCapture:
    """注入层输出的滚动捕获缓冲（armed 协议：未 begin 的前向一律忽略）。

    - ``begin()``：清空本轮槽位并开始记录；
    - hook：仅在 armed 时把层输出（tuple 取第 0 元，与 evo2 ``return_embeddings``
      的 hook 同口径）detach 进本轮槽位；
    - ``commit(n_keep)``：本轮每层拼接后取前 ``n_keep`` 个位置（= 本轮被消费的
      token 数，块内下标 0..j*），fp32 追加进滚动缓冲并截到 ``maxlen``；
    - ``discard()``：丢弃本轮槽位（回退保护）。

    缓冲只含**已消费**位置的 hidden——被拒后缀在 commit 时被 ``n_keep`` 剔除。
    """

    def __init__(self, model, layer_names: tuple[str, ...], maxlen: int):
        self.layer_names = tuple(layer_names)
        self.maxlen = int(maxlen)
        self.total_committed = 0  # 累计已消费位置数（不受 maxlen 截断影响）
        self._buffers: dict[str, torch.Tensor] = {}
        self._slot: dict[str, list[torch.Tensor]] = {}
        self._armed = False
        self._handles = []
        for name in self.layer_names:
            module = model.get_submodule(name)
            self._handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            if not self._armed:
                return
            if isinstance(output, tuple):
                output = output[0]
            self._slot[name].append(output.detach())

        return hook

    def begin(self) -> None:
        self._slot = {name: [] for name in self.layer_names}
        self._armed = True

    def commit(self, n_keep: int) -> None:
        if not self._armed:
            raise RuntimeError("HiddenCapture.commit 前须先 begin")
        self._armed = False
        n_keep = int(n_keep)
        self.total_committed += n_keep
        for name in self.layer_names:
            pieces = self._slot.pop(name, [])
            if not pieces:
                raise RuntimeError(f"层 {name} 本轮未捕获到任何输出")
            cat = torch.cat(pieces, dim=1)[0]  # [T, H]
            if n_keep > cat.shape[0]:
                raise ValueError(f"n_keep={n_keep} 超过本轮捕获长度 {cat.shape[0]}")
            if n_keep > 0:
                # 滚动窗只保留最近 maxlen 位：仅转换能进窗的末段（与整段转换
                # 逐位等价），长 prompt prefill 轮（n_keep≈L）省下整段 fp32 峰值
                take = min(n_keep, self.maxlen)
                kept = cat[n_keep - take : n_keep].float()
                prev = self._buffers.get(name)
                buf = kept if prev is None else torch.cat([prev, kept], dim=0)
                self._buffers[name] = buf[-self.maxlen :]
        self._slot = {}

    def discard(self) -> None:
        self._armed = False
        self._slot = {}

    def context_tensor(self, window: int) -> torch.Tensor | None:
        """滚动缓冲 → ``[T, n_inject*H]``（按 ``layer_names`` 序拼接，近 ``window`` 个）。

        无注入层（S0）返回 ``None``；缓冲为空时 T=0（首 token 之前不存在——
        循环首轮前必有 prefill commit，此处仅防御）。
        """
        if not self.layer_names:
            return None
        pieces = []
        for name in self.layer_names:
            buf = self._buffers.get(name)
            if buf is None:
                raise RuntimeError(f"层 {name} 缓冲为空：prefill 尚未 commit")
            pieces.append(buf[-int(window) :])
        return torch.cat(pieces, dim=-1)

    @property
    def buffer_len(self) -> int:
        if not self.layer_names:
            return 0
        buf = self._buffers.get(self.layer_names[0])
        return 0 if buf is None else int(buf.shape[0])

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def trunk_forward(
    drafter: Drafter,
    anchor_ids: torch.Tensor,
    h_raw: torch.Tensor | None,
    ctx_lens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """并行 trunk：``Drafter.forward`` 去掉 Markov 偏置与 conf 头的同构实现。

    返回 ``(U, h)``：``U [B, γ, V]`` 为加偏置**之前**的 logits（fp32），
    ``h [B, γ, d]`` 为 final_ln 后的 trunk 隐状态（conf 头用）。
    与 ``Drafter.forward`` 逐算子同序，同输入下逐位一致（单测钉死）。
    """
    bsz = int(anchor_ids.shape[0])
    gamma = drafter.gamma
    draft_ids = make_block_ids(anchor_ids, gamma, drafter.mask_id)
    tok_emb = drafter.embed(draft_ids)  # [B, γ, H] 冻结
    h = drafter.in_proj(tok_emb.float())
    pos_ix = torch.arange(gamma, device=h.device).unsqueeze(0).expand(bsz, -1)
    h = h + drafter.pos(pos_ix)
    h_ctx = drafter.project_context(h_raw, bsz)
    mask = build_kv_prefix_mask(ctx_lens, gamma, int(h_ctx.shape[1]), ctx_window=drafter.ctx_window)
    for layer in drafter.layers:
        h = layer(h, h_ctx, mask)
    h = drafter.final_ln(h)
    U = F.linear(drafter.out_proj(h), drafter.embed.weight) * drafter.logit_scale
    return U, h


def serial_sample(
    drafter: Drafter,
    U: torch.Tensor,
    h: torch.Tensor,
    anchor_id: int,
    greedy: bool,
    rng: np.random.Generator,
    *,
    temperature: float = 1.0,
    top_k: int | None = 4,
    forced_tokens: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """串行 Markov 头：``q_k = softmax(U_k + B[x_{k−1}])``，x_0 的前驱 = 锚点。

    采样路径：q 经 ``_q_prime_from_raw``（与 target 相同的 top_k 变换）得 q′，
    从 q′ 采样，返回的 ``q_rows`` 即 q′（verify 契约）。贪心路径不变换、
    argmax（最小 index tie-break），``q_rows`` 为原始 q。
    ``forced_tokens``（长度 γ）：逐位强制草稿取值（teacher-forced 对拍用），
    此时不消耗 rng。

    返回 ``(tokens[γ], q_rows[γ, V], confs[γ])``；conf 用最终前驱序列
    ``[anchor, x_1..x_{γ−1}]`` 一次性计算，与训练侧 ``Drafter.forward`` 同形。
    """
    gamma = drafter.gamma
    if U.shape != (1, gamma, VOCAB_SIZE):
        raise ValueError(f"U 须为 [1, {gamma}, {VOCAB_SIZE}]，得到 {tuple(U.shape)}")
    if forced_tokens is not None and len(forced_tokens) != gamma:
        raise ValueError(f"forced_tokens 长度须为 γ={gamma}")

    tokens = np.empty(gamma, dtype=np.int64)
    q_rows = np.empty((gamma, VOCAB_SIZE), dtype=np.float64)
    prev = int(anchor_id)
    for k in range(gamma):
        logits_k = U[0, k] + drafter.markov[prev]  # [V] fp32
        q_k = torch.softmax(logits_k.float(), dim=-1)
        q_np = q_k.detach().cpu().numpy().astype(np.float64, copy=False)
        if greedy:
            tok = int(np.argmax(q_np))
            row = q_np
        else:
            row = _q_prime_from_raw(q_np, temperature=temperature, top_k=top_k)
            tok = sample_categorical(row, rng)
        if forced_tokens is not None:
            tok = int(forced_tokens[k])
        tokens[k] = tok
        q_rows[k] = row
        prev = tok

    prev_ids = torch.empty(gamma, dtype=torch.long, device=U.device)
    prev_ids[0] = int(anchor_id)
    if gamma > 1:
        prev_ids[1:] = torch.as_tensor(tokens[:-1], dtype=torch.long, device=U.device)
    prev_emb = drafter.in_proj(drafter.embed(prev_ids).float())  # [γ, d]
    conf = torch.sigmoid(drafter.conf_w(torch.cat([h[0], prev_emb], dim=-1)).squeeze(-1))
    confs = conf.detach().float().cpu().numpy().astype(np.float64, copy=False)
    return tokens, q_rows, confs


class NeuralDraftModel:
    """接入投机循环的神经 drafter（接口由 ``loop.speculative_generate`` 鸭子识别）。

    循环侧协议：prefill 前 ``capture_begin()`` / 后 ``capture_commit(L−1)``；
    每轮块前向前 ``capture_begin()`` / 验证切片后 ``capture_commit(j*+1)``；
    草稿经 ``propose_block(y, γ, greedy, rng, ...)``。仅支持 ``rollback="slice"``
    （拒绝剔除与状态切片共用下标 j*）。
    """

    def __init__(
        self,
        model,
        drafter: Drafter,
        layer_names: tuple[str, ...],
        *,
        ctx_maxlen: int | None = None,
    ):
        self.drafter = drafter
        self.layer_names = tuple(layer_names)
        self.gamma = int(drafter.gamma)
        maxlen = int(ctx_maxlen or drafter.ctx_window)
        self.capture = HiddenCapture(model, self.layer_names, maxlen=maxlen)

    @classmethod
    def from_checkpoint(
        cls,
        model,
        ckpt_path: str,
        *,
        device: str | torch.device = "cuda:0",
    ) -> "NeuralDraftModel":
        """从 Step 8/9 checkpoint 构建（含冻结 embed/markov/conf 全量权重）。"""
        ck = torch.load(ckpt_path, map_location="cpu")
        drafter = Drafter.from_scheme(ck["scheme"], d_model=ck["d_model"], gamma=ck["gamma"])
        drafter.load_state_dict(ck["state_dict"])
        drafter = drafter.to(torch.device(device))
        drafter.eval()
        for p in drafter.parameters():
            p.requires_grad_(False)
        inst = cls(model, drafter, SCHEME_LAYERS[ck["scheme"]])
        inst.ckpt_meta = {k: v for k, v in ck.items() if k != "state_dict"}
        return inst

    # -- 循环协议 ----------------------------------------------------------

    def capture_begin(self) -> None:
        self.capture.begin()

    def capture_commit(self, n_keep: int) -> None:
        self.capture.commit(n_keep)

    def capture_buffer_len(self) -> int:
        """累计已消费位置数（外部状态对齐自检，``loop`` 长 prompt 分块预填路径）。"""
        return int(self.capture.total_committed)

    def capture_reset(self) -> None:
        """清空 H_ctx 滚动缓冲与已消费计数。**复用同一实例做多次独立 generate
        前必须调用**（否则上一序列的上下文会泄进下一次的 H_ctx/对齐自检）。"""
        self.capture._buffers = {}
        self.capture.total_committed = 0

    @torch.no_grad()
    def propose_block(
        self,
        anchor_id: int,
        gamma: int,
        greedy: bool,
        rng: np.random.Generator,
        *,
        temperature: float = 1.0,
        top_k: int | None = 4,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """一轮并行草稿：返回 ``(draft_tokens[γ], q_rows[γ, V], confs[γ])``。"""
        if int(gamma) != self.gamma:
            raise ValueError(f"drafter 训练 γ={self.gamma}，不支持循环 γ={gamma}")
        device = self.drafter.pos.weight.device
        h_raw = self.capture.context_tensor(self.drafter.ctx_window)
        ctx_len = 0 if h_raw is None else int(h_raw.shape[0])
        ctx_lens = torch.tensor([ctx_len], dtype=torch.long, device=device)
        anchor = torch.tensor([int(anchor_id)], dtype=torch.long, device=device)
        U, h = trunk_forward(self.drafter, anchor, h_raw, ctx_lens)
        return serial_sample(
            self.drafter,
            U,
            h,
            int(anchor_id),
            greedy,
            rng,
            temperature=temperature,
            top_k=top_k,
        )

    def close(self) -> None:
        self.capture.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


__all__ = [
    "HiddenCapture",
    "NeuralDraftModel",
    "serial_sample",
    "trunk_forward",
]
