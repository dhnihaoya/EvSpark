"""DSpark 改装并行 drafter：2 层因果骨架 + 目标隐层 KV 前缀 + Markov / 置信度头。

接口与损失对齐 ``plans/10_phaseC_smoke.md`` §2–3、§5。不依赖 evo2/vortex，
便于 CPU 单测；冻结 embedding 由调用方写入（训练时从 7B 拷贝）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB_SIZE = 512
TARGET_HIDDEN = 4096
MASK_ID = 256  # 512 词表空闲行；非 ACGTN / eod / pad
DEFAULT_GAMMA = 7
CE_W = 0.1
TV_W = 0.9
CONF_W = 1.0
HEAD_DIM = 64

# 注入方案：hook 名 = ModuleList 路径，get_submodule 可取
SCHEME_LAYERS: dict[str, tuple[str, ...]] = {
    "S0": (),
    "S1": ("blocks.31",),
    "S2": ("blocks.7", "blocks.15", "blocks.23", "blocks.31"),
    "S3": ("blocks.6", "blocks.16", "blocks.27"),
    # Step 9（plans/11）：S3 三个 HCL 各自单独注入的单层扫描
    "L6": ("blocks.6",),
    "L16": ("blocks.16",),
    "L27": ("blocks.27",),
    # Step 13b（plans/16）：补齐其余 HCL 单层 + 两种双层组合
    "L20": ("blocks.20",),
    "L23": ("blocks.23",),
    "L30": ("blocks.30",),
    "D16_27": ("blocks.16", "blocks.27"),
    "D27_30": ("blocks.27", "blocks.30"),
    # Step 16 追加（plans/19）：真实预算层选择扫描——MHA 中段 / HCS / HCM / HCL+MHA 互补
    "L17": ("blocks.17",),
    "L28": ("blocks.28",),
    "L29": ("blocks.29",),
    "D27_31": ("blocks.27", "blocks.31"),
    # 40B（50 层）HCL 对位：L45 ≈ 7B L27（倒数第二拍 HCL），L48 ≈ 7B L30
    "L34": ("blocks.34",),
    "L38": ("blocks.38",),
    "L41": ("blocks.41",),
    "L45": ("blocks.45",),
    "L48": ("blocks.48",),
}


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return self.scale * (x / rms)


def build_kv_prefix_mask(
    ctx_lens: torch.Tensor,
    gamma: int,
    ctx_max: int,
    ctx_window: int = 0,
) -> torch.Tensor:
    """位置 k 可见：合法 H_ctx（t < ctx_len）+ 因果 draft 位（j <= k）。

    ``ctx_window>0`` 时只保留锚点前 ``ctx_window`` 个 H_ctx token，避免 2k
    长前缀把 7 个 draft query 的 softmax 稀释掉（全前缀时 S0≈S1）。

    Returns
    -------
    mask : BoolTensor [B, 1, gamma, ctx_max + gamma]
        True = 允许注意。第 0 维可广播到 n_heads。
    """
    bsz = int(ctx_lens.shape[0])
    device = ctx_lens.device
    t = torch.arange(ctx_max, device=device).view(1, 1, ctx_max)
    cl = ctx_lens.view(bsz, 1, 1).to(device)
    ctx_ok = t < cl
    if ctx_window and int(ctx_window) > 0:
        lo = (cl - int(ctx_window)).clamp(min=0)
        ctx_ok = ctx_ok & (t >= lo)
    q = torch.arange(gamma, device=device).view(1, gamma, 1)
    j = torch.arange(gamma, device=device).view(1, 1, gamma)
    draft_ok = j <= q  # [1, gamma, gamma]
    draft_ok = draft_ok.expand(bsz, -1, -1)
    ctx_ok = ctx_ok.expand(-1, gamma, -1)
    mask = torch.cat([ctx_ok, draft_ok], dim=-1)  # [B, gamma, T+gamma]
    return mask.unsqueeze(1)


def make_block_ids(anchor_ids: torch.Tensor, gamma: int, mask_id: int = MASK_ID) -> torch.Tensor:
    """``[emb(x_anchor), emb(mask)×(γ−1)]`` 的 token id，shape [B, gamma]。"""
    bsz = int(anchor_ids.shape[0])
    ids = torch.full(
        (bsz, gamma),
        int(mask_id),
        dtype=torch.long,
        device=anchor_ids.device,
    )
    ids[:, 0] = anchor_ids.to(dtype=torch.long)
    return ids


def position_weights(gamma: int, device=None, dtype=None) -> torch.Tensor:
    """w_k = exp(−(k−1)/γ)，k = 1..γ，返回 shape [gamma]。"""
    k = torch.arange(gamma, device=device, dtype=dtype)
    return torch.exp(-k / float(gamma))


def apply_transform_torch(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = 4,
) -> torch.Tensor:
    """与 ``specdec.transforms.apply_transform`` 同口径的 batched 实现。

    顺序：logits/T → softmax → top_k 置零再归一化。并列时较小 index 优先
    （stable sort，原始列序 0..V-1）。
    """
    if logits.ndim < 1:
        raise ValueError(f"logits 维数过低: {tuple(logits.shape)}")
    vocab = int(logits.shape[-1])
    flat = logits.float().reshape(-1, vocab)
    if temperature <= 0.0:
        out = torch.zeros_like(flat)
        out.scatter_(1, flat.argmax(dim=-1, keepdim=True), 1.0)
        return out.reshape(logits.shape)
    scaled = flat / float(temperature)
    scaled = scaled - scaled.amax(dim=-1, keepdim=True)
    probs = torch.softmax(scaled, dim=-1)
    if top_k is None or top_k <= 0 or top_k >= vocab:
        return probs.reshape(logits.shape)
    # stable 降序：并列时保留较小 index（原始列序）
    _, order = torch.sort(probs, dim=-1, descending=True, stable=True)
    keep = order[:, : int(top_k)]
    masked = torch.zeros_like(probs)
    masked.scatter_(1, keep, probs.gather(1, keep))
    denom = masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return (masked / denom).reshape(logits.shape)


def analytic_accept(p_d: torch.Tensor, p_t: torch.Tensor) -> torch.Tensor:
    """c* = 1 − ½‖p_d − p_t‖₁，shape 与前缀维一致（最后一维是词表）。"""
    l1 = (p_d.float() - p_t.float()).abs().sum(dim=-1)
    return 1.0 - 0.5 * l1


def predicted_tau(alpha_bar: torch.Tensor) -> torch.Tensor:
    """τ̂ = 1 + Σ_j Π_{i≤j} ᾱ_i（含 bonus；全 1 时 = 1+γ）。近似公式。"""
    alpha = alpha_bar.float().reshape(-1)
    if alpha.numel() == 0:
        return alpha.new_tensor(1.0)
    cprod = torch.cumprod(alpha, dim=0)
    return 1.0 + cprod.sum()


@dataclass
class LossBreakdown:
    total: torch.Tensor
    ce: torch.Tensor
    tv: torch.Tensor
    conf: torch.Tensor
    w_sum: torch.Tensor


def distill_losses(
    logits: torch.Tensor,
    p_t: torch.Tensor,
    target_ids: torch.Tensor,
    conf_pred: torch.Tensor,
    *,
    p_d: torch.Tensor | None = None,
    ce_w: float = CE_W,
    tv_w: float = TV_W,
    conf_w: float = CONF_W,
) -> LossBreakdown:
    """L = 0.1·CE + 0.9·TV + 1.0·conf；位置加权后对 batch 取均值。

    CE 必须走 ``log_softmax(logits)``：先 softmax 再 ``log(clamp(p))`` 会在
    下溢的 p=0 处把梯度掐死。TV/conf 仍用概率。c* 对 p_d detach。
    """
    if p_d is None:
        p_d = torch.softmax(logits.float(), dim=-1)
    if p_d.shape != p_t.shape:
        raise ValueError(f"p_d/p_t shape 不一致: {tuple(p_d.shape)} vs {tuple(p_t.shape)}")
    gamma = int(p_d.shape[1])
    w = position_weights(gamma, device=logits.device, dtype=torch.float32)
    log_pd = F.log_softmax(logits.float(), dim=-1)
    gather_ix = target_ids.long().unsqueeze(-1)
    ce_k = -log_pd.gather(-1, gather_ix).squeeze(-1)
    tv_k = (p_d.float() - p_t.float()).abs().sum(dim=-1)
    c_star = (1.0 - 0.5 * tv_k).clamp(0.0, 1.0).detach()
    conf_k = F.binary_cross_entropy(conf_pred.float().clamp(1e-6, 1.0 - 1e-6), c_star, reduction="none")
    ce = (ce_k * w).sum(dim=-1).mean()
    tv = (tv_k * w).sum(dim=-1).mean()
    conf = (conf_k * w).sum(dim=-1).mean()
    total = ce_w * ce + tv_w * tv + conf_w * conf
    return LossBreakdown(total=total, ce=ce, tv=tv, conf=conf, w_sum=w.sum())


class DraftLayer(nn.Module):
    """Pre-LN 因果注意力：K/V = [W H_ctx ; W H_d]（序列维拼接）。"""

    def __init__(self, d_model: int, n_heads: int, mlp_mult: int = 4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} 不能整除 n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ln1 = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.ln2 = RMSNorm(d_model)
        hidden = mlp_mult * d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, _ = x.shape
        return x.view(bsz, seq, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        bsz, n_heads, seq, _ = x.shape
        return x.transpose(1, 2).contiguous().view(bsz, seq, n_heads * self.head_dim)

    def forward(
        self,
        h_d: torch.Tensor,
        h_ctx: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        h_d : [B, γ, d]
        h_ctx : [B, T, d]（已是共享投影 + RMSNorm 后的 H_ctx；T=0 表示无注入）
        mask : [B, 1, γ, T+γ] True=可见
        """
        bsz, gamma, _ = h_d.shape
        ctx_len = int(h_ctx.shape[1])
        x = self.ln1(h_d)
        q = self._split(self.q_proj(x))
        k_d = self.k_proj(x)
        v_d = self.v_proj(x)
        if ctx_len == 0:
            k = self._split(k_d)
            v = self._split(v_d)
        else:
            k_ctx = self.k_proj(h_ctx)
            v_ctx = self.v_proj(h_ctx)
            k = self._split(torch.cat([k_ctx, k_d], dim=1))
            v = self._split(torch.cat([v_ctx, v_d], dim=1))
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        y = self._merge(torch.matmul(attn, v))
        h_d = h_d + self.o_proj(y)
        h_d = h_d + self.mlp(self.ln2(h_d))
        return h_d


class Drafter(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_layers: int = 2,
        gamma: int = DEFAULT_GAMMA,
        n_inject: int = 1,
        vocab_size: int = VOCAB_SIZE,
        target_hidden: int = TARGET_HIDDEN,
        mask_id: int = MASK_ID,
        n_heads: int | None = None,
        ctx_window: int = 64,
    ):
        super().__init__()
        if n_heads is None:
            n_heads = max(1, d_model // HEAD_DIM)
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} 不能整除 n_heads={n_heads}")
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.gamma = int(gamma)
        self.n_inject = int(n_inject)
        self.vocab_size = int(vocab_size)
        self.target_hidden = int(target_hidden)
        self.mask_id = int(mask_id)
        self.n_heads = int(n_heads)
        self.ctx_window = int(ctx_window)

        self.embed = nn.Embedding(self.vocab_size, self.target_hidden)
        self.embed.weight.requires_grad_(False)
        self.in_proj = nn.Linear(self.target_hidden, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.target_hidden)
        self.pos = nn.Embedding(self.gamma, self.d_model)
        if self.n_inject > 0:
            self.ctx_proj = nn.Linear(self.n_inject * self.target_hidden, self.d_model)
            self.ctx_norm = RMSNorm(self.d_model)
        else:
            self.ctx_proj = None
            self.ctx_norm = None
        self.layers = nn.ModuleList(
            [DraftLayer(self.d_model, self.n_heads) for _ in range(self.n_layers)]
        )
        self.final_ln = RMSNorm(self.d_model)
        # 全矩阵 Markov bias，零初始化
        self.markov = nn.Parameter(torch.zeros(self.vocab_size, self.vocab_size))
        self.conf_w = nn.Linear(2 * self.d_model, 1)
        # 避免随机 embedding 时 logits 爆炸（softmax 下溢）；7B embedding 同此缩放仍可学
        nn.init.zeros_(self.out_proj.bias)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        self.logit_scale = self.target_hidden ** -0.5

    @classmethod
    def from_scheme(
        cls,
        scheme: str,
        d_model: int,
        gamma: int = DEFAULT_GAMMA,
        **kwargs,
    ) -> "Drafter":
        if scheme not in SCHEME_LAYERS:
            raise ValueError(f"未知 scheme={scheme}，可选 {tuple(SCHEME_LAYERS)}")
        n_inject = len(SCHEME_LAYERS[scheme])
        return cls(d_model=d_model, gamma=gamma, n_inject=n_inject, **kwargs)

    def load_frozen_embedding(self, weight: torch.Tensor) -> None:
        if tuple(weight.shape) != tuple(self.embed.weight.shape):
            raise ValueError(
                f"embedding 形状 {tuple(weight.shape)} != {tuple(self.embed.weight.shape)}"
            )
        with torch.no_grad():
            self.embed.weight.copy_(weight.detach().to(self.embed.weight.dtype))
        self.embed.weight.requires_grad_(False)

    def trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def frozen_named_tensors(self) -> dict[str, torch.Tensor]:
        return {"embed.weight": self.embed.weight}

    def project_context(self, h_raw: torch.Tensor | None, batch: int) -> torch.Tensor:
        """共享 [T,H]/[1,T,H] 或独立 [B,T,H] 上下文 → [B,T,d]。"""
        if self.ctx_proj is None or h_raw is None:
            return torch.zeros(batch, 0, self.d_model, device=self.pos.weight.device, dtype=self.pos.weight.dtype)
        if h_raw.ndim == 3:
            if h_raw.shape[0] not in (1, batch):
                raise ValueError(f"上下文 batch={h_raw.shape[0]} 与 batch={batch} 不兼容")
            h_ctx = self.ctx_norm(self.ctx_proj(h_raw.float()))
            return h_ctx.expand(batch, -1, -1)
        if h_raw.ndim != 2:
            raise ValueError(f"上下文须为 [T,H] 或 [B,T,H]，得到 {tuple(h_raw.shape)}")
        h_ctx = self.ctx_norm(self.ctx_proj(h_raw.float()))
        return h_ctx.unsqueeze(0).expand(batch, -1, -1)

    def forward(
        self,
        anchor_ids: torch.Tensor,
        prev_ids: torch.Tensor,
        ctx_lens: torch.Tensor,
        h_raw: torch.Tensor | None = None,
        draft_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        anchor_ids : [B]
        prev_ids : [B, γ]  各位的前一 token（teacher forcing；第 0 位 = anchor）
        ctx_lens : [B]     可见 H_ctx 长度（= anchor 位 + 1）
        h_raw : [T, n_inject·4096]  整窗拼接隐层；S0 可 None
        draft_ids : [B, γ]  覆盖默认 [anchor, mask...]（单测用）

        Returns
        -------
        logits, probs, conf, hidden  形状 [B, γ, V] / [B, γ, V] / [B, γ] / [B, γ, d]
        """
        bsz = int(anchor_ids.shape[0])
        gamma = self.gamma
        if draft_ids is None:
            draft_ids = make_block_ids(anchor_ids, gamma, self.mask_id)
        if draft_ids.shape != (bsz, gamma):
            raise ValueError(f"draft_ids 期望 {[bsz, gamma]}，得到 {tuple(draft_ids.shape)}")
        tok_emb = self.embed(draft_ids)  # [B, γ, 4096] 无梯度
        h = self.in_proj(tok_emb.float())
        pos_ix = torch.arange(gamma, device=h.device).unsqueeze(0).expand(bsz, -1)
        h = h + self.pos(pos_ix)
        h_ctx = self.project_context(h_raw, bsz)
        ctx_max = int(h_ctx.shape[1])
        mask = build_kv_prefix_mask(ctx_lens, gamma, ctx_max, ctx_window=self.ctx_window)
        for layer in self.layers:
            h = layer(h, h_ctx, mask)
        h = self.final_ln(h)
        logits = F.linear(self.out_proj(h), self.embed.weight) * self.logit_scale
        logits = logits + self.markov[prev_ids.long()]
        probs = torch.softmax(logits.float(), dim=-1)
        prev_emb = self.in_proj(self.embed(prev_ids.long()).float())
        conf = torch.sigmoid(self.conf_w(torch.cat([h, prev_emb], dim=-1)).squeeze(-1))
        return logits, probs, conf, h
