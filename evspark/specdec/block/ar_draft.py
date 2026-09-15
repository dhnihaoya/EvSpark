"""自回归 drafter（Phase 1 D1）：First-N 截断自草稿 + 独立小模型。

两条臂都走 ``propose_block`` 鸭子接口（与神经 drafter 并列），**没有**
``capture_begin``——循环据此与 HiddenCapture 路径分叉。

- **FirstNDraftModel**：target 前 N 块 + final norm + unembed（BioSpecDec 同款）。
  前 N 层 FIR/IIR/KV **与 target 共享一份** inference_params；propose 前快照、
  逐步推进、返回前恢复，避免 draft 污染后续块验证。Hyena 在 prefill 之后
  对多 token 会 ``u[:, -1]`` 静默截断，故 First-N 逐步必须 **T=1**。
- **IndependentARDraftModel**：另一份模型（``evo2_1b_base``）自管状态。
  ``prefill_prompt`` 吃完整 prompt（含锚点 y）；propose 从缓存末位 logits
  采 γ 步后恢复快照；``commit_tokens`` 再步进 accepted 前缀。1B 上下文 8k，
  超限直接报「无资格」。

本模块引入 torch + vortex，CPU 套件请勿 import。
"""

from __future__ import annotations

import numpy as np
import torch

from evspark.specdec.block.driver import (
    PARAM_KEYS,
    clone_inference_params,
    prefill,
    step_forward_reference,
    step_seqlen_offsets,
)
from evspark.specdec.block.loop import VOCAB_SIZE, _logits_rows, _q_prime_from_raw
from evspark.specdec.verifier import sample_categorical

_HYENA_KEYS = ("hcl", "hcm", "hcs")
_HYENA_ATTRS = ("fir_state_dict", "fir_inner_state_dict", "state_dict")

# evo2_1b_base 官方 max_seqlen；超此标无资格，不要硬跑。
Evo2_1B_MAX_CONTEXT = 8192


def _q_and_token_from_logits(
    logits_np: np.ndarray,
    greedy: bool,
    rng: np.random.Generator,
    temperature: float,
    top_k: int | None,
) -> tuple[np.ndarray, int]:
    """logits [V] → (q_row, token)。贪心：原始 softmax + argmax；采样：q′。"""
    logits_np = np.asarray(logits_np, dtype=np.float64).ravel()
    if logits_np.shape[0] != VOCAB_SIZE:
        raise ValueError(f"logits 须为 [{VOCAB_SIZE}]，得到 {logits_np.shape}")
    z = logits_np - float(np.max(logits_np))
    q = np.exp(z)
    q /= q.sum()
    if greedy:
        return q, int(np.argmax(q))
    row = _q_prime_from_raw(q, temperature=temperature, top_k=top_k)
    return row, int(sample_categorical(row, rng))


def prefix_layer_indices(n_blocks: int) -> tuple[int, ...]:
    return tuple(range(int(n_blocks)))


def snapshot_prefix_states(ip: dict, layer_indices: tuple[int, ...]) -> dict:
    """只快照前 N 层可能被 First-N 改写的状态 + 四键 seqlen_offset。

    MHA 的 seqlen_offset 全层共享；不恢复就会让未跑到的深层 attention 误读 KV。
    """
    layers = tuple(int(i) for i in layer_indices)
    snap: dict = {
        "seqlen_offsets": {k: int(ip[k].seqlen_offset) for k in PARAM_KEYS},
        "mha_kv": {},
        "mha_lengths": None,
    }
    mha = ip["mha"]
    for li in layers:
        t = mha.key_value_memory_dict.get(li)
        if t is not None:
            snap["mha_kv"][li] = t.detach().clone()
    if mha.lengths_per_sample is not None:
        snap["mha_lengths"] = mha.lengths_per_sample.detach().clone()
    for key in _HYENA_KEYS:
        entry: dict = {}
        params = ip[key]
        for attr in _HYENA_ATTRS:
            d = getattr(params, attr, None)
            if not d:
                continue
            sub = {}
            for li in layers:
                if li in d:
                    sub[li] = d[li].detach().clone()
            if sub:
                entry[attr] = sub
        snap[key] = entry
    return snap


def restore_prefix_states(ip: dict, snap: dict) -> None:
    """把 First-N 推进过的前 N 层状态与 offset 写回快照（in-place）。"""
    for k, off in snap["seqlen_offsets"].items():
        ip[k].seqlen_offset = int(off)
    mha = ip["mha"]
    for li, t in snap["mha_kv"].items():
        dst = mha.key_value_memory_dict.get(li)
        if dst is None:
            mha.key_value_memory_dict[li] = t.clone()
        else:
            dst.copy_(t)
    if snap["mha_lengths"] is not None and mha.lengths_per_sample is not None:
        mha.lengths_per_sample.copy_(snap["mha_lengths"])
    for key in _HYENA_KEYS:
        params = ip[key]
        for attr, sub in snap[key].items():
            d = getattr(params, attr)
            for li, t in sub.items():
                d[int(li)] = t.clone()


@torch.inference_mode()
def first_n_forward(model, input_ids: torch.Tensor, ip: dict, n_blocks: int) -> torch.Tensor:
    """embedding → blocks[:N] → final norm → unembed。返回 logits [1, T, V]。

    ``T`` 必须为 1：prefill 之后 Hyena ``sequential_forward`` 会静默丢弃多 token。
    """
    if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        raise ValueError(f"input_ids 须为 [1, T]，得到 {tuple(input_ids.shape)}")
    if int(input_ids.shape[1]) != 1:
        raise ValueError(
            "first_n_forward 仅支持单 token（prefill 后 Hyena sequential 会静默截断多 token）"
        )
    n_blocks = int(n_blocks)
    n_layers = int(len(model.blocks))
    if not 1 <= n_blocks <= n_layers:
        raise ValueError(f"n_blocks={n_blocks} 越界：须 1..{n_layers}")

    x = model.embedding_layer(input_ids)
    for block_idx in range(n_blocks):
        block = model.blocks[block_idx]
        params = ip[model.block_idx_to_name(block_idx)]
        x = model.cross_device_transfer(x, block_idx)
        x, _ = block(x, inference_params=params)
    x = x.to(model.block_idx_to_device[0])
    if model.norm is not None:
        x = model.norm(x)
    x = model.unembed(x)
    return x


class FirstNDraftModel:
    """截断自草稿：前 N 层与 target 共享状态，propose 内快照/恢复。

    循环侧：有 ``propose_block``、无 ``capture_begin``。需要循环把 target 的
    ``inference_params_dict`` 传进 ``propose_block``。
    """

    def __init__(self, model, n_blocks: int):
        n_layers = int(len(model.blocks))
        n_blocks = int(n_blocks)
        if not 1 <= n_blocks <= n_layers:
            raise ValueError(f"n_blocks={n_blocks} 越界：须 1..{n_layers}")
        self.model = model
        self.n_blocks = n_blocks
        self.layer_indices = prefix_layer_indices(n_blocks)

    def reset(self) -> None:
        return None

    @torch.inference_mode()
    def propose_block(
        self,
        anchor_id: int,
        gamma: int,
        greedy: bool,
        rng: np.random.Generator,
        *,
        temperature: float = 1.0,
        top_k: int | None = 4,
        inference_params_dict: dict | None = None,
        **_kwargs,
    ) -> tuple[np.ndarray, np.ndarray, None]:
        if inference_params_dict is None:
            raise ValueError("First-N drafter 需要循环传入 target 的 inference_params_dict")
        if int(gamma) < 1:
            raise ValueError("gamma 须 ≥ 1")
        ip = inference_params_dict
        device = next(self.model.parameters()).device
        snap = snapshot_prefix_states(ip, self.layer_indices)
        tokens = np.empty(int(gamma), dtype=np.int64)
        q_rows = np.empty((int(gamma), VOCAB_SIZE), dtype=np.float64)
        try:
            x = torch.tensor([[int(anchor_id)]], dtype=torch.long, device=device)
            logits = first_n_forward(self.model, x, ip, self.n_blocks)
            step_seqlen_offsets(ip, 1)
            logits_np = _logits_rows(logits)[0]
            for i in range(int(gamma)):
                q_i, tok = _q_and_token_from_logits(
                    logits_np, greedy, rng, temperature, top_k
                )
                tokens[i] = tok
                q_rows[i] = q_i
                if i + 1 < int(gamma):
                    x = torch.tensor([[tok]], dtype=torch.long, device=device)
                    logits = first_n_forward(self.model, x, ip, self.n_blocks)
                    step_seqlen_offsets(ip, 1)
                    logits_np = _logits_rows(logits)[0]
        finally:
            restore_prefix_states(ip, snap)
        return tokens, q_rows, None


class IndependentARDraftModel:
    """独立模型自回归 drafter（Leviathan 小模型基线）。自管一份 inference_params。

    状态不变量：``ip`` 对应完整 ``seen``（含当前锚点 y），与 target 的
    「最后 token 未消费」差一位。``max_context`` 默认 8192（1B）；
    ``L + n_tokens > max_context`` 视为无资格。
    """

    def __init__(self, model, *, max_context: int = Evo2_1B_MAX_CONTEXT):
        self.model = model
        self.max_context = int(max_context)
        self.ip: dict | None = None
        self._last_logits: np.ndarray | None = None

    def reset(self) -> None:
        self.ip = None
        self._last_logits = None

    def prefill_prompt(self, prompt_ids: torch.Tensor, *, max_seqlen: int | None = None) -> None:
        """预填**完整** prompt（含 y），使 1B 状态对齐 ``seen``。"""
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids[None]
        if prompt_ids.ndim != 2 or int(prompt_ids.shape[0]) != 1:
            raise ValueError(f"prompt_ids 须为 [1, L]，得到 {tuple(prompt_ids.shape)}")
        L = int(prompt_ids.shape[1])
        cap = int(max_seqlen or (L + 8))
        if cap > self.max_context or L > self.max_context:
            raise ValueError(
                f"独立 drafter 无资格：需要 max_seqlen={cap}（prompt={L}）>"
                f"{self.max_context}（evo2_1b_base 仅 8k 上下文）"
            )
        self.ip = self.model.initialize_inference_params(max_seqlen=cap)
        with torch.inference_mode():
            logits = prefill(self.model, prompt_ids, self.ip)
        self._last_logits = _logits_rows(logits)[-1]

    @torch.inference_mode()
    def propose_block(
        self,
        anchor_id: int,
        gamma: int,
        greedy: bool,
        rng: np.random.Generator,
        *,
        temperature: float = 1.0,
        top_k: int | None = 4,
        inference_params_dict: dict | None = None,
        **_kwargs,
    ) -> tuple[np.ndarray, np.ndarray, None]:
        del inference_params_dict  # 独立模型，不用 target 的 ip
        del anchor_id  # 已含在 self.ip / _last_logits 里
        if self.ip is None or self._last_logits is None:
            raise RuntimeError("IndependentARDraftModel.propose_block 前须 prefill_prompt")
        if int(gamma) < 1:
            raise ValueError("gamma 须 ≥ 1")
        device = next(self.model.parameters()).device
        snap = clone_inference_params(self.ip)
        saved_logits = np.array(self._last_logits, copy=True)
        tokens = np.empty(int(gamma), dtype=np.int64)
        q_rows = np.empty((int(gamma), VOCAB_SIZE), dtype=np.float64)
        logits_np = self._last_logits
        try:
            for i in range(int(gamma)):
                q_i, tok = _q_and_token_from_logits(
                    logits_np, greedy, rng, temperature, top_k
                )
                tokens[i] = tok
                q_rows[i] = q_i
                if i + 1 < int(gamma):
                    x = torch.tensor([[tok]], dtype=torch.long, device=device)
                    step_logits = step_forward_reference(self.model, x, self.ip)
                    logits_np = _logits_rows(step_logits)[0]
        finally:
            self.ip = snap
            self._last_logits = saved_logits
        return tokens, q_rows, None

    @torch.inference_mode()
    def commit_tokens(self, taken: np.ndarray) -> None:
        """从轮初 ``seen`` 步进 accepted 前缀（含 bonus），更新末位 logits。"""
        if self.ip is None:
            raise RuntimeError("commit_tokens 前须 prefill_prompt")
        taken = np.asarray(taken, dtype=np.int64).ravel()
        if taken.size == 0:
            return
        device = next(self.model.parameters()).device
        with torch.inference_mode():
            x = torch.as_tensor(taken, dtype=torch.long, device=device)[None]
            logits = step_forward_reference(self.model, x, self.ip)
        self._last_logits = _logits_rows(logits)[-1]


def load_evo2_7b_bf16(*, local_path: str | None = None, use_kernels: bool = True):
    """在 TE 已装的 env 里仍以 **bf16** 加载 7B（生产口径，禁止 7B 走 FP8）。

    1B 对照臂必须与 v2 旗舰同一 target 数值路径，否则加速比不可比。
    """
    import os

    import pkgutil
    import yaml
    from vortex.model.model import StripedHyena
    from vortex.model.utils import dotdict, load_checkpoint
    from evo2.utils import CONFIG_MAP

    config = yaml.safe_load(pkgutil.get_data("evo2", CONFIG_MAP["evo2_7b"]))
    config = dotdict(config)
    config.use_fp8_input_projections = False
    if use_kernels:
        config.use_hcs_kernel = True
        config.use_hcm_kernel = True
        config.use_hcl_kernel = True
    model = StripedHyena(config)
    path = local_path
    if path is None:
        hf = os.environ.get("HF_HOME", "")
        for cand in (
            os.path.join(hf, "evo2_7b.pt") if hf else "",
            "models/evo2_7b/evo2_7b.pt",
        ):
            if cand and os.path.isfile(cand):
                path = cand
                break
    if not path or not os.path.isfile(path):
        raise FileNotFoundError("找不到 evo2_7b.pt（HF_HOME 或 models/evo2_7b/）")
    load_checkpoint(model, path)
    model.eval()
    return model


__all__ = [
    "Evo2_1B_MAX_CONTEXT",
    "FirstNDraftModel",
    "IndependentARDraftModel",
    "first_n_forward",
    "load_evo2_7b_bf16",
    "prefix_layer_indices",
    "restore_prefix_states",
    "snapshot_prefix_states",
]
