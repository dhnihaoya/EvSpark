"""端到端投机解码循环（Plan 02 A.6 / Step 5 + A.3 / Step 7）。

循环结构（plans/07 §2，语义红线）：

- 已发射且已被模型消费的序列记为 E；锚点 y = 最新发射但尚未被模型消费的 token。
- 首轮引导：prompt 只 prefill 前 L−1 个，第 L 个作为 y₀，此后每轮形状统一。
- 稳态轮：drafter 看见 E+y 后提出 γ 个草稿；块输入 = ``[y, x₁..x_γ]``（γ+1），
  ``block_forward`` 输出 ``[γ+1, V]`` 即验证器契约的 ``target_probs``，无需拼接。
- 采样路径：draft 原始 q 经与 target 相同的 ``apply_transform`` 得 q′，从 q′ 采样，
  ``verify_round`` 收到的 ``draft_probs`` 即 q′。贪心路径不变换。
- 验证必须直接调用 Step 1 的 ``verify_round`` / ``verify_round_greedy``，禁止重写。
- 回滚（``rollback``）：
  - ``"slice"``（默认，Step 7）：块前向保留逐位置中间态，被拒后按 j*=k
    （截断则 take−1）切片写回，消掉第二次前向。
  - ``"snapshot"``（Step 5）：k==γ 且本轮产出全部被接收 → 跳过恢复与重放；
    否则恢复轮初快照，再 ``block_forward`` 重放提交前缀。

本模块引入 torch + vortex（经 ``specdec.block.driver``），CPU 套件请勿 import。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from specdec.block.driver import (
    block_forward,
    clone_inference_params,
    get_seqlen_offset,
    prefill,
    snapshot_states,
    step_forward_reference,
)
from specdec.block.slice import slice_states_to_accept
from specdec.transforms import apply_transform
from specdec.verifier import sample_categorical, verify_round, verify_round_greedy

VOCAB_SIZE = 512


@dataclass
class RoundLog:
    """一轮记账。``t_*`` 单位秒。"""

    k: int  # accepted_len（0..γ）
    replay_len: int  # 重放 token 数；全接受且未截断时为 0
    t_draft: float
    t_block: float
    t_verify: float
    t_restore_replay: float
    from_residual: bool = False
    fallback: bool = False
    n_emitted: int = 0
    t_snapshot: float = 0.0
    t_slice: float = 0.0
    conf: list[float] | None = None  # 神经 drafter 逐位置信度 c_k（长度 γ）；免费 drafter 为 None


@dataclass
class SpecGenerateResult:
    """``speculative_generate`` 返回值。可解包为 ``(emitted_ids, rounds_log)``。"""

    emitted_ids: np.ndarray
    rounds_log: list[RoundLog]
    inference_params_dict: dict = field(repr=False)
    anchor: int = -1
    emitted_logits: np.ndarray | None = None  # 可选 [n, V]，贪心对拍用
    rollback: str = "slice"

    def __iter__(self):
        yield self.emitted_ids
        yield self.rounds_log


def _as_prompt_ids(prompt_ids: torch.Tensor) -> torch.Tensor:
    if not isinstance(prompt_ids, torch.Tensor):
        prompt_ids = torch.as_tensor(prompt_ids, dtype=torch.long)
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids[None]
    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError(f"prompt_ids 须为 [1, L]，得到 {tuple(prompt_ids.shape)}")
    return prompt_ids.long()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _logits_rows(logits: torch.Tensor) -> np.ndarray:
    """``[1, T, V]`` → float64 ``[T, V]``。"""
    return logits[0].detach().float().cpu().numpy().astype(np.float64, copy=False)


def _q_prime_from_raw(
    q: np.ndarray,
    temperature: float,
    top_k: int | None,
) -> np.ndarray:
    """draft 原始分布 → 与 target 相同的采样变换。

    以 ``log(q)`` 作 logits 再走 ``apply_transform``（Step 2 适配器注释约定）。
    T=1 时 softmax 在支撑集上还原 q，效果即 top_k 截断 + 重归一化。
    """
    q = np.asarray(q, dtype=np.float64).ravel()
    logits = np.full(q.shape[0], -np.inf, dtype=np.float64)
    pos = q > 0.0
    if np.any(pos):
        logits[pos] = np.log(q[pos])
    else:
        logits[:] = 0.0
    return apply_transform(logits, temperature=temperature, top_k=top_k)


def _propose_draft(
    draft,
    prefix: np.ndarray,
    gamma: int,
    greedy: bool,
    rng: np.random.Generator,
    *,
    temperature: float = 1.0,
    top_k: int | None = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """自回归提出 γ 个草稿并记录 q_i。采样与 verify 共用 ``rng``。

    采样路径：从 q′ 采样，返回的 ``q_rows`` 即 q′（verify 必须用同一分布）。
    贪心路径：不施加变换，草稿 = argmax(原始 q)。
    """
    draft_tokens = np.empty(gamma, dtype=np.int64)
    q_rows = np.empty((gamma, VOCAB_SIZE), dtype=np.float64)
    buf = np.empty(int(prefix.size) + gamma, dtype=np.int64)
    buf[: prefix.size] = prefix
    length = int(prefix.size)
    for i in range(gamma):
        q_i = np.asarray(draft.probs(buf[:length]), dtype=np.float64).ravel()
        if q_i.shape[0] != VOCAB_SIZE:
            raise ValueError(f"draft.probs 须返回 [{VOCAB_SIZE}]，得到 {q_i.shape}")
        if not greedy:
            q_i = _q_prime_from_raw(q_i, temperature=temperature, top_k=top_k)
        q_rows[i] = q_i
        tok = int(np.argmax(q_i)) if greedy else sample_categorical(q_i, rng)
        draft_tokens[i] = tok
        buf[length] = tok
        length += 1
    return draft_tokens, q_rows


def _target_probs_from_logits(
    logits_rows: np.ndarray,
    greedy: bool,
    temperature: float,
    top_k: int | None,
) -> np.ndarray:
    """块输出 → 验证器 ``target_probs [γ+1, V]``。

    贪心模式直接把 logits 交给 ``verify_round_greedy``（只用 argmax，
    与 softmax 同序，并列取最小 index）。采样模式对每一行施加
    ``apply_transform``（与 vortex ``sample.py`` 在 T=1 时等价，见报告）。
    """
    if greedy:
        return logits_rows
    rows = [
        apply_transform(logits_rows[i], temperature=temperature, top_k=top_k)
        for i in range(logits_rows.shape[0])
    ]
    return np.stack(rows, axis=0)


def speculative_generate(
    model,
    draft,
    prompt_ids: torch.Tensor,
    n_tokens: int,
    gamma: int,
    greedy: bool,
    rng: np.random.Generator,
    *,
    temperature: float = 1.0,
    top_k: int | None = 4,
    inference_params_dict: dict | None = None,
    record_logits: bool = False,
    rollback: str = "slice",
) -> SpecGenerateResult:
    """端到端投机解码。

    ``prompt_ids``: ``[1, L]``。返回 ``SpecGenerateResult``（可解包为
    ``(emitted_ids, rounds_log)``）。末状态对应「prompt + emitted[:-1]」
    （最后发射的 token 是未消费锚点）；截断轮也会回滚到该不变量，
    以便状态对拍（比计划「截断后不维护」更严，成本可忽略）。

    ``rollback``: ``"slice"``（默认）切片回滚；``"snapshot"`` 快照+重放。

    神经 drafter（Step 10，``specdec.block.neural_draft.NeuralDraftModel``）：
    鸭子识别 ``propose_block`` / ``capture_begin`` / ``capture_commit``。
    每轮一次并行 drafter 前向 + 串行 Markov 偏置采样（替代 ``_propose_draft``
    的逐位自回归）；注入层 hidden 由 draft 侧的 hook 捕获，prefill 提交全部
    L−1 位、每轮提交 j*+1 位（与切片同下标，被拒后缀不进 H_ctx）。
    限定 ``rollback="slice"`` 且由本函数内部 prefill（外部 inference_params
    无法对齐 H_ctx 缓冲，拒绝）。
    """
    prompt_ids = _as_prompt_ids(prompt_ids)
    if n_tokens < 0:
        raise ValueError("n_tokens 不能为负")
    if gamma < 1:
        raise ValueError("gamma 须 ≥ 1")
    if rollback not in ("slice", "snapshot"):
        raise ValueError(f"rollback 须为 'slice' 或 'snapshot'，得到 {rollback!r}")
    neural = hasattr(draft, "propose_block")
    if neural:
        if rollback != "slice":
            raise ValueError("神经 drafter 需要 rollback='slice'（拒绝剔除与切片同下标 j*）")
        # 外部状态（长 prompt 分块预填）不再一律拒绝：下方 ip 分支做 H_ctx
        # 缓冲对齐自检（capture_buffer_len == L-1）。
    device = prompt_ids.device
    L = int(prompt_ids.shape[1])
    if L < 2:
        raise ValueError(f"prompt 长度须 ≥ 2（首轮引导需要锚点），得到 {L}")

    if n_tokens == 0:
        ip = inference_params_dict or model.initialize_inference_params(max_seqlen=L + gamma + 2)
        return SpecGenerateResult(
            emitted_ids=np.empty(0, dtype=np.int64),
            rounds_log=[],
            inference_params_dict=ip,
            anchor=int(prompt_ids[0, -1].item()),
            rollback=rollback,
        )

    prompt_np = prompt_ids[0].detach().cpu().numpy().astype(np.int64, copy=False)
    y = int(prompt_np[-1])
    seen = prompt_np.copy()

    if inference_params_dict is None:
        ip = model.initialize_inference_params(max_seqlen=L + n_tokens + gamma + 2)
        if neural:
            draft.capture_begin()
        with torch.inference_mode():
            prefill(model, prompt_ids[:, :-1], ip)
        if neural:
            draft.capture_commit(L - 1)
    else:
        ip = inference_params_dict
        if neural:
            # 长 prompt 分块预填路径（Step 15 长度曲线）：调用方须在分块预填期间
            # 保持 capture_begin()（HiddenCapture 跨块累加），完成后 capture_commit(L-1)。
            got = int(draft.capture_buffer_len())
            if got != L - 1:
                raise ValueError(
                    f"神经 drafter 外部状态的 H_ctx 缓冲未对齐：已消费 {got} ≠ L-1={L - 1}；"
                    f"须在分块预填前 capture_begin()、完成后 capture_commit(L-1)"
                )

    emitted: list[int] = []
    rounds_log: list[RoundLog] = []
    logit_rows: list[np.ndarray] = [] if record_logits else []

    while len(emitted) < n_tokens:
        remaining = n_tokens - len(emitted)

        snap = None
        t_snapshot = 0.0
        if rollback == "snapshot":
            _sync(device)
            t0 = time.perf_counter()
            snap = clone_inference_params(ip)
            _sync(device)
            t_snapshot = time.perf_counter() - t0

        if neural:
            _sync(device)
        t0 = time.perf_counter()
        if neural:
            draft_tokens, draft_probs, conf_rows = draft.propose_block(
                y,
                gamma,
                greedy,
                rng,
                temperature=temperature,
                top_k=top_k,
            )
        else:
            draft_tokens, draft_probs = _propose_draft(
                draft,
                seen,
                gamma,
                greedy,
                rng,
                temperature=temperature,
                top_k=top_k,
            )
            conf_rows = None
        if neural:
            _sync(device)
        t_draft = time.perf_counter() - t0

        chunk_np = np.empty(gamma + 1, dtype=np.int64)
        chunk_np[0] = y
        chunk_np[1:] = draft_tokens
        chunk = torch.tensor(chunk_np, dtype=torch.long, device=device)[None]

        stash = None
        if neural:
            draft.capture_begin()
        with torch.inference_mode():
            _sync(device)
            t0 = time.perf_counter()
            if rollback == "slice":
                logits, stash = block_forward(model, chunk, ip, retain=True)
            else:
                logits = block_forward(model, chunk, ip)
            _sync(device)
            t_block = time.perf_counter() - t0

        logits_np = _logits_rows(logits)
        target_probs = _target_probs_from_logits(logits_np, greedy, temperature, top_k)

        t0 = time.perf_counter()
        if greedy:
            result = verify_round_greedy(draft_tokens, target_probs)
        else:
            result = verify_round(draft_tokens, draft_probs, target_probs, rng)
        t_verify = time.perf_counter() - t0

        k = int(result.accepted_len)
        take = min(int(result.tokens.shape[0]), remaining)
        taken = result.tokens[:take]
        # 提交到 cache 的内容 = [y] + taken[:-1]；新锚点 = taken[-1]
        consume = np.empty(take, dtype=np.int64)
        consume[0] = y
        if take > 1:
            consume[1:] = taken[:-1]
        new_y = int(taken[-1])

        full_accept_kept = (k == gamma) and (take == k + 1)
        # j* = 最后被消费 token 的块内下标 = take - 1（未截断时 = k）
        j_star = int(consume.shape[0]) - 1
        replay_len = 0
        t_restore_replay = t_snapshot
        t_slice = 0.0
        if not full_accept_kept:
            if rollback == "snapshot":
                replay_len = int(consume.shape[0])
                with torch.inference_mode():
                    _sync(device)
                    t0 = time.perf_counter()
                    stale = ip
                    ip = snap
                    snap = None
                    replay = torch.tensor(consume, dtype=torch.long, device=device)[None]
                    block_forward(model, replay, ip)
                    _sync(device)
                    t_restore_replay = t_snapshot + (time.perf_counter() - t0)
                del stale
            else:
                with torch.inference_mode():
                    _sync(device)
                    t0 = time.perf_counter()
                    slice_states_to_accept(ip, stash, j_star)
                    _sync(device)
                    t_slice = time.perf_counter() - t0
        if neural:
            # H_ctx 只提交被消费位置（块内 0..j*），与切片同下标；被拒后缀剔除
            draft.capture_commit(j_star + 1)
        del snap
        del stash

        if record_logits:
            # tokens[i] 对应 target 的第 i 行（拒绝位/bonus 均落在 p_k）
            logit_rows.extend(logits_np[i] for i in range(take))

        emitted.extend(int(t) for t in taken.tolist())
        seen = np.concatenate([seen, taken])
        y = new_y
        rounds_log.append(
            RoundLog(
                k=k,
                replay_len=replay_len,
                t_draft=t_draft,
                t_block=t_block,
                t_verify=t_verify,
                t_restore_replay=t_restore_replay,
                from_residual=bool(result.from_residual),
                fallback=bool(result.fallback),
                n_emitted=take,
                t_snapshot=t_snapshot,
                t_slice=t_slice,
                conf=None if conf_rows is None else [float(c) for c in conf_rows],
            )
        )

    out_ids = np.asarray(emitted[:n_tokens], dtype=np.int64)
    out_logits = None
    if record_logits:
        out_logits = np.stack(logit_rows[:n_tokens], axis=0)
    return SpecGenerateResult(
        emitted_ids=out_ids,
        rounds_log=rounds_log,
        inference_params_dict=ip,
        anchor=y,
        emitted_logits=out_logits,
        rollback=rollback,
    )


def native_greedy_reference(
    model,
    prompt_ids: torch.Tensor,
    n_tokens: int,
    *,
    record_logits: bool = False,
) -> tuple[np.ndarray, dict, np.ndarray | None]:
    """逐步 argmax 参照（最小 index 并列约定，与 ``np.argmax`` / vortex greedy 一致）。

    返回 ``(emitted_ids, inference_params_dict, logits_or_none)``。
    末状态：prefill 全 prompt 后再消费 emitted[:-1]（最后 token 未写入 cache），
    与投机循环结束时的状态对齐。
    """
    prompt_ids = _as_prompt_ids(prompt_ids)
    if n_tokens < 0:
        raise ValueError("n_tokens 不能为负")
    device = prompt_ids.device
    L = int(prompt_ids.shape[1])
    ip = model.initialize_inference_params(max_seqlen=L + max(n_tokens, 0) + 2)
    if n_tokens == 0:
        return np.empty(0, dtype=np.int64), ip, None

    with torch.inference_mode():
        logits = prefill(model, prompt_ids, ip)
    last = logits[0, -1]

    emitted = np.empty(n_tokens, dtype=np.int64)
    logit_rows: list[np.ndarray] = [] if record_logits else []
    with torch.inference_mode():
        for i in range(n_tokens):
            last_np = last.detach().float().cpu().numpy().astype(np.float64, copy=False)
            if record_logits:
                logit_rows.append(last_np)
            tok = int(np.argmax(last_np))
            emitted[i] = tok
            if i + 1 < n_tokens:
                x = torch.tensor([[tok]], dtype=torch.long, device=device)
                step_logits = step_forward_reference(model, x, ip)
                last = step_logits[0, -1]
    out_logits = np.stack(logit_rows, axis=0) if record_logits else None
    return emitted, ip, out_logits


def native_sample_reference(
    model,
    prompt_ids: torch.Tensor,
    n_tokens: int,
    rng: np.random.Generator,
    *,
    temperature: float = 1.0,
    top_k: int | None = 4,
) -> np.ndarray:
    """逐步采样参照：每步 ``apply_transform`` 后 ``sample_categorical``（与投机共用变换）。"""
    prompt_ids = _as_prompt_ids(prompt_ids)
    if n_tokens < 0:
        raise ValueError("n_tokens 不能为负")
    if n_tokens == 0:
        return np.empty(0, dtype=np.int64)
    device = prompt_ids.device
    L = int(prompt_ids.shape[1])
    ip = model.initialize_inference_params(max_seqlen=L + n_tokens + 2)
    with torch.inference_mode():
        logits = prefill(model, prompt_ids, ip)
        last = logits[0, -1]
        emitted = np.empty(n_tokens, dtype=np.int64)
        for i in range(n_tokens):
            last_np = last.detach().float().cpu().numpy().astype(np.float64, copy=False)
            probs = apply_transform(last_np, temperature=temperature, top_k=top_k)
            tok = sample_categorical(probs, rng)
            emitted[i] = tok
            if i + 1 < n_tokens:
                x = torch.tensor([[tok]], dtype=torch.long, device=device)
                step_logits = step_forward_reference(model, x, ip)
                last = step_logits[0, -1]
    return emitted


def teacher_force_to_spec_state(
    model,
    prompt_ids: torch.Tensor,
    emitted_ids: np.ndarray,
    *,
    max_seqlen: int | None = None,
) -> dict:
    """把「同一序列」逐步 decode 到与投机循环结束相同的 cache 长度。

    prefill 全 prompt（L），再 ``step_forward`` ``emitted[:-1]``。
    结果 offset = L + n − 1，对应 prompt + emitted[:-1]。
    """
    prompt_ids = _as_prompt_ids(prompt_ids)
    emitted_ids = np.asarray(emitted_ids, dtype=np.int64).ravel()
    L = int(prompt_ids.shape[1])
    n = int(emitted_ids.size)
    if max_seqlen is None:
        max_seqlen = L + n + 2
    ip = model.initialize_inference_params(max_seqlen=max_seqlen)
    with torch.inference_mode():
        prefill(model, prompt_ids, ip)
        if n > 1:
            chunk = torch.tensor(emitted_ids[:-1], dtype=torch.long, device=prompt_ids.device)[None]
            step_forward_reference(model, chunk, ip)
    return ip


def spec_end_kv_len(prompt_len: int, n_emitted: int) -> int:
    """投机循环结束时已写入 KV 的 token 数 = L + n − 1。"""
    return int(prompt_len) + int(n_emitted) - 1


# 给测试/bench 复用
__all__ = [
    "RoundLog",
    "SpecGenerateResult",
    "get_seqlen_offset",
    "native_greedy_reference",
    "native_sample_reference",
    "snapshot_states",
    "slice_states_to_accept",
    "spec_end_kv_len",
    "speculative_generate",
    "teacher_force_to_spec_state",
]
