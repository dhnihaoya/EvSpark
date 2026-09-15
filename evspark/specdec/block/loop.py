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

from evspark.specdec.block.driver import (
    block_forward,
    clone_inference_params,
    collapse_inference_params,
    expand_inference_params,
    get_seqlen_offset,
    prefill,
    snapshot_states,
    step_forward_reference,
)
from evspark.specdec.block.slice import slice_states_to_accept
from evspark.specdec.transforms import apply_transform
from evspark.specdec.verifier import (
    sample_categorical,
    verify_round,
    verify_round_greedy,
    verify_round_greedy_multicand,
    verify_round_multicand,
)

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
    n_candidates: int = 1
    winner: int = 0


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
        raise ValueError(
            f"prompt_ids 须为 [1, L]（独立多序列不走 generate 主路径；"
            f"多候选用 n_candidates 在块验证时扩 batch），得到 {tuple(prompt_ids.shape)}"
        )
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


def _target_probs_batch(
    logits_btv: np.ndarray,
    greedy: bool,
    temperature: float,
    top_k: int | None,
) -> np.ndarray:
    """``[B, T, V]`` logits → 验证器用的 target 分布 / 贪心 logits。"""
    out = np.empty_like(logits_btv, dtype=np.float64)
    for b in range(int(logits_btv.shape[0])):
        out[b] = _target_probs_from_logits(logits_btv[b], greedy, temperature, top_k)
    return out


def _propose_multi(
    draft,
    prefix: np.ndarray,
    y: int,
    gamma: int,
    greedy: bool,
    rng: np.random.Generator,
    n_candidates: int,
    has_propose: bool,
    *,
    temperature: float,
    top_k: int | None,
    inference_params_dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """返回 ``tokens[B, γ]``、``q[B, γ, V]``、可选 ``conf[B, γ]``。"""
    B = int(n_candidates)
    if has_propose and hasattr(draft, "propose_blocks"):
        return draft.propose_blocks(
            y,
            gamma,
            greedy,
            rng,
            B,
            temperature=temperature,
            top_k=top_k,
            inference_params_dict=inference_params_dict,
        )
    toks: list[np.ndarray] = []
    qs: list[np.ndarray] = []
    confs: list[np.ndarray | None] = []
    for _ in range(B):
        if has_propose:
            t, q, c = draft.propose_block(
                y,
                gamma,
                greedy,
                rng,
                temperature=temperature,
                top_k=top_k,
                inference_params_dict=inference_params_dict,
            )
        else:
            t, q = _propose_draft(
                draft,
                prefix,
                gamma,
                greedy,
                rng,
                temperature=temperature,
                top_k=top_k,
            )
            c = None
        toks.append(t)
        qs.append(q)
        confs.append(c)
    stacked_c = None if confs[0] is None else np.stack(confs, axis=0)
    return np.stack(toks, axis=0), np.stack(qs, axis=0), stacked_c


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
    n_candidates: int = 1,
) -> SpecGenerateResult:
    """端到端投机解码。

    ``prompt_ids``: ``[1, L]``。返回 ``SpecGenerateResult``（可解包为
    ``(emitted_ids, rounds_log)``）。末状态对应「prompt + emitted[:-1]」
    （最后发射的 token 是未消费锚点）；截断轮也会回滚到该不变量，
    以便状态对拍（比计划「截断后不维护」更严，成本可忽略）。

    ``rollback``: ``"slice"``（默认）切片回滚；``"snapshot"`` 快照+重放。
    ``n_candidates``: SpecInfer 式多候选（默认 1）。B 条路径放在 batch 维做
    一次块验证，按胜出路径切片后收回 B=1。``n_candidates>1`` 只支持
    ``rollback="slice"``。须在 ``initialize_inference_params`` 前把
    ``model.config.max_batch_size`` 设到 ≥ B。

    Drafter 鸭子协议：

    - 有 ``probs(prefix)``：CPU 逐步 ``_propose_draft``（恒等 / lookup / Markov）。
    - 有 ``propose_block``：每轮一次提出 γ 个草稿（神经 γ-parallel / First-N /
      独立 AR 小模型）。循环把 target 的 ``inference_params_dict`` 传入，不需要
      的实现可忽略。
    - 有 ``propose_blocks``：一次提出 B 条（神经：采样 i.i.d. / 贪心首位 top-B）。
    - 另有 ``capture_begin`` / ``capture_commit``：神经 HiddenCapture，限定
      ``rollback="slice"``；prefill 提交 L−1 位、每轮提交 j*+1 位。
    - 另有 ``prefill_prompt`` / ``commit_tokens``：独立 AR drafter 自管状态
      （1B）；prefill 吃完整 prompt，验证后步进 accepted 前缀。无 capture。
    """
    prompt_ids = _as_prompt_ids(prompt_ids)
    n_candidates = int(n_candidates)
    if n_candidates < 1:
        raise ValueError(f"n_candidates 须 ≥ 1，得到 {n_candidates}")
    if n_tokens < 0:
        raise ValueError("n_tokens 不能为负")
    if gamma < 1:
        raise ValueError("gamma 须 ≥ 1")
    if rollback not in ("slice", "snapshot"):
        raise ValueError(f"rollback 须为 'slice' 或 'snapshot'，得到 {rollback!r}")
    if n_candidates > 1 and rollback != "slice":
        raise ValueError("多候选只支持 rollback='slice'")
    has_propose = hasattr(draft, "propose_block")
    has_capture = hasattr(draft, "capture_begin")
    if has_capture:
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
        prev_b = int(model.config.get("max_batch_size", 1) or 1)
        if n_candidates > prev_b:
            model.config.max_batch_size = n_candidates
        ip = model.initialize_inference_params(max_seqlen=L + n_tokens + gamma + 2)
        if n_candidates > prev_b:
            model.config.max_batch_size = prev_b
        if int(ip["mha"].max_batch_size) < n_candidates:
            raise ValueError(
                f"推理 KV max_batch_size={ip['mha'].max_batch_size} < n_candidates={n_candidates}"
            )
        if has_capture:
            draft.capture_begin()
        if hasattr(draft, "prefill_prompt"):
            draft.prefill_prompt(prompt_ids, max_seqlen=L + n_tokens + gamma + 2)
        with torch.inference_mode():
            prefill(model, prompt_ids[:, :-1], ip)
        if has_capture:
            draft.capture_commit(L - 1)
    else:
        ip = inference_params_dict
        if n_candidates > 1 and int(ip["mha"].max_batch_size) < n_candidates:
            raise ValueError(
                f"外部 inference_params max_batch_size={ip['mha'].max_batch_size} "
                f"< n_candidates={n_candidates}；须在 initialize 前加大 config.max_batch_size"
            )
        if has_capture:
            # 长 prompt 分块预填路径（Step 15 长度曲线）：调用方须在分块预填期间
            # 保持 capture_begin()（HiddenCapture 跨块累加），完成后 capture_commit(L-1)。
            got = int(draft.capture_buffer_len())
            if got != L - 1:
                raise ValueError(
                    f"神经 drafter 外部状态的 H_ctx 缓冲未对齐：已消费 {got} ≠ L-1={L - 1}；"
                    f"须在分块预填前 capture_begin()、完成后 capture_commit(L-1)"
                )
        if hasattr(draft, "prefill_prompt") and getattr(draft, "ip", None) is None:
            raise ValueError(
                "独立 AR drafter 外部 target 状态须先自行 prefill_prompt（本循环不再预填）"
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

        if has_propose:
            _sync(device)
        t0 = time.perf_counter()
        if n_candidates == 1:
            if has_propose:
                with torch.inference_mode():
                    draft_tokens, draft_probs, conf_rows = draft.propose_block(
                        y,
                        gamma,
                        greedy,
                        rng,
                        temperature=temperature,
                        top_k=top_k,
                        inference_params_dict=ip,
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
        else:
            with torch.inference_mode():
                draft_tokens, draft_probs, conf_rows = _propose_multi(
                    draft,
                    seen,
                    y,
                    gamma,
                    greedy,
                    rng,
                    n_candidates,
                    has_propose,
                    temperature=temperature,
                    top_k=top_k,
                    inference_params_dict=ip,
                )
        if has_propose:
            _sync(device)
        t_draft = time.perf_counter() - t0

        if n_candidates == 1:
            chunk_np = np.empty(gamma + 1, dtype=np.int64)
            chunk_np[0] = y
            chunk_np[1:] = draft_tokens
            chunk = torch.tensor(chunk_np, dtype=torch.long, device=device)[None]
        else:
            chunk_np = np.empty((n_candidates, gamma + 1), dtype=np.int64)
            chunk_np[:, 0] = y
            chunk_np[:, 1:] = draft_tokens
            chunk = torch.tensor(chunk_np, dtype=torch.long, device=device)

        stash = None
        if has_capture:
            draft.capture_begin()
        with torch.inference_mode():
            if n_candidates > 1:
                expand_inference_params(ip, n_candidates)
            _sync(device)
            t0 = time.perf_counter()
            if rollback == "slice":
                logits, stash = block_forward(model, chunk, ip, retain=True)
            else:
                logits = block_forward(model, chunk, ip)
            _sync(device)
            t_block = time.perf_counter() - t0

        t0 = time.perf_counter()
        if n_candidates == 1:
            logits_np = _logits_rows(logits)
            target_probs = _target_probs_from_logits(logits_np, greedy, temperature, top_k)
            if greedy:
                result = verify_round_greedy(draft_tokens, target_probs)
            else:
                result = verify_round(draft_tokens, draft_probs, target_probs, rng)
            winner = 0
        else:
            logits_np = logits.detach().float().cpu().numpy().astype(np.float64, copy=False)
            target_probs = _target_probs_batch(logits_np, greedy, temperature, top_k)
            if greedy:
                result = verify_round_greedy_multicand(draft_tokens, target_probs)
            else:
                result = verify_round_multicand(draft_tokens, draft_probs, target_probs, rng)
            winner = int(result.winner)
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
        if n_candidates > 1:
            with torch.inference_mode():
                collapse_inference_params(ip, winner)
        if has_capture:
            # H_ctx 只提交被消费位置（块内 0..j*），与切片同下标；被拒后缀剔除
            draft.capture_commit(j_star + 1, batch_idx=winner)
        if hasattr(draft, "commit_tokens"):
            _sync(device)
            t_c0 = time.perf_counter()
            with torch.inference_mode():
                draft.commit_tokens(taken)
            _sync(device)
            t_draft += time.perf_counter() - t_c0
        del snap
        del stash

        if record_logits:
            # tokens[i] 对应 target 的第 i 行（拒绝位/bonus 均落在 p_k）
            src = logits_np if n_candidates == 1 else logits_np[winner]
            logit_rows.extend(src[i] for i in range(take))

        if conf_rows is None:
            conf_list = None
        elif n_candidates == 1:
            conf_list = [float(c) for c in conf_rows]
        else:
            conf_list = [float(c) for c in conf_rows[winner]]

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
                conf=conf_list,
                n_candidates=n_candidates,
                winner=winner,
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
