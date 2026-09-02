"""per-anchor 神经/Markov 路由 drafter（Step 14 后处理；不训新组件）。

每轮两臂各算一个标量置信：

- 神经臂：conf 头逐位置预测值的均值（训练目标 = 逐位边缘接受率 c*，见
  ``drafter.distill_losses`` 的 BCE 目标）；
- Markov 臂：k=5 查表沿草稿前缀逐位行 maxprob 的均值。

``markov_score > neural_score + theta`` 时用 Markov 草稿，否则用神经草稿。
``theta`` 由 ``scripts/train/step14_route_calib.py`` 在留出窗上扫网格标定
（标定时用 p_t 真值算两臂解析 ᾱ，但运行时的路由信号只有 conf 与 maxprob——
不引入任何新训练组件）。

无损性论证：验证器与 draft 源无关——采样路径返回的 ``q_rows`` 就是实际采样
分布（q′ 变换复用 ``loop._q_prime_from_raw``，与免费/神经路径同函数同数值）；
贪心路径 verify 不读 q_rows。路由只改变 q 的选择，不破坏拒绝采样正确性。

Markov 上下文自追踪：神经协议的 ``propose_block`` 只传锚点 y（loop.py 不给
全量已发射流），wrapper 自行维护——每轮 propose 追加 y、``capture_commit``
按消费数追加被接受的草稿位（prefill 的 commit 无对应 propose，跳过簿记）。
开局历史不足 k=5 时由 ``MarkovDraftModel.probs`` 既有左 pad N 惯例兜底。

本模块引入 torch（经 ``specdec.block.neural_draft``），CPU 套件请勿 import。
"""

from __future__ import annotations

import numpy as np
import torch

from specdec.block.loop import _q_prime_from_raw
from specdec.block.neural_draft import NeuralDraftModel
from specdec.markov import MarkovDraftModel
from specdec.verifier import sample_categorical
from train.drafter import VOCAB_SIZE


class RoutedDraftModel:
    """神经/Markov per-anchor 路由（接口由 ``loop.speculative_generate`` 鸭子识别）。

    参数 ``theta``：路由阈值，Markov 胜出条件 ``markov_score > neural_score + theta``。
    运行侧统计：``n_rounds`` / ``n_markov`` / ``score_log``（逐轮两臂得分，供
    bench 汇总路由占比与事后分析）。
    """

    def __init__(self, neural: NeuralDraftModel, markov: MarkovDraftModel, theta: float = 0.0):
        self.neural = neural
        self.markov = markov
        self.theta = float(theta)
        self.gamma = int(neural.gamma)
        self._hist: list[int] = []
        self._drafts: np.ndarray | None = None
        self.n_rounds = 0
        self.n_markov = 0
        self.score_log: list[tuple[float, float]] = []
        self.ckpt_meta: dict = getattr(neural, "ckpt_meta", {})

    @property
    def layer_names(self) -> tuple[str, ...]:
        return self.neural.layer_names

    @property
    def drafter(self):
        return self.neural.drafter

    @classmethod
    def from_checkpoint(
        cls,
        model,
        ckpt_path: str,
        markov: MarkovDraftModel,
        theta: float = 0.0,
        *,
        device: str | torch.device = "cuda:0",
    ) -> "RoutedDraftModel":
        neural = NeuralDraftModel.from_checkpoint(model, ckpt_path, device=device)
        inst = cls(neural, markov, theta)
        inst.ckpt_meta = getattr(neural, "ckpt_meta", {})
        return inst

    # -- 循环协议 ----------------------------------------------------------

    def capture_begin(self) -> None:
        self.neural.capture_begin()

    def capture_commit(self, n_keep: int) -> None:
        self.neural.capture_commit(n_keep)
        if self._drafts is not None:
            keep = max(int(n_keep) - 1, 0)  # n_keep = j*+1，含锚点 y（propose 时已入账）
            if keep:
                self._hist.extend(int(t) for t in self._drafts[:keep])
            self._drafts = None

    def close(self) -> None:
        self.neural.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # -- Markov 臂（串行查表展开） ------------------------------------------

    def _markov_block(
        self,
        greedy: bool,
        rng: np.random.Generator,
        *,
        temperature: float,
        top_k: int | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        gamma = self.gamma
        tokens = np.empty(gamma, dtype=np.int64)
        q_rows = np.empty((gamma, VOCAB_SIZE), dtype=np.float64)
        tops = np.empty(gamma, dtype=np.float64)
        prefix = np.asarray(self._hist, dtype=np.int64)
        for k in range(gamma):
            row = np.asarray(self.markov.probs(prefix), dtype=np.float64)  # [V]，左 pad N 惯例
            tops[k] = float(row.max())
            if greedy:
                tok = int(np.argmax(row))  # 第一个最大元 = 最小 index tie-break（同 loop）
                q_rows[k] = row
            else:
                q = _q_prime_from_raw(row, temperature=temperature, top_k=top_k)
                tok = sample_categorical(q, rng)
                q_rows[k] = q
            tokens[k] = tok
            prefix = np.append(prefix, tok)
        return tokens, q_rows, tops

    # -- 主入口 -------------------------------------------------------------

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
        if int(gamma) != self.gamma:
            raise ValueError(f"drafter 训练 γ={self.gamma}，不支持循环 γ={gamma}")
        self._hist.append(int(anchor_id))  # 锚点 = 上一轮新发射的 token
        n_tok, n_q, confs = self.neural.propose_block(
            anchor_id, gamma, greedy, rng, temperature=temperature, top_k=top_k
        )
        m_tok, m_q, tops = self._markov_block(greedy, rng, temperature=temperature, top_k=top_k)
        neural_score = float(np.mean(confs))
        markov_score = float(np.mean(tops))
        use_markov = markov_score > neural_score + self.theta
        self.n_rounds += 1
        self.n_markov += int(use_markov)
        self.score_log.append((neural_score, markov_score))
        self._drafts = m_tok if use_markov else n_tok
        if use_markov:
            return m_tok, m_q, tops
        return n_tok, n_q, confs


__all__ = ["RoutedDraftModel"]
