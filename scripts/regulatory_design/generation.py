"""独立候选生成；共享只读前缀、私有新增 KV/FIR/IIR 与隐藏状态。

状态始终消费到序列倒数第二位，最后一位为尚未消费的 anchor。
不同候选分别拒绝采样，绝不进行 winner 选择或跨候选残差修正。
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import time
import numpy as np
import torch

from evspark.specdec.block.driver import (prefill, block_forward, set_seqlen_offsets,
                                  step_seqlen_offsets, _flash_kvcache_enabled)
from evspark.specdec.block.chunk import install_chunk_forward
from evspark.specdec.block.neural_draft import NeuralDraftModel, trunk_forward
from evspark.specdec.block.slice import slice_states_to_accept
from evspark.specdec.transforms import apply_transform
from evspark.specdec.verifier import sample_categorical, verify_round, verify_round_greedy

HYENA = ('hcl', 'hcm', 'hcs')
ATTRS = ('fir_state_dict', 'fir_inner_state_dict', 'state_dict')


class Meter:
    """同步阶段 wall time 与 CUDA event 时间；嵌套阶段不用于总和。"""
    def __init__(self):
        self.wall = {}
        self.gpu = {}

    @contextmanager
    def stage(self, name):
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        t0 = time.perf_counter()
        start.record()
        try:
            yield
        finally:
            end.record()
            end.synchronize()
            self.wall[name] = self.wall.get(name, 0.) + time.perf_counter() - t0
            self.gpu[name] = self.gpu.get(name, 0.) + start.elapsed_time(end) / 1000


@dataclass
class KVPrefix:
    """历史分支只保留 KV；已不再需要的旧 FIR/IIR/hidden 不沿祖先链累积。"""
    length: int
    kv: dict
    parent: 'KVPrefix | None' = None


@dataclass
class CandidateState:
    length: int
    anchor: int
    kv: dict
    hyena: dict
    context: torch.Tensor
    parent: KVPrefix | None = None

    def as_prefix(self):
        return KVPrefix(self.length,self.kv,self.parent)

    def chain(self):
        nodes, node = [], self
        while node is not None:
            nodes.append(node)
            node = node.parent
        return nodes[::-1]


@dataclass
class Candidate:
    token_ids: list[int]
    state: CandidateState
    seed: int
    rng_state: dict
    trace: list

    @property
    def sequence(self):
        # 非 DNA 输出必须可见，不静默过滤或改变 target 分布。
        return ''.join(chr(t) if 0 <= t < 256 else '\ufffd' for t in self.token_ids)


def row_rngs(seeds):
    if len(set(seeds)) != len(seeds):
        raise ValueError('独立候选种子不得重复')
    return [np.random.default_rng(int(seed)) for seed in seeds]


class CandidateGenerator:
    def __init__(self, model, checkpoint, *, batch=1, max_length=63000, gamma=12, native_only=False):
        self.model = model
        self.device = next(model.parameters()).device
        self.batch, self.gamma, self.max_length = int(batch), int(gamma), int(max_length)
        self.model.config.max_batch_size = self.batch
        install_chunk_forward()
        self.native_only=bool(native_only)
        self.neural = None if native_only else NeuralDraftModel.from_checkpoint(model, str(checkpoint), device=self.device)
        self.drafter = None if native_only else self.neural.drafter
        self.capture = None if native_only else self.neural.capture
        self.ip = None
        self.meter = Meter()

    def _captured(self):
        output = torch.cat([torch.cat(self.capture._slot[name], dim=1)
                            for name in self.neural.layer_names], dim=-1)
        self.capture.discard()
        return output

    def _extract(self, row, length, anchor, context, parent):
        start = 0 if parent is None else parent.length
        kv = {i: t[row:row+1, start:length].clone()
              for i, t in self.ip['mha'].key_value_memory_dict.items()}
        hyena = {k: {attr: {i: t[row:row+1].clone()
                            for i, t in getattr(self.ip[k], attr, {}).items()}
                     for attr in ATTRS} for k in HYENA}
        return CandidateState(length, int(anchor), kv, hyena, context.clone(),
                              None if parent is None else parent.as_prefix())

    @torch.inference_mode()
    def start(self, prompt):
        ids = np.frombuffer(prompt.encode('ascii'), dtype=np.uint8).astype(np.int64)
        if len(ids) < 128:
            raise ValueError('候选接口要求至少 128 bp 前缀')
        # prefill 只有一个共同前缀，不能提前分配 B 份完整 KV 挤占 FFT 工作区。
        self.ip = None
        self.model.config.max_batch_size = 1
        self.ip = self.model.initialize_inference_params(max_seqlen=self.max_length)
        self.model.config.max_batch_size = self.batch
        context = torch.empty((0,0),device=self.device) if self.native_only else None
        with self.meter.stage('prefill'):
            # 两臂相同的分块 prefill；长前缀限制峰值内存，不保存全长 hidden。
            for start in range(0, len(ids) - 1, 1024):
                piece = torch.as_tensor(ids[start:min(start+1024, len(ids)-1)], device=self.device)[None]
                if self.capture is not None:
                    self.capture.begin()
                if start == 0:
                    prefill(self.model, piece, self.ip)
                else:
                    block_forward(self.model, piece, self.ip)
                if self.capture is not None:
                    h = self._captured()[0, -self.drafter.ctx_window:].float()
                    context = h if context is None else torch.cat([context, h])[-self.drafter.ctx_window:]
        with self.meter.stage('cache'):
            return self._extract(0, len(ids)-1, ids[-1], context, None)

    def _restore(self, parent, batch):
        with self.meter.stage('cache'):
            self.ip['mha'].max_batch_size = self.batch
            # prefill 的末态可能是全位置轨迹的 view；先替换成紧凑状态再分配 KV。
            for k in HYENA:
                for attr in ATTRS:
                    setattr(self.ip[k], attr, {i: t.expand(batch, *t.shape[1:]).clone()
                                              for i, t in parent.hyena[k][attr].items()})
            cache=self.ip['mha'].key_value_memory_dict
            if any(t.shape[0]<self.batch for t in cache.values()):
                # parent 已持有紧凑的只读前缀；先释放旧 arena，再分配批量 arena。
                # 逐项覆盖且保留 list(cache.items()) 会同时占用新旧两套 KV，误判 B=8 OOM。
                layout={i:(tuple(t.shape[2:]),t.dtype,t.device) for i,t in cache.items()}
                cache.clear()
                for i,(shape,dtype,device) in layout.items():
                    cache[i]=torch.empty(self.batch,self.max_length,*shape,dtype=dtype,device=device)
            for node in parent.chain():
                start = 0 if node.parent is None else node.parent.length
                for i, segment in node.kv.items():
                    self.ip['mha'].key_value_memory_dict[i][:batch, start:node.length].copy_(segment)
            set_seqlen_offsets(self.ip, parent.length)
            # rotary 要求 shape 恰为实际 B，而非 arena 的最大 B。
            self.ip['mha'].lengths_per_sample = torch.full((batch,), parent.length, dtype=torch.int32, device=self.device)

    def _draft(self, anchors, contexts, rngs, greedy):
        b = len(anchors)
        ctxlen = torch.full((b,), contexts.shape[1], dtype=torch.long, device=self.device)
        a = torch.as_tensor(anchors, device=self.device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            U, _ = trunk_forward(self.drafter, a, contexts, ctxlen, gamma=self.gamma)
        # Markov 依赖仅在各行内部；一次搬运整批 logits，减少同步。
        U = U.float()
        prev = a
        drafts = np.empty((b, self.gamma), dtype=np.int64)
        q = np.empty((b, self.gamma, U.shape[-1]), dtype=np.float64)
        for j in range(self.gamma):
            raw = (U[:, j] + self.drafter.markov[prev]).float().cpu().numpy()
            for row in range(b):
                q[row, j] = apply_transform(raw[row], temperature=0 if greedy else 1., top_k=4)
                drafts[row, j] = np.argmax(raw[row]) if greedy else sample_categorical(q[row, j], rngs[row])
            prev = torch.as_tensor(drafts[:, j], device=self.device)
        return drafts, q

    @torch.inference_mode()
    def generate_candidates(self, parent, seeds, *, length=128, backend='native', greedy=False,
                            rng_states=None):
        if backend not in ('native', 'evspark') or length < 1:
            raise ValueError('backend 或 length 无效')
        if self.native_only and backend!='native':
            raise ValueError('纯 native 状态不含 drafter 上下文，不能用于投机生成')
        if backend=='evspark' and parent.context.numel()==0:
            raise ValueError('缺少 drafter 上下文，须重建前缀后投机生成')
        # 最多 length 轮；已完成行每轮只空转一个位置，不需要 γ*length 的余量。
        extra=(length+self.gamma) if backend=='evspark' else 0
        if parent.length + length + extra + 2 > self.max_length:
            raise ValueError('缓存容量不足（包含完成较早的 batch 行暂存空间）')
        if backend == 'evspark' and self.batch > 1 and not _flash_kvcache_enabled(self.model):
            raise ValueError('独立批量投机要求 flash_attn per-sample KV 路径')
        rngs = row_rngs(seeds)
        if rng_states is not None:
            for rng, state in zip(rngs, rng_states, strict=True):
                rng.bit_generator.state = state
        results = []
        for begin in range(0, len(seeds), self.batch):
            rr = rngs[begin:begin+self.batch]
            b = len(rr)
            self._restore(parent, b)
            anchors = np.full(b, parent.anchor, dtype=np.int64)
            contexts = parent.context[None].expand(b, -1, -1).clone()
            lengths = np.full(b, parent.length, dtype=np.int64)
            outputs, states = [[] for _ in range(b)], [None] * b
            traces = [[] for _ in range(b)]
            while any(s is None for s in states):
                with self.meter.stage('generation'):
                    if backend == 'native':
                        chunk = anchors[:, None].copy()
                        if self.capture is not None:
                            self.capture.begin()
                        logits, _ = self.model(torch.as_tensor(anchors, device=self.device)[:, None],
                                               inference_params_dict=self.ip)
                        if self.capture is not None:
                            h = self._captured()
                        raw = logits[:, -1].float().cpu().numpy()
                        rows = [[int(np.argmax(raw[i]) if greedy else sample_categorical(
                            apply_transform(raw[i], temperature=1., top_k=4), rr[i]))] for i in range(b)]
                        takes = np.ones(b, dtype=np.int64)
                        lengths += 1
                        step_seqlen_offsets(self.ip, 1)
                        self.ip['mha'].lengths_per_sample.add_(1)
                    else:
                        drafts, q = self._draft(anchors, contexts, rr, greedy)
                        chunk = np.concatenate([anchors[:, None], drafts], axis=1)
                        self.capture.begin()
                        logits, stash = block_forward(self.model, torch.as_tensor(chunk, device=self.device), self.ip, retain=True)
                        h = self._captured()
                        raw = logits.float().cpu().numpy()
                        rows, takes = [], []
                        for i in range(b):
                            p = np.array([apply_transform(v, temperature=0 if greedy else 1., top_k=4) for v in raw[i]])
                            vr = verify_round_greedy(drafts[i], p) if greedy else verify_round(drafts[i], q[i], p, rr[i])
                            take = min(len(vr.tokens), length - len(outputs[i])) if states[i] is None else 1
                            rows.append(vr.tokens[:take].tolist())
                            takes.append(take)
                        takes = np.asarray(takes)
                        slice_states_to_accept(self.ip, stash, takes-1)
                        lengths += takes
                        del stash
                    next_contexts = []
                    for i, take in enumerate(takes):
                        if not self.native_only:
                            next_contexts.append(torch.cat([contexts[i], h[i, :take].float()])[-self.drafter.ctx_window:])
                        anchors[i] = rows[i][-1]
                        if states[i] is None:
                            traces[i].append({'inputs':chunk[i].tolist(), 'consumed':int(take)})
                            outputs[i].extend(rows[i])
                    if not self.native_only:
                        contexts = torch.stack(next_contexts)
                # 完成行立即保存，后续空转不得修改已完成的私有状态。
                finished=[i for i in range(b) if states[i] is None and len(outputs[i])==length]
                if finished:
                    with self.meter.stage('cache'):
                        for i in finished:
                            states[i] = self._extract(i, int(lengths[i]), anchors[i], contexts[i], parent)
                            results.append((begin+i, Candidate(outputs[i], states[i], int(seeds[begin+i]), rr[i].bit_generator.state, traces[i])))
            del contexts
        return [item for _, item in sorted(results)]

    def close(self):
        if self.capture is not None:
            self.capture.close()
        self.ip = None
