"""块验证前向子包（Plan 02 A.2 / Step 4）。

``chunk``：Hyena 层带初始状态的块前向（monkeypatch dispatch，不改 site-packages）；
``driver``：prefill / block_forward / 逐 token 参照 / 状态快照比对；
``slice``：块前向保留中间量后按接受位切片写回（A.3）；
``loop``：端到端投机循环（slice / snapshot 双回滚）。

语义契约见 plans/06_step4_block_forward.md §3.1（p₁..p_{γ+1} off-by-one 红线）。
注意：import 本子包会引入 torch 与 vortex，CPU 测试套件请勿依赖。
"""

from evspark.specdec.block.chunk import install_chunk_forward
from evspark.specdec.block.driver import (
    PARAM_KEYS,
    block_forward,
    clone_inference_params,
    collapse_inference_params,
    diff_states,
    expand_inference_params,
    get_seqlen_offset,
    prefill,
    set_seqlen_offsets,
    snapshot_states,
    step_forward_reference,
    step_seqlen_offsets,
)
from evspark.specdec.block.slice import ChunkStash, slice_states_to_accept

__all__ = [
    "PARAM_KEYS",
    "ChunkStash",
    "block_forward",
    "clone_inference_params",
    "collapse_inference_params",
    "diff_states",
    "expand_inference_params",
    "get_seqlen_offset",
    "install_chunk_forward",
    "prefill",
    "set_seqlen_offsets",
    "slice_states_to_accept",
    "snapshot_states",
    "step_forward_reference",
    "step_seqlen_offsets",
]
