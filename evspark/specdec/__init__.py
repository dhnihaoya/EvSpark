"""EvSpark 投机解码：拒绝采样验证器与 CPU harness。

本包接口契约见 plans/03_step1_verifier_harness.md §4，签名勿擅自更改。
"""

from evspark.specdec.harness import NextProbModel, run_native, run_speculative
from evspark.specdec.transforms import apply_transform
from evspark.specdec.verifier import (
    VerifyResult,
    verify_round,
    verify_round_greedy,
    verify_round_greedy_multicand,
    verify_round_multicand,
)

__all__ = [
    "NextProbModel",
    "VerifyResult",
    "apply_transform",
    "run_native",
    "run_speculative",
    "verify_round",
    "verify_round_greedy",
    "verify_round_greedy_multicand",
    "verify_round_multicand",
]
