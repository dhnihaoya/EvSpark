"""EvSpark 门面（evspark 包）：CPU 测纯函数；GPU 端到端冒烟自动 skip。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# evspark 模块级只引入 numpy（torch/evo2 延迟到 EvSpark.load），CPU 可测
from evspark import EvSpark, clean_prompt, ids_to_text  # noqa: E402
from evspark.checkpoints import default_ckpt_dir  # noqa: E402

_CKPT = default_ckpt_dir() / "L27_g12_80M_s1.pt"


def test_clean_prompt_normalizes():
    assert clean_prompt(" acgt\n" * 20) == "ACGT" * 20


def test_clean_prompt_rejects_bad_chars():
    with pytest.raises(ValueError, match="non-ACGTN"):
        clean_prompt("ACGT" * 20 + "X")


def test_clean_prompt_rejects_short():
    with pytest.raises(ValueError, match="too short"):
        clean_prompt("ACGT")


def test_ids_to_text_roundtrip():
    ids = np.array([65, 67, 71, 84, 78], dtype=np.int64)
    assert ids_to_text(ids) == "ACGTN"
    assert ids_to_text(ids[None, :]) == "ACGTN"  # [1, N] 也可以


def test_ids_to_text_clips_oob():
    assert len(ids_to_text(np.array([0, 255, 999]))) == 3


@pytest.mark.skipif(
    not _CKPT.is_file(),
    reason="需要已下载的 drafter ckpt（python scripts/download_ckpt.py L27_g12_80M_s1）",
)
def test_gpu_generate_greedy_lossless():
    """GPU 冒烟：16 token 贪心，投机与原生逐位一致。"""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("需要 CUDA")
    if not os.environ.get("HF_HOME") and not (_ROOT / "hf_home").exists():
        pytest.skip("需要 HF_HOME（evo2 权重缓存）")

    prompt = "ACGT" * 32
    with EvSpark.load(str(_CKPT)) as es:
        spec = es.generate(prompt, n_tokens=16, greedy=True)
        nat = es.generate_native(prompt, n_tokens=16, greedy=True)
    assert spec.ids.tolist() == nat.ids.tolist()
    assert spec.mean_tau is not None and spec.n_rounds is not None
