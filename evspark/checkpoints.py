"""EvSpark drafter checkpoint 解析与下载（Hugging Face / ModelScope 双源）。

Checkpoints 各 ~160MB，含 drafter 全部权重（冻结 embedding、Markov 头、
γ-parallel confidence 头）与元数据（scheme / d_model / γ）；
``evspark.specdec.block.neural_draft.NeuralDraftModel.from_checkpoint`` 直接加载。

默认目录：``$EVSPARK_CKPT_DIR`` 或 ``~/.cache/evspark/checkpoints``。
"""

from __future__ import annotations

import os
from pathlib import Path

# 已发布 checkpoint（说明见 README §Checkpoints）
KNOWN_CKPTS = [
    "L27_g12_150M_s1",     # 7B 旗舰：γ=12，150M 蒸馏位置，all-48 3.27× / real-43 2.96×
    "L27_g12_150M_s2",     # 旗舰，第二种子
    "L27_g12_30M_s1",      # 1 GPU 时成本最优格，3.15×
    "L27_g12_30M_s2",
    "phase3_20b_L20_g12_b1",  # 20B drafter（blocks.20 注入），10M 位置，2.53×/2.18×
    "phase3_20b_L20_g12_b2",  # 20B，30M，2.51×/2.22×
    "phase3_40b_L45_g12_b1",  # 40B drafter（blocks.45 注入），10M，2.72×/2.44×
    "phase3_40b_L45_g12_b2",  # 40B，30M，2.78×/2.46×
    "phase3_40b_L45_g12_b3",  # 40B，80M，2.72×/2.42×
]

HF_REPO = os.environ.get("EVSPARK_HF_REPO", "dinghhhhhhhhhhhhhhh/EvSpark")
MS_REPO = os.environ.get("EVSPARK_MS_REPO", "dinghao1120/EvSpark")


def default_ckpt_dir() -> Path:
    env = os.environ.get("EVSPARK_CKPT_DIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "evspark" / "checkpoints"


def _from_hf(name: str, dest: Path) -> Path:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(HF_REPO, f"{name}.pt", local_dir=str(dest))
    return Path(path)


def _from_modelscope(name: str, dest: Path) -> Path:
    from modelscope import snapshot_download

    snapshot_download(MS_REPO, allow_patterns=[f"{name}.pt"], local_dir=str(dest))
    return dest / f"{name}.pt"


def ensure_ckpt(name_or_path: str, dest: Path | None = None,
                source: str = "auto") -> Path:
    """解析 checkpoint：本地路径直接透传；已知名字缺失时自动下载。"""
    p = Path(name_or_path)
    if p.is_file():
        return p
    name = p.name.removesuffix(".pt")
    dest = dest or default_ckpt_dir()
    local = dest / f"{name}.pt"
    if local.is_file():
        return local
    if name not in KNOWN_CKPTS:
        raise SystemExit(
            f"unknown checkpoint {name!r}; known: {', '.join(KNOWN_CKPTS)} "
            f"(or pass a local .pt path)"
        )
    dest.mkdir(parents=True, exist_ok=True)
    sources = {"hf": _from_hf, "modelscope": _from_modelscope}
    order = [source] if source != "auto" else ["hf", "modelscope"]
    errors = []
    for src in order:
        try:
            print(f"[ckpt] downloading {name}.pt from {src} ({HF_REPO if src == 'hf' else MS_REPO}) ...",
                  flush=True)
            got = sources[src](name, dest)
            print(f"[ckpt] -> {got}")
            return got
        except Exception as e:  # noqa: BLE001 - fall through to next mirror
            errors.append(f"{src}: {e}")
            print(f"[ckpt] {src} failed ({type(e).__name__}), trying next source ...", flush=True)
    raise SystemExit("all download sources failed:\n" + "\n".join(errors))
