"""CPU 测试：终版配比（final15）、加权窗分配器、epoch 计划、dump imgvr 接线。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from evspark.train.mix import (  # noqa: E402
    FINAL_MIX,
    allocate_epoch_windows,
    build_epoch_plan,
    distribute_int,
    load_mix,
)


def test_final_mix_sums_to_one():
    assert abs(sum(FINAL_MIX.values()) - 1.0) < 1e-9
    # plans/17 §1.5 钉住的四个值
    assert FINAL_MIX["gtdb"] == pytest.approx(0.20)
    assert FINAL_MIX["mrna"] == pytest.approx(0.315)
    assert FINAL_MIX["imgvr"] == pytest.approx(0.10)
    assert FINAL_MIX["euk"] == pytest.approx(0.10)


def test_load_mix_variants(tmp_path):
    assert load_mix("uniform") is None
    assert load_mix("") is None
    assert load_mix("final15") == dict(FINAL_MIX)
    got = load_mix('{"gtdb": 0.5, "mrna": 0.5}')
    assert got == {"gtdb": 0.5, "mrna": 0.5}
    p = tmp_path / "mix.json"
    p.write_text('{"a": 1, "b": 3}')
    assert load_mix(str(p)) == {"a": 1.0, "b": 3.0}
    with pytest.raises(ValueError):
        load_mix('{"a": -1, "b": 3}')
    with pytest.raises(ValueError):
        load_mix('{"a": 0, "b": 0}')


def test_allocate_uniform_none_returns_all():
    avail = {"gtdb": 12, "mrna": 7, "imgvr": 5}
    assert allocate_epoch_windows(avail, None) == avail


def test_allocate_matches_weights_when_uncapped():
    avail = {"a": 10_000, "b": 10_000, "c": 10_000}
    w = {"a": 0.5, "b": 0.3, "c": 0.2}
    meta = allocate_epoch_windows(avail, w, cap_epochs=2.0)
    capped = meta.pop("__capped__")
    raw = meta.pop("__raw__")
    assert capped == []
    total = sum(avail.values())
    assert sum(meta.values()) == total
    assert meta["a"] == pytest.approx(0.5 * total, abs=2)
    assert meta["b"] == pytest.approx(0.3 * total, abs=2)
    assert raw["a"] == pytest.approx(0.5 * total, rel=1e-6)


def test_allocate_caps_tiny_pool_and_redistributes():
    # a 池极小但权重高：最多 2 遍；剩余预算让给 b（严格按 b 权重放大 E）
    avail = {"a": 10, "b": 990}
    w = {"a": 0.9, "b": 0.1}
    meta = allocate_epoch_windows(avail, w, cap_epochs=2.0)
    capped = meta.pop("__capped__")
    meta.pop("__raw__")
    assert "a" in capped
    assert meta["a"] == 20  # 2 遍封顶
    assert sum(meta.values()) == sum(avail.values())
    assert meta["b"] == 980


def test_allocate_rejects_infeasible_cap():
    with pytest.raises(ValueError):
        allocate_epoch_windows({"a": 5, "b": 5}, {"a": 0.5, "b": 0.5}, cap_epochs=0.5)


def test_allocate_allzero_weights_raise():
    with pytest.raises(ValueError):
        allocate_epoch_windows({"a": 5, "b": 5}, {"z": 1.0})


def test_distribute_int_exact():
    got = distribute_int(10, {"x": 3, "y": 7})
    assert sum(got.values()) == 10
    assert got == {"x": 3, "y": 7}
    got = distribute_int(7, {"x": 1, "y": 1, "z": 1})
    assert sum(got.values()) == 7
    assert max(got.values()) - min(got.values()) <= 1


def _fake_metas(spec):
    """spec: {source: [(n_pos, ...)]} → metas 列表（全局下标稳定）。"""
    metas = []
    for src, sizes in spec.items():
        for n in sizes:
            metas.append({"source": src, "n_pos": int(n)})
    return metas


def test_build_epoch_plan_uniform_covers_every_window():
    metas = _fake_metas({"gtdb": [4096 * 3, 4096 * 2], "mrna": [4096]})
    rng = np.random.default_rng(0)
    plan, info = build_epoch_plan(metas, None, 4096, rng)
    assert info["n_windows_epoch"] == 6
    per_shard: dict[int, set[int]] = {}
    for gi, wins in plan:
        assert all(0 <= w < metas[gi]["n_pos"] // 4096 for w in wins)
        s = per_shard.setdefault(gi, set())
        assert not (s & set(wins)), "uniform 下同窗重复"
        s.update(wins)
    assert all(len(s) == metas[gi]["n_pos"] // 4096 for gi, s in per_shard.items())


def test_build_epoch_plan_weighted_respects_alloc_and_dedup():
    metas = _fake_metas({"big": [4096 * 100], "small": [4096 * 4]})
    w = {"big": 0.5, "small": 0.5}
    rng = np.random.default_rng(1)
    plan, info = build_epoch_plan(metas, w, 4096, rng, cap_epochs=2.0)
    assert "small" in info["capped"]
    visited = {"big": 0, "small": 0}
    from collections import Counter

    small_counts: Counter[int] = Counter()
    for gi, wins in plan:
        src = metas[gi]["source"]
        visited[src] += len(wins)
        if src == "small":
            small_counts.update(wins)
    # small 超额 = 整遍×2：每窗恰好 2 次（无放回余量均衡）
    assert all(c == 2 for c in small_counts.values()) and len(small_counts) == 4
    assert visited["small"] == 8
    assert sum(visited.values()) == info["n_windows_epoch"] == 104


def test_build_epoch_plan_zero_weight_source_skipped():
    metas = _fake_metas({"a": [4096 * 10], "b": [4096 * 10]})
    rng = np.random.default_rng(2)
    plan, info = build_epoch_plan(metas, {"a": 1.0}, 4096, rng)
    sources = {metas[gi]["source"] for gi, _ in plan}
    assert sources == {"a"}
    assert info["alloc"]["b"] == 0


def test_dump_imgvr_wiring():
    """dump 侧 SOURCE_SUB/配额公式（不 import torch：源码文本 + 纯函数两种口径）。"""
    src = (_SCRIPTS / "scripts" / "dump_c1_dataset.py").read_text()
    assert '"imgvr": "imgvr_untagged"' in src
    # 配额公式为模块级纯 python，但模块 import evspark.train.* 链重；文本断言足够防回归
    assert "IMGVR_WEIGHT = 0.10" in src


def test_train_drafter_mix_flag_wired():
    src = (_SCRIPTS / "evspark" / "train" / "train_drafter.py").read_text()
    assert "--mix" in src and "final15" in src
    assert "build_epoch_plan" in src


def test_dump_split_options_wired():
    src = (_SCRIPTS / "scripts" / "dump_c1_dataset.py").read_text()
    assert "--source-files" in src and "--shard-tag" in src
    assert "name_prefix" in src  # 分片名带 tag、meta.source 不带
