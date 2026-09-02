"""Step 13b / C.1 打包与判定的 CPU 单测。无 GPU、不读 OG2。"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="需要 torch（step12_retrain/drafter 依赖）")

from train.c1_pack import (
    arrays_crc,
    dequantize_hidden,
    entropy_bits,
    pack_topk,
    quantize_hidden,
    reconstruct_pt,
)
from train.drafter import SCHEME_LAYERS, Drafter
from train.distill import ALL_INJECT_LAYERS
from train.step12_data import choose_holdout_prefixes
from train.step12_retrain import STEP13B_CELLS, STEP14_CELLS, parse_holdout_pin
from train.step13b_merge import merge_and_judge


def test_scheme_layers_step13b():
    assert SCHEME_LAYERS["L20"] == ("blocks.20",)
    assert SCHEME_LAYERS["L23"] == ("blocks.23",)
    assert SCHEME_LAYERS["L30"] == ("blocks.30",)
    assert SCHEME_LAYERS["D16_27"] == ("blocks.16", "blocks.27")
    assert SCHEME_LAYERS["D27_30"] == ("blocks.27", "blocks.30")
    assert "blocks.20" in ALL_INJECT_LAYERS
    assert "blocks.30" in ALL_INJECT_LAYERS
    m = Drafter.from_scheme("D16_27", d_model=64, gamma=4, n_heads=2)
    assert m.n_inject == 2
    m30 = Drafter.from_scheme("L30", d_model=64, gamma=4, n_heads=2)
    assert m30.n_inject == 1


def test_step13b_cell_bank():
    tags = {c["tag"] for c in STEP13B_CELLS}
    assert len(STEP13B_CELLS) == 18
    assert "L20_d1024_g7_3M_s1" in tags
    assert "D27_30_d1024_g7_3M_s2" in tags
    assert "S3_d1024_g7_3M_s1" in tags
    assert {c["tag"] for c in STEP14_CELLS} >= {
        "L27_d1024_g7_100M_step14",
        "S3_d1024_g7_100M_step14",
        "D16_27_d1024_g7_100M_step14",
    }
    assert parse_holdout_pin("DPLL01,JARLHP") == ("DPLL01", "JARLHP")
    assert parse_holdout_pin(None) is None


def test_choose_holdout_deterministic():
    stats = {f"ABC{i:03d}": [10, 8_000_000] for i in range(10)}
    a = choose_holdout_prefixes(stats, n_hold=2, seed=20260821)
    b = choose_holdout_prefixes(stats, n_hold=2, seed=20260821)
    assert a == b and len(a) == 2


def test_c1_pack_roundtrip():
    rng = np.random.default_rng(0)
    h = rng.normal(0, 1, size=(16, 4096)).astype(np.float32)
    q, s = quantize_hidden(h)
    rec = dequantize_hidden(q, s)
    rel = np.max(np.abs(rec - h)) / (np.max(np.abs(h)) + 1e-8)
    assert rel < 0.02
    p = rng.random((16, 512)).astype(np.float32)
    p = p / p.sum(-1, keepdims=True)
    idx, val, resid = pack_topk(p, k=8)
    assert idx.dtype == np.uint16 and val.dtype == np.float16
    recon = reconstruct_pt(idx, val)
    mass = recon.sum(-1) + resid.astype(np.float32)
    assert np.allclose(mass, 1.0, atol=0.02)
    ent = entropy_bits(p)
    assert ent.shape == (16,) and np.all(ent > 0)
    crc = arrays_crc(q, idx)
    assert len(crc) == 8


def test_merge_judge_incumbent_l27(tmp_path):
    def cell(tag, scheme, seed, tau):
        return {
            "tag": tag,
            "scheme": scheme,
            "seed": seed,
            "layers": list(SCHEME_LAYERS[scheme]),
            "eval": {"tau_hat": tau},
            "freeze": {"embed_max_abs_delta": 0.0, "target_max_abs_delta": 0.0},
        }

    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    # 单层几乎并列，L27 incumbent；双层不超极差
    cells_a = [
        cell("L27_s1", "L27", 20260821, 3.30),
        cell("L6_s1", "L6", 20260821, 3.29),
        cell("D16_27_s1", "D16_27", 20260821, 3.31),
        cell("S3_s1", "S3", 20260821, 3.305),
    ]
    cells_b = [
        cell("L27_s2", "L27", 20260822, 3.302),
        cell("L6_s2", "L6", 20260822, 3.288),
        cell("D16_27_s2", "D16_27", 20260822, 3.308),
        cell("S3_s2", "S3", 20260822, 3.301),
    ]
    a.write_text(__import__("json").dumps({"env": {}, "cells": cells_a}))
    b.write_text(__import__("json").dumps({"env": {}, "cells": cells_b}))
    out = tmp_path / "merged.json"
    obj = merge_and_judge([a, b], out)
    assert obj["decision"]["scheme"] == "L27"
    assert obj["decision"]["n_layers"] == 1
    assert obj["decision"]["upgrade"] is False
    assert obj["decision"]["layers"] == ["blocks.27"]


def test_check_smoke_zero_freeze(tmp_path):
    """冻结 Δ=0.0 是合法值（`or 1` 陷阱回归：0.0 or 1 → 1 曾把正确冒烟判死）。"""
    import json

    from train.step13b_merge import check_smoke

    pin = tmp_path / "pin.json"
    pin.write_text(json.dumps({
        "step13_pool_stats_anchor": {"gtdb": {"n_records": 1, "bp": 100}},
        "gtdb_holdout_pin": ["DPLL01"],
    }))
    env = {
        "pool_stats": {"gtdb": {"n_records": 1, "bp": 100}},
        "gtdb_holdout": {"DPLL01": {}},
    }
    good = tmp_path / "good.json"
    good.write_text(json.dumps({
        "env": env,
        "cells": [{"freeze": {"embed_max_abs_delta": 0.0, "target_max_abs_delta": 0.0}}],
    }))
    check_smoke(good, pin)  # 不应抛

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({
        "env": env,
        "cells": [{"freeze": {"embed_max_abs_delta": 1e-9, "target_max_abs_delta": 0.0}}],
    }))
    with pytest.raises(SystemExit):
        check_smoke(bad, pin)
