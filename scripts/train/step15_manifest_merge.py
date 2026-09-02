"""Step 15 manifest 重合并（plans/17 §1.2）：manifest.json ∪ manifest_imgvr.json。

照 2026-08-23 euk 补片的三方合并法：per_source 按 key 覆盖合并（补片源整条
替换/新增），``parts`` 记各子 manifest 名与其 n_positions，n_positions 重算。
幂等可重跑。

用法::

    /home/dh/miniconda3/envs/evo2/bin/python scripts/train/step15_manifest_merge.py \\
        --parts manifest_gpu0.json manifest_gpu2.json manifest_gpu2euk.json \\
               manifest_imgvr.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DUMP = REPO / "dump_c1"


def main() -> None:
    ap = argparse.ArgumentParser(description="dump_c1 manifest 分片合并")
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--out", type=str, default="manifest.json")
    args = ap.parse_args()

    base_path = DUMP / args.out
    man = json.loads(base_path.read_text()) if base_path.exists() else {}
    parts: dict[str, int] = dict(man.get("parts") or {})

    # per_source 一律从 parts 重 build（幂等）：同源多 part（双卡分流 A/B）按
    # n_pos/n_new/shard 数累加，标量字段取后写者
    acc: dict[str, dict] = {}
    config: dict = {}
    for name in args.parts:
        p = DUMP / name
        if not p.exists():
            raise SystemExit(f"缺子 manifest: {p}")
        sub = json.loads(p.read_text())
        for src, rec in (sub.get("per_source") or {}).items():
            if src in acc:
                for k in ("n_pos", "n_new", "n_shards"):
                    if rec.get(k) is not None:
                        acc[src][k] = int(acc[src].get(k) or 0) + int(rec[k])
                acc[src].update({k: v for k, v in rec.items() if k not in ("n_pos", "n_new", "n_shards")})
            else:
                acc[src] = dict(rec)
        # 配置字段（window/scheme/layers/shard_pos/with_euk…）取首个含它的 part
        for k in ("window", "scheme", "layers", "shard_pos", "with_euk", "with_imgvr"):
            if k in sub and k not in config:
                config[k] = sub[k]
        parts[name] = int(sub.get("n_positions") or 0)
    merged_sources = acc

    n_total = sum(int(v.get("n_pos") or 0) for v in merged_sources.values())
    man.update(
        {
            "status": "done",
            "n_positions": n_total,
            "per_source": merged_sources,
            "parts": parts,
            **config,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    tmp = base_path.with_suffix(base_path.suffix + ".tmp")
    tmp.write_text(json.dumps(man, indent=2, ensure_ascii=False))
    tmp.replace(base_path)
    print(
        f"[merge] {args.out}: n_positions={n_total:,} sources={sorted(merged_sources)} "
        f"parts={parts}"
    )


if __name__ == "__main__":
    main()
