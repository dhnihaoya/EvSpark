# EvSpark

**Lossless speculative decoding for Evo2 — StripedHyena2 hybrid DNA foundation models**

[**Paper (bioRxiv)**](https://www.biorxiv.org/content/10.64898/2026.09.02.749017)
· [DOI](https://doi.org/10.64898/2026.09.02.749017)
· [Hugging Face](https://huggingface.co/dinghhhhhhhhhhhhhhh/EvSpark)
· [ModelScope](https://www.modelscope.cn/models/dinghao1120/EvSpark)

![EvSpark](assets/fig1_hero.png)

EvSpark accelerates single-stream Evo2 generation by **2.2–3.3×** (7B→40B) with a
small distilled drafter, while staying **provably lossless**: every block is verified
against the target model's distribution (rejection sampling), and greedy output is
token-for-token identical to native decoding.

![demo](assets/evspark_demo.gif)

*Same prompt, same weights — left: native, right: EvSpark. Both sample from the same distribution (T=1.0, top_k=4) with different random seeds, so the sequences legitimately differ; the guarantee is distributional equivalence (and token-for-token identity in greedy mode).*

Evo2 is a **StripedHyena2 hybrid** (hyena IIR filters + sparse attention), so classic
KV-cache rollback does not apply. EvSpark contributes replay-free **state-slice
rollback** over Hyena IIR/FIR state and attention KV, a **γ-parallel distilled
drafter** conditioned on injected mid/late-layer hidden states, and **exact block
verification** with initial-state injection.

## Results at a glance

Evo2 7B (bf16), single stream, RTX 4090. Suite = 48 genomic prompts (43 real +
5 random) × 1024 tokens, three training seeds. The packaged prompt pack
(`evspark/data/eval_prompts_24.json`) is a self-contained 24-prompt subset.

| Metric | Value |
|---|---|
| Suite mean speedup, flagship drafter (γ=12, 150M distill tokens), all-48 | **3.27×** |
| — flagship, real-sequence subset (43 prompts) | 2.96× |
| Suite mean, cost-optimal drafter (γ=12, 30M) | 3.15× / 2.82× (all-48 / real-43) |
| Long context: *E. coli* genome, 262k | 1.97× / 2.28× / 2.43× (3 seeds) |
| Speedup vs context length | flat from 1k to 51k (GTDB 3-contig) |
| Larger targets, same drafter recipe (H20, SDPA): Evo2 20B / 40B | 2.18–2.46× real-43 · 2.51–2.78× all-48 |
| Regulatory-DNA design workflow vs batched native baseline | median **1.57×** complete-design speedup |
| Greedy losslessness, 48 prompts × 6 checkpoints | **0 non-tie divergences** |
| Drafter training cost, γ=12@30M (→ 3.15×) | **~1.06 GPU-hour** on one 4090 |
| Training-free reference (same suite): prompt-lookup k=2 / k=3 | 1.276× / 1.282× |

Full protocol, per-region table, ablations and cost model are in the
[paper](https://www.biorxiv.org/content/10.64898/2026.09.02.749017).

## Install & quickstart

Requirements: Linux, Python 3.11, one GPU with ~40 GB VRAM (RTX 4090/5090, A100,
H100). Ada GPUs (4090) run Evo2 in **bf16** only — the FP8/Transformer-Engine
path requires Hopper.

```bash
git clone https://github.com/dhnihaoya/EvSpark && cd EvSpark
conda create -n evo2 python=3.11 -y && conda activate evo2
pip install torch==2.7.1 flash-attn==2.8.0.post2 evo2   # evo2 pulls vortex (vtx)
python -m evo2.test.test_evo2_generation --model_name evo2_7b   # smoke test
```

Everything runs from the repo root (optionally `pip install -e .`); the first run
downloads the Evo2 7B weights (~13 GB) automatically — set
`HF_ENDPOINT=https://hf-mirror.com` if huggingface.co is unreachable.

```bash
python scripts/download_ckpt.py L27_g12_150M_s1              # drafter ckpt (~210 MB)
python scripts/demo.py --ckpt L27_g12_150M_s1 --greedy       # asserts exact equality
python scripts/demo.py --ckpt L27_g12_150M_s1 --n-tokens 1024  # wall-clock speedup
python scripts/demo.py --ckpt L27_g12_150M_s1 --prompt-file my_window.fa
```

Speedup is region-dependent: bacterial coding is the hardest region, random regions
reach 5.95×. Greedy mode on coding regions degenerates into repeats (a known
property of greedy DNA decoding) — use it for the exact-match check, not as a
speed showcase.

## Use it as a library

```python
from evspark import EvSpark

with EvSpark.load("L27_g12_150M_s1") as es:      # auto-downloads the drafter ckpt
    res = es.generate("ACGTACGT...", n_tokens=1024)   # sampling (T=1.0, top_k=4)
    print(res.text, f"{res.tok_s:.1f} tok/s, tau={res.mean_tau:.2f}")

    g = es.generate(prompt, greedy=True)
    nat = es.generate_native(prompt, greedy=True)
    assert g.ids.tolist() == nat.ids.tolist()         # token-for-token identical

    # decode-time γ′ without retraining: any γ′ ≤ training γ is an exact prefix
    res = es.generate(prompt, n_tokens=1024, gamma=8)
```

`EvSpark.load()` accepts a checkpoint name (table below), a local `.pt` path, or
another Evo2 model via `model_name=`.

## Checkpoints

Published on [Hugging Face](https://huggingface.co/dinghhhhhhhhhhhhhhh/EvSpark) and [ModelScope](https://www.modelscope.cn/models/dinghao1120/EvSpark) (same files, sha256 on the hub pages);
loadable via `NeuralDraftModel.from_checkpoint` (HF first, ModelScope fallback):

| Checkpoint | Target | γ | Distill budget | Suite speedup (all-48 / real-43) |
|---|---|---:|---:|---|
| `L27_g12_150M_s1` / `_s2` | Evo2 7B | 12 | 150M | **3.27× / 2.96×** (flagship) |
| `L27_g12_30M_s1` / `_s2` | Evo2 7B | 12 | 30M | 3.15× / 2.82× ("~1 GPU-hour" cell) |
| `phase3_20b_L20_g12_b1` / `_b2` | Evo2 20B | 12 | 10M / 30M | 2.53×/2.18× · 2.51×/2.22× |
| `phase3_40b_L45_g12_b1` / `_b2` / `_b3` | Evo2 40B | 12 | 10M / 30M / 80M | 2.72×/2.44× · 2.78×/2.46× · 2.72×/2.42× |

All drafters: single-layer injection (7B: blocks.27; 20B: blocks.20; 40B: blocks.45),
d_model=1024, distilled offline from frozen Evo2 hidden states; the target model is
never fine-tuned. 20B/40B drafters are published for reproducibility — running those
targets needs their official Transformer-Engine path (measured on one H20 96 GB;
no official path on Ada GPUs).

## Reproduce the paper numbers

See [docs/reproduce.md](docs/reproduce.md): hidden-state dump → drafter training
(single GPU or DDP) → evaluation suite, all off the packaged 24-prompt subset.

## Citation

```bibtex
@article{ding2026evspark,
  title   = {EvSpark: Lossless Speculative Decoding for Hybrid DNA Foundation Models},
  author  = {Ding, Hao and Wu, Nannan and Qiu, Tianyi},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.09.02.749017},
  url     = {https://www.biorxiv.org/content/10.64898/2026.09.02.749017}
}
```

## License

MIT (see [LICENSE](LICENSE)). EvSpark builds on
[Evo2](https://github.com/arcinstitute/evo2) and
[Vortex](https://github.com/Zymrael/vortex) — check their licenses for model
weights and runtime code.
