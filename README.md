# EvSpark

**Lossless speculative decoding for Evo2 — StripedHyena2 hybrid DNA foundation models**

![EvSpark](assets/fig1_hero.png)

EvSpark accelerates single-stream Evo2 7B generation by **2.9–3.2×** with a small
distilled drafter that proposes γ tokens per round, while staying **provably
lossless**: every block is verified against the target model's distribution
(rejection sampling), and greedy output is token-for-token identical to native
decoding.

![demo](assets/evspark_demo.gif)

*Same prompt, same weights — left: native Evo2 decoding, right: EvSpark speculative decoding.*

## Why this is non-trivial

Evo2 is a **StripedHyena2 hybrid** (hyena IIR filters + sparse attention), so classic
KV-cache rollback does not apply. EvSpark contributes:

- **State-slice rollback** over both Hyena IIR filter states and attention KV
  (`scripts/specdec/block/slice.py`) — no full replay, rollback cost <5% of a step.
- **A distilled γ-parallel drafter** conditioned on injected mid/late-layer hidden
  states, with a Markov-bias head and per-position confidence
  (`scripts/train/drafter.py`, `scripts/specdec/block/neural_draft.py`).
- **Exact verification**: block forward with initial-state injection
  (`scripts/specdec/block/driver.py`) + rejection-sampling verifier
  (`scripts/specdec/verifier.py`) — greedy-exact, sampling-distribution-exact.

## Results at a glance

Evo2 7B (bf16), single stream, RTX 4090. Suite = 24 genomic prompts × 1024 tokens,
dual training seeds.

| Metric | Value |
|---|---|
| Suite mean speedup, flagship drafter (γ=12, 80M distill tokens) | **3.16×** |
| Suite mean, main-table drafter (γ=7, 150M) | 2.89× / 2.87× (s1/s2) |
| Long context: *E. coli* genome, 262k | 2.18× / 2.18× |
| Long context: *B. subtilis*, 262k | 2.07× |
| Speedup vs context length | flat from 1k to 51k (GTDB 3-contig) |
| Greedy losslessness, 24 prompts × 4 checkpoints | **0 non-tie divergences** |
| Sampling equivalence | KL ≤ 1.6× of a native-vs-native floor on 10 prompts |
| Drafter training cost, γ=12@30M (→ 3.03×) | **~1.06 GPU-hour** on one 4090 |
| Baselines (same protocol): Markov-k5 / prompt-lookup k=2, k=3 | 2.65× / 1.24× / 1.30× |

Full protocol, per-region table, ablations (injection layer, drafter width, γ,
distill budget) and cost model are in the paper (link TBD).

## Install

Requirements: Linux, Python 3.11, one GPU with ~40 GB VRAM (RTX 4090/5090, A100,
H100). Note: Ada GPUs (4090) run Evo2 in **bf16** only — the FP8/Transformer-Engine
path requires Hopper.

```bash
conda create -n evo2 python=3.11 -y && conda activate evo2
pip install torch==2.7.1            # cu126/cu128 wheels both work on Ada
pip install flash-attn==2.8.0.post2 # prebuilt wheels on PyPI for torch 2.7
pip install evo2                    # pulls vortex (vtx)
python -m evo2.test.test_evo2_generation --model_name evo2_7b   # smoke test
```

The first run downloads the Evo2 7B weights (~13 GB) from Hugging Face
automatically; set `HF_ENDPOINT=https://hf-mirror.com` if huggingface.co is
unreachable.

## Quickstart

```bash
# 1. get a drafter checkpoint (~160 MB; HF first, ModelScope fallback)
python scripts/download_ckpt.py L27_g12_80M_s1

# 2. run the demo: speculative vs native on a 1 kb lacZ prompt,
#    greedy mode asserts token-for-token equality
python scripts/demo.py --ckpt L27_g12_80M_s1 --greedy
#    sampling mode measures wall-clock speedup:
python scripts/demo.py --ckpt L27_g12_80M_s1 --n-tokens 1024

# 3. your own sequence
python scripts/demo.py --ckpt L27_g12_80M_s1 --prompt-file my_genome_window.fa
```

Expected output (single RTX 4090, greedy, 128 tokens):

```
[demo] drafter=L27_g12_80M_s1.pt layers=['blocks.27'] gamma=12
[demo] prompt=1024 bp, generating 1024 tokens (sampling T=1.0 top_k=4)

  native      :   23.51 s  (  43.6 tok/s)
  speculative :    9.01 s  ( 113.6 tok/s)
  speedup     : 2.61x   (mean accepted tau=6.04, rounds=171)
```

Single-prompt demo on a human chr21 intergenic window. Speedup is
region-dependent: bacterial coding is the hardest cell (~1.1–1.3×), intergenic
and random regions reach 4–6×; the 24-prompt suite mean is 3.16×. Greedy mode
on coding regions degenerates into repeats (a known property of greedy DNA
decoding) — use it for the exact-match check, not as a speed showcase.

## Checkpoints

Published on [ModelScope](https://www.modelscope.cn/models/dinghao1120/EvSpark) (and Hugging Face, pending), loadable directly
via `NeuralDraftModel.from_checkpoint`:

| Checkpoint | γ | Distill budget | Suite speedup | Note |
|---|---:|---:|---:|---|
| `L27_g12_80M_s1` / `_s2` | 12 | 80M | **3.16×** (mean of seeds) | flagship |
| `L27_g12_30M_s1` / `_s2` | 12 | 30M | 3.03× | "~1 GPU-hour" cell |
| `L27_final15_150M_s1` / `_s2` | 7 | 150M | 2.89× / 2.87× | main-table model |

All drafters: single injection layer L27, d_model=1024, distilled offline from
frozen Evo2 hidden states; the target model is never fine-tuned.

## Repository layout

```
scripts/specdec/    speculative-decoding engine (verifier, transforms, block
                    forward, state slicing, spec loop, statistical drafters)
scripts/train/      drafter definition, offline distillation, evaluation suite
scripts/tests/      pytest suite (CPU + GPU; GPU tests auto-skip without CUDA)
scripts/demo.py     minimal end-to-end demo (losslessness check + speedup)
scripts/download_ckpt.py   checkpoint fetcher (Hugging Face / ModelScope)
docs/reproduce.md   full evaluation & training reproduction chain
```

## Reproduce the paper numbers

See [docs/reproduce.md](docs/reproduce.md): hidden-state dump → drafter training
(single GPU or DDP) → 24-prompt evaluation suite.

## Citation

```
TBD (preprint in preparation)
```

## License

MIT (see [LICENSE](LICENSE)). EvSpark builds on
[Evo2](https://github.com/arcinstitute/evo2) and
[Vortex](https://github.com/Zymrael/vortex) — check their licenses for model
weights and runtime code.
