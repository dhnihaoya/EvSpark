# Reproduction guide

End-to-end chain: environment → Evo2 weights → (optional) hidden-state dump →
drafter training → evaluation. Everything below assumes the repo root as CWD and
the conda env from [README §Install](../README.md). No `pip install` of EvSpark
itself is needed — the `evspark` package imports fine from the repo root
(`pip install -e .` only if you want it importable from anywhere).

> You do **not** need the training chain to use EvSpark: published checkpoints
> (`scripts/download_ckpt.py`) are sufficient for inference and evaluation.

## 1. Environment

```bash
git clone https://github.com/dhnihaoya/EvSpark && cd EvSpark
conda create -n evo2 python=3.11 -y && conda activate evo2
pip install torch==2.7.1
pip install flash-attn==2.8.0.post2   # must match torch's cxx11 ABI; PyPI wheel works for torch 2.7.1
pip install evo2                      # pulls vortex (vtx >= 1.1.0)
python -m evo2.test.test_evo2_generation --model_name evo2_7b
```

Hardware used in the paper: 3× RTX 4090. A single ~48 GB GPU is enough for
inference, evaluation, and drafter training.

## 2. Quick functional test

```bash
python scripts/demo.py --ckpt L27_g12_150M_s1 --greedy   # losslessness + speedup
python -m pytest tests -q                                # CPU suite; GPU tests auto-skip
```

## 3. Evaluation suite (24 prompts × 1024 tokens)

This is the protocol behind every speedup number in the paper. The suite that
ships inside the package, `evspark/data/eval_prompts_24.json`, is a
self-contained 24-prompt subset of the paper's v2 48-prompt suite (43 real +
5 random; the full pack lives in the main repository as
`benchmarks/step18_suite_v2.json`). Regions: bacterial/viral coding, human
intergenic, human repeats, OpenGenome2 domain slices, IMG-VR coding, random
ACGT — the evaluation needs **no external corpus**. Checkpoints can be given
by name (see `python scripts/download_ckpt.py --list`) or as a local `.pt`
path.

```bash
python scripts/download_ckpt.py L27_g12_150M_s1 L27_g12_150M_s2
CUDA_VISIBLE_DEVICES=0 python -u -m evspark.train.eval_suite \
  --ckpt L27_g12_150M_s1 --tag repro_s1 --n-tokens 1024 --modes smoke,grid
# results land in benchmarks/eval_suite_repro_s1.json
```

Use `--prompts my_prompts.json` to evaluate on a custom prompt pack (same
schema: `{"prompts": [{name, region, source, seq}, ...]}`).

## 4. Train your own drafter

### 4.1 Dump frozen hidden states (only needed once per injection scheme)

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/dump_c1_dataset.py \
  --scheme L27 --sources gtdb,mrna,hg38,ncbi,ncrna,organelle,promoters,euk,imgvr \
  --total-positions 30000000
```

The dump is large (~4 GB per 1M positions, int8-quantized top-8 distributions);
30M positions is enough for the "1 GPU-hour" drafter. Sources are public corpora
(GTDB / OpenGenome2 / IMG-VR / NCBI / hg38); see the script header for download
pointers. `python -m evspark.train.fetch_ncbi_coding` regenerates the NCBI
coding slice (`data/ncbi_coding/`).

### 4.2 Offline distillation

```bash
# single GPU
CUDA_VISIBLE_DEVICES=0 python -u -m evspark.train.train_drafter \
  --scheme L27 --mix final15 --window 4096 --gamma 12 --max-positions 30000000 \
  --seed 20260821 --ckpt-name my_drafter.pt --tag my_drafter

# multi-GPU: one rank per visible GPU (evo2/vortex pins the model to every
# visible device, so do NOT use a shared torchrun launch)
GPUS="0 1 2" bash scripts/launch_ddp.sh \
  --scheme L27 --mix final15 --window 4096 --gamma 12 --max-positions 30000000 \
  --seed 20260821 --results benchmarks/my_cell.json \
  --ckpt-name my_drafter.pt --tag my_drafter
```

Training mix (`final15`) is defined once in `evspark/train/mix.py`.
γ=12 @ 30M positions costs ~1 GPU-hour on one RTX 4090.

### 4.3 Evaluate the new drafter

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m evspark.train.eval_suite \
  --ckpt benchmarks/phasec_ckpts/my_drafter.pt --tag my_drafter \
  --n-tokens 1024 --modes smoke,grid
```

## 5. Greedy losslessness check (24-prompt suite)

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/bench_lossless.py \
  --greedy-ckpt L27_g12_150M_s1
```

Expected: 0 non-tie divergences vs native greedy; occasional bf16 tie-flips are
kernel reduction-order artifacts and are logged with their logit gaps.
`--kl-ckpt <ckpt>` additionally measures sampling equivalence (KL/TVD vs a
native-vs-native bootstrap floor).

## Notes

- The speculative engine never fine-tunes the target model; all drafter
  checkpoints are self-contained (frozen embedding + Markov head + metadata).
- `evspark/specdec/` is pure-python + numpy/torch and is unit-tested on CPU
  (`tests/`); the block/loop paths additionally need CUDA.
- `scripts/evalpack.py` centralizes prompt-pack loading, per-prompt seeding,
  acceptance metrics, and the greedy comparison helpers shared by
  `scripts/bench_neural_c5.py` (measurement engine) and `scripts/bench_lossless.py`.
- If huggingface.co is unreachable: `export HF_ENDPOINT=https://hf-mirror.com`
  before the first Evo2 load, and use `--source modelscope` for checkpoints.
