# Reproduction guide

End-to-end chain: environment → Evo2 weights → (optional) hidden-state dump →
drafter training → evaluation. Everything below assumes the repo root as CWD and
the conda env from [README §Install](../README.md).

> You do **not** need the training chain to use EvSpark: published checkpoints
> (`scripts/download_ckpt.py`) are sufficient for inference and evaluation.

## 1. Environment

```bash
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
python scripts/demo.py --ckpt L27_g12_80M_s1 --greedy   # losslessness + speedup
python -m pytest scripts/tests -q                        # CPU suite; GPU tests auto-skip
```

## 3. Evaluation suite (24 prompts × 1024 tokens)

This is the protocol behind every speedup number in the paper.

```bash
python scripts/download_ckpt.py L27_g12_80M_s1 L27_g12_80M_s2
CUDA_VISIBLE_DEVICES=0 python -u scripts/train/step15_c5_suite.py \
  --ckpt checkpoints/L27_g12_80M_s1.pt --tag repro_s1_t1024 \
  --n-tokens 1024 --modes smoke,grid \
  --extra-prompts-json benchmarks/step15_main_suite_extra.json
# results land in benchmarks/step15_c5_repro_s1_t1024.json
```

Prompts cover six regions: bacterial coding, human intergenic, human repeats,
viral coding, OpenGenome2 eukaryotic slices, and random ACGT
(`benchmarks/step15_main_suite_extra.json` contains the exact sequences).

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
pointers.

### 4.2 Offline distillation

```bash
# single GPU
CUDA_VISIBLE_DEVICES=0 python -u scripts/train/step14_offline.py \
  --scheme L27 --mix final15 --window 4096 --gamma 12 --max-positions 30000000 \
  --seed 20260821 --ckpt-name my_drafter.pt --tag my_drafter

# multi-GPU: one rank per visible GPU (evo2/vortex pins the model to every
# visible device, so do NOT use a shared torchrun launch)
GPUS="0 1 2" bash scripts/train/step15_launch_ddp.sh \
  --scheme L27 --mix final15 --window 4096 --gamma 12 --max-positions 30000000 \
  --seed 20260821 --results benchmarks/my_cell.json \
  --ckpt-name my_drafter.pt --tag my_drafter
```

Training mix (`final15`) is defined once in `scripts/train/step15_mix.py`.
γ=12 @ 30M positions costs ~1 GPU-hour on one RTX 4090.

### 4.3 Evaluate the new drafter

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train/step15_c5_suite.py \
  --ckpt checkpoints/my_drafter.pt --tag my_drafter_t1024 \
  --n-tokens 1024 --modes smoke,grid \
  --extra-prompts-json benchmarks/step15_main_suite_extra.json
```

## 5. Greedy losslessness check (24-prompt suite)

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/bench_step15_lossless.py \
  --ckpt checkpoints/L27_g12_80M_s1.pt
```

Expected: 0 non-tie divergences vs native greedy; occasional bf16 tie-flips are
kernel reduction-order artifacts and are logged with their logit gaps.

## Notes

- The speculative engine never fine-tunes the target model; all drafter
  checkpoints are self-contained (frozen embedding + Markov head + metadata).
- `scripts/specdec/` is pure-python + numpy/torch and is unit-tested on CPU
  (`scripts/tests`); the block/loop paths additionally need CUDA.
- If huggingface.co is unreachable: `export HF_ENDPOINT=https://hf-mirror.com`
  before the first Evo2 load, and use `--source modelscope` for checkpoints.
