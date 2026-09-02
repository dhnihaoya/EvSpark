#!/usr/bin/env bash
# Step 12 DDP 双卡启动器：手动起 2 进程、每 rank 绑一张 48G 卡（默认 GPU0/GPU2）。
#
# 为什么不用 torchrun 直起：evo2/vortex 会把模型平铺到**所有可见 GPU**
# （Evo2 docstring：'Vortex automatically handles device placement … splits model
# across multiple GPUs if available'）。若两 rank 共享 CUDA_VISIBLE_DEVICES=0,2，
# 每个 rank 各自把 7B 拆到两张卡上、互相踩踏（冒烟实测：HCL residues 落错
# 上下文，Triton 报 'cpu tensor?'）。因此每个 rank 只可见自己那张卡，用
# env:// rendezvous 手拉 NCCL 组；torchrun 的弹性重启本管线用不上。
#
# 用法：bash scripts/train/step12_launch_ddp.sh <step12_retrain.py 的透传参数...>
# 可用环境变量：GPUS（默认 "0 2"）、MASTER_PORT（默认 29517）、WORLD_SIZE（默认 2）
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

read -r -a GPUS <<< "${GPUS:-0 2}"
N=${WORLD_SIZE:-2}
if [ "${#GPUS[@]}" -lt "$N" ]; then
  echo "[ddp] GPU 列表(${GPUS[*]})少于 world_size=$N" >&2
  exit 2
fi
export WORLD_SIZE="$N" MASTER_ADDR=127.0.0.1 MASTER_PORT="${MASTER_PORT:-29517}" OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

PIDS=()
for ((r = 0; r < N; r++)); do
  CUDA_VISIBLE_DEVICES="${GPUS[$r]}" RANK="$r" LOCAL_RANK="$r" \
    python -u scripts/train/step12_retrain.py "$@" &
  PIDS+=($!)
  echo "[ddp] rank$r pid=${PIDS[$r]} → GPU${GPUS[$r]}"
done

FAIL=0
for ((r = 0; r < N; r++)); do
  wait "${PIDS[$r]}"; rc=$?
  [ "$rc" -ne 0 ] && { echo "[ddp] rank$r rc=$rc" >&2; FAIL=1; }
done
exit "$FAIL"
