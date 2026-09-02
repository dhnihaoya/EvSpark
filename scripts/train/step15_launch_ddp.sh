#!/usr/bin/env bash
# Step 15 离线训练 3 卡 DDP 启动器：每 rank 独占一卡（step12_launch_ddp.sh 同法）。
#
# evo2/vortex 会把模型铺到所有可见卡，禁止共享可见设备直起——每 rank 只可见
# 自己那张卡，env:// rendezvous 手拉 NCCL 组。drafter-only 训练 ~3GB，24G 的
# GPU1 也可参战（Step 14 已验证）。
#
# 用法: bash scripts/train/step15_launch_ddp.sh <step14_offline.py 的透传参数...>
# 可用环境变量：GPUS（默认 "0 1 2"）、MASTER_PORT（默认 29527）、WORLD_SIZE（默认 3）
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

read -r -a GPUS <<< "${GPUS:-0 1 2}"
N=${WORLD_SIZE:-3}
if [ "${#GPUS[@]}" -lt "$N" ]; then
  echo "[ddp15] GPU 列表(${GPUS[*]})少于 world_size=$N" >&2
  exit 2
fi
export WORLD_SIZE="$N" MASTER_ADDR=127.0.0.1 MASTER_PORT="${MASTER_PORT:-29527}" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" HF_HOME="${HF_HOME:-$PWD/hf_home}"

PY=/home/dh/miniconda3/envs/evo2/bin/python
PIDS=()
for ((r = 0; r < N; r++)); do
  # 错峰 30s/rank：避免三 rank 同时冷读同一 13GB 权重文件（2026-08-24 02:56
  # g12 格三路并发读时 rank2 被杀、rank0/1 永久 D 态的疑似诱因）
  sleep $((r * 30))
  CUDA_VISIBLE_DEVICES="${GPUS[$r]}" RANK="$r" LOCAL_RANK="$r" \
    "$PY" -u scripts/train/step14_offline.py "$@" &
  PIDS+=($!)
  echo "[ddp15] rank$r pid=${PIDS[$r]} → GPU${GPUS[$r]}"
done

FAIL=0
for ((r = 0; r < N; r++)); do
  wait "${PIDS[$r]}"; rc=$?
  [ "$rc" -ne 0 ] && { echo "[ddp15] rank$r rc=$rc" >&2; FAIL=1; }
done
exit "$FAIL"
