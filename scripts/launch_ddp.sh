#!/usr/bin/env bash
# 离线训练多卡 DDP 启动器：每 rank 独占一卡。
#
# evo2/vortex 会把模型铺到所有可见卡，禁止共享可见设备直起——每 rank 只可见
# 自己那张卡，env:// rendezvous 手拉 NCCL 组。drafter-only 训练显存 ~3GB，
# 24G 卡也可参战。
#
# 用法: bash scripts/launch_ddp.sh <train_drafter 的透传参数...>
# 可用环境变量：GPUS（默认 "0 1 2"）、WORLD_SIZE（默认 3）、MASTER_PORT（默认 29527）、
#              PYTHON（默认 python，指向装好 evo2 的环境）
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

read -r -a GPUS <<< "${GPUS:-0 1 2}"
N=${WORLD_SIZE:-3}
if [ "${#GPUS[@]}" -lt "$N" ]; then
  echo "[ddp] GPU 列表(${GPUS[*]})少于 world_size=$N" >&2
  exit 2
fi
export WORLD_SIZE="$N" MASTER_ADDR=127.0.0.1 MASTER_PORT="${MASTER_PORT:-29527}" \
  OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" HF_HOME="${HF_HOME:-$PWD/hf_home}"

PY="${PYTHON:-python}"
PIDS=()
for ((r = 0; r < N; r++)); do
  # 错峰 30s/rank：避免多 rank 同时冷读同一 13GB 权重文件
  sleep $((r * 30))
  CUDA_VISIBLE_DEVICES="${GPUS[$r]}" RANK="$r" LOCAL_RANK="$r" \
    "$PY" -u -m evspark.train.train_drafter "$@" &
  PIDS+=($!)
  echo "[ddp] rank$r pid=${PIDS[$r]} → GPU${GPUS[$r]}"
done

FAIL=0
for ((r = 0; r < N; r++)); do
  wait "${PIDS[$r]}"; rc=$?
  [ "$rc" -ne 0 ] && { echo "[ddp] rank$r rc=$rc" >&2; FAIL=1; }
done
exit "$FAIL"
