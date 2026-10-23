#!/usr/bin/env bash
# Serve the Qwen3.8 MLA retrofit on 2x gfx906 (16 GB each), text-only.
#
# Measured 2026-10-23 with this recipe: 10.6 GiB of weights per GPU, 21.5k tokens of KV at
# a 2 GiB cache, 14.5-20.9 tok/s decode (batch 1) and 411 tok/s prefill on a 342-token
# prompt. Prefix caching uses sub-block hits (QWEN_MLA_PREFIX_UNIT=16, the default): a
# 1,223-token prompt served cold and repeated reused 1,216 tokens with correct output.
#
# Knobs, in the order they are worth touching:
#   --kv-cache-memory-bytes   the real KV budget; gpu_memory_utilization alone over-asks on
#                             a 16 GB card (OOM in the KV allocation at 0.92)
#   --max-model-len           only bounds a single sequence; the KV pool is sized by the cap
#   QWEN_MLA_PREFIX_UNIT      "16" (default: sub-block prefix-cache hits) or "0" (block
#                             granularity; hits then round to the LCM of the group block
#                             sizes, so a 2k context never reuses anything)
#   QWEN_MLA_ROW_DEBUG=0      silence the per-layer row-width log
set -euo pipefail

MODEL=${MODEL:-/home/neko/Models/Qwen3.8-27B-GLA-g2-INT4}
PORT=${PORT:-8000}
KV_BYTES=${KV_BYTES:-2147483648}      # 2 GiB
MAX_LEN=${MAX_LEN:-2048}
MAX_SEQS=${MAX_SEQS:-2}
BATCHED=${BATCHED:-1024}
GPU_MEM=${GPU_MEM:-0.86}

cd "$(dirname "$0")/../.."
exec env HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0,1} \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/vllm serve "$MODEL" \
    --served-model-name qwen-mla \
    --port "$PORT" \
    --tensor-parallel-size 2 \
    --dtype float16 \
    --kv-cache-dtype float16 \
    --max-model-len "$MAX_LEN" \
    --max-num-seqs "$MAX_SEQS" \
    --max-num-batched-tokens "$BATCHED" \
    --gpu-memory-utilization "$GPU_MEM" \
    --kv-cache-memory-bytes "$KV_BYTES" \
    --reasoning-parser qwen3 \
    --limit-mm-per-prompt '{"image":0,"video":0}'
