#!/bin/bash

export HF_HUB_OFFLINE=1
export VLLM_USE_V1=0
export VLLM_USE_TRITON_FLASH_ATTN=0
export VLLM_ROCM_USE_AITER=0
export VLLM_ROCM_USE_AITER_PAGED_ATTN=1
export VLLM_ROCM_USE_AITER_LINEAR=0
export VLLM_ROCM_USE_AITER_MOE=0
export VLLM_ROCM_USE_AITER_RMSNORM=0
export VLLM_TORCH_PROFILER_DIR=EmbeddedLLM_Qwen2.5-1.5B-Instruct-FP8-Dynamic_pa_fp8kvcache3

vllm serve EmbeddedLLM/Qwen2.5-1.5B-Instruct-FP8-Dynamic \
    --tensor-parallel-size 1 \
    --kv-cache-dtype fp8 \
    --trust-remote-code \
    --disable-log-requests \
    --max-seq-len-to-capture 32768 \
    --num-scheduler-steps 10 \
    --port 8733 \
    > launch_server_$VLLM_TORCH_PROFILER_DIR.log 2>&1
