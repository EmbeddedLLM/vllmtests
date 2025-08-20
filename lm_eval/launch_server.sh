#!/bin/bash

rm -rf /root/.cache/vllm

# MODEL=deepseek-ai/DeepSeek-R1

# VLLM_USE_V1=1 
# VLLM_ROCM_USE_AITER=1 \
# vllm serve $MODEL \
# --tensor-parallel-size 8 \
# --max-num-batched-tokens 32768 \
# --disable-log-requests \
# --compilation-config '{"full_cuda_graph":true}' \
# --trust-remote-code \
# --block-size 1 \
# --port 6789 \
# > server-deepseek-ai_DeepSeek-R1-aiter-v1-tritonfp8.log 2>&1


# MODEL=/app/model/QuantLLM/deepseek-r1-FP8-Dynamic-from-BF16-calib1024

# VLLM_USE_V1=1 
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_ASMMOE=1 \
# vllm serve $MODEL \
# --tensor-parallel-size 8 \
# --max-num-batched-tokens 32768 \
# --disable-log-requests \
# --compilation-config '{"full_cuda_graph":true}' \
# --trust-remote-code \
# --served_model_name "deepseek-r1-FP8-Dynamic-from-BF16-calib1024" \
# --block-size 1 \
# --port 6789 \
# > server-deepseek-r1-FP8-Dynamic-from-BF16-calib1024-aiter-v1-tritonfp8.log 2>&1



MODEL=Qwen/Qwen3-235B-A22B-FP8

VLLM_USE_V1=1 
VLLM_ROCM_USE_AITER=1 \
vllm serve $MODEL \
--tensor-parallel-size 8 \
--max-num-batched-tokens 32768 \
--disable-log-requests \
--kv-cache-dtype fp8 \
--compilation-config '{"cudagraph_mode": "FULL"}' \
--trust-remote-code \
--enable_expert_parallel \
--port 6789 \
> server-Qwen_Qwen3-235B-A22B-FP8-aiter-v1-fp8-cudagraph_FULL.log 2>&1