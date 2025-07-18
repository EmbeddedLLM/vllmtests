#!/bin/bash
rm -rf /root/.cache/vllm/
echo "deepseek-ai/DeepSeek-R1"
export VLLM_RPC_TIMEOUT=1800000
export VLLM_USE_V1=1
export VLLM_ROCM_USE_AITER=1
export SAFETENSORS_FAST_GPU=1
export MODEL_PATH=deepseek-ai/DeepSeek-R1
vllm serve $MODEL_PATH \
-tp 8 \
--block-size 1 \
--trust-remote-code \
--disable-log-requests \
--max-seq-len-to-capture 32768 \
--max-num-batched-tokens 32768 \
--no-enable-prefix-caching \
> deepseek-ai_DeepSeek-R1_v1_server-w8a8_utils.log 2>&1