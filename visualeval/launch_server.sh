#!/bin/bash
rm -rf /root/.cache/vllm

export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
echo "meta-llama/Llama-4-Scout-17B-16E-Instruct aiter v1"

# VLLM_RPC_TIMEOUT=1800000 \
# VLLM_USE_V1=1 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_MHA=1 \
# VLLM_ROCM_USE_AITER_RMSNORM=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# SAFETENSORS_FAST_GPU=1 \
# vllm serve meta-llama/Llama-4-Scout-17B-16E-Instruct \
#  -tp 4 \
#  --max_model_len 10000 \
#  --trust_remote_code \
# > server_gsm8k-meta-llama_Llama-4-Scout-17B-16E-Instruct-aiter-v1-upstream-mha.log 2>&1


VLLM_RPC_TIMEOUT=1800000 \
VLLM_USE_V1=1 \
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_USE_AITER_MHA=1 \
VLLM_ROCM_USE_AITER_RMSNORM=0 \
VLLM_USE_TRITON_FLASH_ATTN=0 \
SAFETENSORS_FAST_GPU=1 \
vllm serve meta-llama/Llama-4-Scout-17B-16E-Instruct \
 -tp 8 \
 --max_model_len 32768 \
 --trust_remote_code \
> server_meta-llama_Llama-4-Scout-17B-16E-Instruct-aiter-v1-upstream-aitermha-llama4vision-2.log 2>&1