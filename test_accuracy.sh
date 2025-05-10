#!/bin/bash

export HF_TOKEN=""

rm -rf /root/.cache/vllm
# export HF_HUB_OFFLINE=0

echo "test v1"

# echo "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic aiter v1"
# VLLM_USE_V1=1 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_RMSNORM=0 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic -tp 8 --gpu_memory_utilization 0.95 \
# --max-model-len 430000 \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FASTCHECK_RedHatAI_Llama-4-Scout-17B-16E-Instruct-FP8-dynamic-aiter-v1.log 2>&1


# echo "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 aiter v1"
# VLLM_USE_V1=0 \
# VLLM_USE_TRITON_FLASH_ATTN=1 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_RMSNORM=0 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 -tp 8 --gpu_memory_utilization 0.95 \
# --max-model-len 430000 \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FASTCHECK_meta-llama_Llama-4-Maverick-17B-128E-Instruct-FP8-aiter-v0.log 2>&1


# PASSED
# echo "Qwen/Qwen3-30B-A3B-FP8 aiter block gemm"
# VLLM_USE_V1=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=0 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model Qwen/Qwen3-30B-A3B-FP8 -tp 8 --no-enable-chunked-prefill \
# --max-model-len 32768 --enable-expert-parallel \
# --trust-remote-code --swap-space 16 --distributed-executor-backend mp \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FINAL_Qwen_Qwen3-30B-A3B-FP8-aiter-v0-1.log 2>&1

# echo "Qwen/Qwen3-30B-A3B-FP8 aiter debug"
# VLLM_USE_V1=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_MOE=0 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model Qwen/Qwen3-30B-A3B-FP8 -tp 1 --no-enable-chunked-prefill \
# --max-model-len 32768 --quantization fp8 --kv-cache-dtype fp8 \
# --trust-remote-code --swap-space 16 --distributed-executor-backend mp \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FINAL_Qwen_Qwen3-30B-A3B-FP8-aiter-v0-1-debug4.log 2>&1

# echo "Qwen/Qwen3-30B-A3B-FP8 aiter block gemm"
# VLLM_USE_V1=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=0 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model Qwen/Qwen3-30B-A3B-FP8 -tp 1 --no-enable-chunked-prefill \
# --max-model-len 32768 \
# --trust-remote-code --swap-space 16 --distributed-executor-backend mp \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FINAL_Qwen_Qwen3-30B-A3B-FP8-aiter-v0-1-tuned-triton.log 2>&1

# echo "Qwen/Qwen3-30B-A3B-FP8 aiter block gemm"
# VLLM_USE_V1=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=1 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model Qwen/Qwen3-30B-A3B-FP8 -tp 1 --no-enable-chunked-prefill \
# --max-model-len 32768 \
# --trust-remote-code --swap-space 16 --distributed-executor-backend mp \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FINAL_Qwen_Qwen3-30B-A3B-FP8-aiter-v0-1-tuned-triton.log 2>&1

# Qwen/Qwen3-235B-A22B-FP8
# echo "Qwen/Qwen3-30B-A3B-FP8 aiter block gemm"
# VLLM_USE_V1=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_PAGED_ATTN=1 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model Qwen/Qwen3-30B-A3B-FP8 -tp 1 --no-enable-chunked-prefill \
# --max-model-len 32768 \
# --trust-remote-code --swap-space 16 --distributed-executor-backend mp \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FINAL_Qwen_Qwen3-30B-A3B-FP8-aiter-v0-1-tuned-pa.log 2>&1

# echo "Qwen/Qwen3-235B-A22B-FP8 aiter block gemm debug moe"
# # AITER_TUNE_GEMM=1 \
# VLLM_USE_V1=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_MOE=1 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model Qwen/Qwen3-235B-A22B-FP8 -tp 4 --no-enable-chunked-prefill \
# --max-model-len 32768 --quantization fp8 --kv-cache-dtype fp8 \
# --trust-remote-code --swap-space 16 --distributed-executor-backend mp \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FINAL_Qwen_Qwen3-235B-A22B-FP8-aiter-v0-1-debug-aiter-moe-23.log 2>&1


# echo "Qwen/Qwen3-235B-A22B-FP8 aiter block gemm debug moe"
# # AITER_TUNE_GEMM=1 \
# VLLM_USE_V1=1 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_MOE=1 \
# VLLM_ROCM_USE_AITER_RMSNORM=1 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model Qwen/Qwen3-235B-A22B-FP8 -tp 4 --no-enable-chunked-prefill \
# --max-model-len 32768 \
# --trust-remote-code --swap-space 16 --distributed-executor-backend mp \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FINAL_Qwen_Qwen3-235B-A22B-FP8-aiter-v1-perf-improvement-2.log 2>&1


# echo "Qwen/Qwen3-235B-A22B-FP8 aiter block gemm debug moe"
# # AITER_TUNE_GEMM=1 \
# VLLM_USE_V1=1 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_MOE=1 \
# VLLM_ROCM_USE_AITER_RMSNORM=1 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model Qwen/Qwen3-235B-A22B-FP8 -tp 4 --no-enable-chunked-prefill \
# --max-model-len 32768 \
# --trust-remote-code --swap-space 16 --distributed-executor-backend mp \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FINAL_Qwen_Qwen3-235B-A22B-FP8-aiter-v1-perf-improvement-4.log 2>&1


# echo "Qwen/Qwen3-235B-A22B-FP8 aiter block gemm debug moe"
# # AITER_TUNE_GEMM=1 \
# VLLM_USE_V1=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_MOE=0 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model Qwen/Qwen3-235B-A22B-FP8 -tp 4 --no-enable-chunked-prefill \
# --max-model-len 32768 --quantization fp8 --kv-cache-dtype fp8 \
# --trust-remote-code --swap-space 16 --distributed-executor-backend mp \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FINAL_Qwen_Qwen3-235B-A22B-FP8-aiter-v0-1-debug-triton-moe-22.log 2>&1

# echo "Qwen/Qwen3-235B-A22B-FP8 aiter block gemm"
# # AITER_TUNE_GEMM=1 \
# VLLM_USE_V1=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_MOE=0 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model Qwen/Qwen3-235B-A22B-FP8 -tp 8 --no-enable-chunked-prefill \
# --max-model-len 32768 --quantization fp8 --kv-cache-dtype fp8 --enable-expert-parallel \
# --trust-remote-code --swap-space 16 --distributed-executor-backend mp \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FINAL_Qwen_Qwen3-235B-A22B-FP8-aiter-v0-triton-moe.log 2>&1


# echo "Qwen/Qwen3-30B-A3B-FP8 aiter block gemm"
# VLLM_USE_V1=1 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=0 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model Qwen/Qwen3-30B-A3B-FP8 -tp 4 --no-enable-chunked-prefill \
# --max-model-len 32768 --enable-expert-parallel \
# --trust-remote-code --swap-space 16 --distributed-executor-backend mp \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FINAL_Qwen_Qwen3-30B-A3B-FP8-aiter-v1-2.log 2>&1


# echo "Qwen/Qwen3-30B-A3B-FP8 aiter block gemm"
# VLLM_USE_V1=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=0 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model Qwen/Qwen3-30B-A3B-FP8 -tp 4 --no-enable-chunked-prefill \
# --max-model-len 32768 --enable-expert-parallel \
# --trust-remote-code --swap-space 16 --distributed-executor-backend mp \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FINAL_Qwen_Qwen3-30B-A3B-FP8-aiter-v0-3.log 2>&1


# echo "Qwen/Qwen3-30B-A3B-FP8 aiter block gemm"
# VLLM_USE_V1=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=1 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model Qwen/Qwen3-30B-A3B-FP8 -tp 4 --no-enable-chunked-prefill \
# --max-model-len 32768 --enable-expert-parallel \
# --trust-remote-code --swap-space 16 --distributed-executor-backend mp \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FINAL_Qwen_Qwen3-30B-A3B-FP8-aiter-v0-12.log 2>&1


# echo "Qwen/Qwen3-30B-A3B-FP8 aiter block gemm"
# VLLM_USE_V1=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_2STAGE_MOE=0 \
# VLLM_ROCM_USE_AITER_LINEAR=0 \
# VLLM_ROCM_USE_AITER_RMSNORM=0 \
# VLLM_ROCM_USE_AITER_MLA=0 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model Qwen/Qwen3-30B-A3B-FP8 -tp 4 --no-enable-chunked-prefill \
# --max-model-len 32768 --enable-expert-parallel \
# --trust-remote-code --swap-space 16 --distributed-executor-backend mp \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FINAL_Qwen_Qwen3-30B-A3B-FP8-aiter-v0-13.log 2>&1


# # echo "Qwen/Qwen3-30B-A3B-FP8 aiter block gemm"
# VLLM_USE_V1=1 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=1 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model Qwen/Qwen3-32B -tp 1 --no-enable-chunked-prefill \
# --max-model-len 32768 \
# --trust-remote-code --swap-space 16 --distributed-executor-backend mp \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FINAL_Qwen_Qwen3-32B-aiter-v0-13.log 2>&1


# echo "Qwen/Qwen3-235B-A22B-FP8 aiter block gemm"
# VLLM_USE_V1=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=1 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model Qwen/Qwen3-235B-A22B-FP8 -tp 4 --no-enable-chunked-prefill \
# --max-model-len 32768 \
# --trust-remote-code --swap-space 16 --distributed-executor-backend mp \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FINAL_Qwen_Qwen3-235B-A22B-FP8-aiter-v0-2.log 2>&1

# echo "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic aiter v1"
# VLLM_USE_V1=1 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_RMSNORM=0 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic -tp 8 --gpu_memory_utilization 0.95 \
# --max-model-len 430000 \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FINAL_RedHatAI_Llama-4-Scout-17B-16E-Instruct-FP8-dynamic-aiter-1stage-v1-2.log 2>&1

# # PASSED
# echo "meta-llama/Llama-4-Scout-17B-16E-Instruct aiter v1"
# VLLM_USE_V1=1 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_RMSNORM=0 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model meta-llama/Llama-4-Scout-17B-16E-Instruct -tp 8 --gpu_memory_utilization 0.95 \
# --max-model-len 430000 \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FINAL_meta-llama_Llama-4-Scout-17B-16E-Instruct-aiter-1stage-v1-2.log 2>&1

# # PASSED
# echo "deepseek-ai/DeepSeek-V3"
# VLLM_USE_V1=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=1 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model deepseek-ai/DeepSeek-V3 \
# -tp 8 \
# --no-enable-chunked-prefill \
# --max-model-len 32768 \
# --trust-remote-code \
# --block-size 1 \
# --max_seq_len_to_capture 32768 \
# --trust-remote-code --swap-space 16 --distributed-executor-backend mp \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FINAL_deepseek-ai_DeepSeek-V3-v0-moe.log 2>&1


# # PASSED
# echo "deepseek-ai/DeepSeek-V2-Lite aiter block gemm"
# VLLM_USE_V1=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_RMSNORM=0 \
# VLLM_ROCM_USE_AITER_LINEAR=0 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model deepseek-ai/DeepSeek-V2-Lite -tp 1 --block-size 1 --no-enable-chunked-prefill \
# --max-model-len 32768 --trust-remote-code \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FINAL_deepseek-ai_DeepSeek-V2-Lite-aiter-paged-attn-block-gemm-fp8-4.log 2>&1

# # PASSED
# echo "deepseek-ai/DeepSeek-V2-Lite aiter block gemm"
# HIP_VISIBLE_DEVICES=4,5,6,7 \
# VLLM_USE_V1=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_BLOCK_GEMM=1 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model deepseek-ai/DeepSeek-V2-Lite -tp 1 --block-size 1 --no-enable-chunked-prefill \
# --max-model-len 32768 \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FINAL_deepseek-ai_DeepSeek-V2-Lite-aiter-paged-attn-block-gemm-fp8-2.log 2>&1


# # deepseek-ai/DeepSeek-V2-Lite
# # PASSED
# echo "mistralai/Mixtral-8x7B-Instruct-v0.1 aiter"
# VLLM_USE_V1=1 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_2STAGE_MOE=0 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model mistralai/Mixtral-8x7B-Instruct-v0.1 -tp 2 --gpu_memory_utilization 0.95 \
# --max-model-len 16384 \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FINAL_mistralai_Mixtral-8x7B-Instruct-v0.1-aiter-v1-1stage-moe.log 2>&1


# deepseek-ai/DeepSeek-V2-Lite
# # PASSED
# echo "mistralai/Mixtral-8x7B-Instruct-v0.1 aiter"
# VLLM_USE_V1=1 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_2STAGE_MOE=0 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model mistralai/Mixtral-8x7B-Instruct-v0.1 -tp 2 --gpu_memory_utilization 0.95 \
# --max-model-len 16384 --quantization fp8 \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FINAL_mistralai_Mixtral-8x7B-Instruct-v0.1-fp8-aiter-v1-1stage-moe.log 2>&1


# # PASSED
# echo "amd/Mixtral-8x7B-Instruct-v0.1-FP8-KV aiter"
# VLLM_USE_V1=1 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_2STAGE_MOE=0 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model amd/Mixtral-8x7B-Instruct-v0.1-FP8-KV -tp 2 --gpu_memory_utilization 0.95 \
# --max-model-len 16384 --quantization fp8 \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FASTCHECK_amd_Mixtral-8x7B-Instruct-v0.1-FP8-KV-fp8-aiter-v1-1stage-moe.log 2>&1

# # deepseek-ai/DeepSeek-V2-Lite
# # PASSED
# echo "mistralai/Mixtral-8x7B-Instruct-v0.1 aiter"
# VLLM_USE_V1=1 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_2STAGE_MOE=1 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model mistralai/Mixtral-8x7B-Instruct-v0.1 -tp 2 --gpu_memory_utilization 0.95 \
# --max-model-len 16384 \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FASTCHECK_mistralai_Mixtral-8x7B-Instruct-v0.1-aiter-1stage-v0-3.log 2>&1

# # PASSED
# echo "mistralai/Mixtral-8x7B-Instruct-v0.1 aiter v0"
# VLLM_USE_V1=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_2STAGE_MOE=0 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model mistralai/Mixtral-8x7B-Instruct-v0.1 -tp 2 --gpu_memory_utilization 0.95 \
# --max-model-len 16384 \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FINAL_mistralai_Mixtral-8x7B-Instruct-v0.1-aiter-1stage-v0.log 2>&1

# # PASSED
# echo "mistralai/Mixtral-8x7B-Instruct-v0.1 aiter v1"
# VLLM_USE_V1=1 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_2STAGE_MOE=0 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model mistralai/Mixtral-8x7B-Instruct-v0.1 -tp 2 --gpu_memory_utilization 0.95 \
# --max-model-len 16384 \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FINAL_mistralai_Mixtral-8x7B-Instruct-v0.1-aiter-1stage-v1-2.log 2>&1

# # PASSED
# echo "mistralai/Mixtral-8x7B-Instruct-v0.1 aiter"
# VLLM_USE_V1=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=1 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model mistralai/Mixtral-8x7B-Instruct-v0.1 -tp 2 --gpu_memory_utilization 0.95 \
# --max-model-len 16384 --quantization fp8 --kv-cache-dtype fp8 \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FINAL_mistralai_Mixtral-8x7B-Instruct-v0.1-fp8-aiter-v0-1.log 2>&1

# # # PASSED
# echo "mistralai/Mixtral-8x7B-Instruct-v0.1 aiter"
# VLLM_USE_V1=1 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=1 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model mistralai/Mixtral-8x7B-Instruct-v0.1 -tp 2 --gpu_memory_utilization 0.95 \
# --max-model-len 16384 --quantization fp8 \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FASTCHECK_mistralai_Mixtral-8x7B-Instruct-v0.1-fp8-aiter-v1-1.log 2>&1

# # PASSED
# echo "amd/Mixtral-8x7B-Instruct-v0.1-FP8-KV aiter"
# VLLM_USE_V1=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=1 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model amd/Mixtral-8x7B-Instruct-v0.1-FP8-KV -tp 2 --gpu_memory_utilization 0.95 \
# --max-model-len 16384 --quantization fp8 --kv-cache-dtype fp8 \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FINAL_amd_Mixtral-8x7B-Instruct-v0.1-FP8-KV-fp8-aiter-2stage-v0-1.log 2>&1

# # PASSED
# echo "amd/Mixtral-8x7B-Instruct-v0.1-FP8-KV aiter"
# VLLM_USE_V1=1 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=1 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model amd/Mixtral-8x7B-Instruct-v0.1-FP8-KV -tp 2 --gpu_memory_utilization 0.95 \
# --max-model-len 16384 --quantization fp8 \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FINAL_amd_Mixtral-8x7B-Instruct-v0.1-FP8-KV-fp8-aiter-2stage-v1-1.log 2>&1

# # PASSED
# echo "meta-llama/Llama-3.1-8B-Instruct aiter linear unquantized"
# VLLM_USE_V1=1 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_SKINNY_GEMM=1 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model meta-llama/Llama-3.1-8B-Instruct -tp 1 \
# --max-model-len 32768 \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FINAL_meta-llama_Llama-3.1-8B-Instruct-bf16-aiter-v1.log 2>&1


# # PASSED
# echo "meta-llama/Llama-3.1-8B-Instruct aiter fp8"
# VLLM_USE_V1=1 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_SKINNY_GEMM=1 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model meta-llama/Llama-3.1-8B-Instruct -tp 1 --quantization fp8 \
# --max-model-len 32768 \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FINAL_meta-llama_Llama-3.1-8B-Instruct-fp8-aiter-v1.log 2>&1

# # PASSED
# echo "meta-llama/Llama-3.1-8B-Instruct aiter fp8"
# VLLM_USE_V1=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_SKINNY_GEMM=1 \
# AITER_TUNE_GEMM=1 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model meta-llama/Llama-3.1-8B-Instruct -tp 1 --quantization fp8 --kv-cache-dtype fp8 \
# --max-model-len 32768 \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FINAL_meta-llama_Llama-3.1-8B-Instruct-fp8-aiter-v0-aiter-tune.log 2>&1

# # PASSED
# echo "Qwen/Qwen2.5-7B-Instruct aiter fp8"
# VLLM_USE_V1=1 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_SKINNY_GEMM=1 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model Qwen/Qwen2.5-7B-Instruct -tp 1 --quantization fp8 \
# --max-model-len 32768 \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FINAL_Qwen_Qwen2.5-7B-Instruct-fp8-aiter-v1-1.log 2>&1


# # PASSED
# echo "Qwen/Qwen2.5-72B-Instruct aiter linear unquantized"
# VLLM_USE_V1=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_SKINNY_GEMM=1 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model Qwen/Qwen2.5-72B-Instruct -tp 4 \
# --max-model-len 32768 \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FINAL_Qwen_Qwen2.5-72B-Instruct-bf16-aiter.log 2>&1


# # PASSED
# echo "Qwen/Qwen2.5-72B-Instruct aiter fp8"
# VLLM_USE_V1=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_SKINNY_GEMM=1 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model Qwen/Qwen2.5-72B-Instruct -tp 4 --quantization fp8 --kv-cache-dtype fp8 \
# --max-model-len 32768 \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FINAL_Qwen_Qwen2.5-72B-Instruct-fp8-aiter.log 2>&1

# # PASSED
# echo "Qwen/Qwen2.5-72B-Instruct aiter fp8"
# VLLM_USE_V1=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_PAGED_ATTN=1 \
# VLLM_ROCM_USE_SKINNY_GEMM=1 \
# SAFETENSORS_FAST_GPU=1 \
# python3 test_accuracy.py \
# --model Qwen/Qwen2.5-72B-Instruct -tp 4 --quantization fp8 --kv-cache-dtype fp8 \
# --max-model-len 32768 \
# --input-len 32 --output-len 128 --batch-size 8 --n 1 --num-iters-warmup 10 --num-iters 10 \
# > FINAL_Qwen_Qwen2.5-72B-Instruct-fp8-aiter-pa.log 2>&1