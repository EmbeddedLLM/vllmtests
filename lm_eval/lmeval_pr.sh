#!/bin/bash
rm -rf /root/.cache/vllm

# echo "deepseek-ai/DeepSeek-V3 aiter v0"

# VLLM_USE_V1=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# SAFETENSORS_FAST_GPU=1 \
# lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-V3,tensor_parallel_size=8,max_model_len=32768,block_size=1 --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto \
# > pr_gsm8k-deepseek-ai_DeepSeek-V3-aiter-v0.log 2>&1


# echo "deepseek-ai/DeepSeek-V3 aiter v1"

# VLLM_RPC_TIMEOUT=1800000 \
# VLLM_USE_V1=1 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# SAFETENSORS_FAST_GPU=1 \
# lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-V3,tensor_parallel_size=8,max_model_len=32768,block_size=1 --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto \
# > pr_gsm8k-deepseek-ai_DeepSeek-V3-aiter-v1-1.log 2>&1

# echo "deepseek-ai/DeepSeek-V3 aiter v1"

# VLLM_RPC_TIMEOUT=1800000 \
# VLLM_USE_V1=1 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_LINEAR=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# SAFETENSORS_FAST_GPU=1 \
# lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-V3,tensor_parallel_size=8,max_model_len=32768,block_size=1 --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto \
# > pr_gsm8k-deepseek-ai_DeepSeek-V3-aiter-v1-2-noaiterblockscalegemm.log 2>&1

# # VLLM_ROCM_USE_AITER_RMSNORM=0 \
# # pass
# echo "Qwen/Qwen3-235B-A22B-FP8 aiter v0"

# VLLM_USE_V1=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_MOE=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# SAFETENSORS_FAST_GPU=1 \
# lm_eval --model vllm --model_args pretrained=Qwen/Qwen3-235B-A22B-FP8,tensor_parallel_size=4,max_model_len=10000,quantization=fp8,kv_cache_dtype=fp8 --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto \
# > pr_gsm8k-Qwen_Qwen3-235B-A22B-FP8-aiter-triton-moe-v0-1.log 2>&1

# VLLM_ROCM_USE_AITER_RMSNORM=0 \
# # pass
# echo "Qwen/Qwen3-235B-A22B-FP8 aiter v0"

# VLLM_USE_V1=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# SAFETENSORS_FAST_GPU=1 \
# lm_eval --model vllm --model_args pretrained=Qwen/Qwen3-235B-A22B-FP8,tensor_parallel_size=4,max_model_len=10000,quantization=fp8,kv_cache_dtype=fp8 --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto \
# > pr_gsm8k-Qwen_Qwen3-235B-A22B-FP8-aiter-v0-2.log 2>&1


# echo "Qwen/Qwen3-235B-A22B-FP8 no aiter"
# VLLM_USE_V1=0 \
# VLLM_ROCM_USE_AITER=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER_MOE=0 \
# SAFETENSORS_FAST_GPU=1 \
# lm_eval --model vllm --model_args pretrained=Qwen/Qwen3-235B-A22B-FP8,tensor_parallel_size=4,max_model_len=10000 --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto \
# > pr_gsm8k-Qwen_Qwen3-235B-A22B-FP8-no-aiter-v0-1.log 2>&1


# echo "Qwen/Qwen3-235B-A22B-FP8 with aiter native rms norm, triton fused moe"
# VLLM_USE_V1=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER_MOE=0 \
# VLLM_ROCM_USE_AITER_RMSNORM=0 \
# SAFETENSORS_FAST_GPU=1 \
# lm_eval --model vllm --model_args pretrained=Qwen/Qwen3-235B-A22B-FP8,tensor_parallel_size=4,max_model_len=10000 --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto \
# > pr_gsm8k-Qwen_Qwen3-235B-A22B-FP8-triton-fused-moe-PR-v0-1.log 2>&1

# sleep 10


# echo "Qwen/Qwen3-235B-A22B-FP8 with aiter fused moe v1 kvcachedtype auto quant auto"

# VLLM_USE_V1=1 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_RMSNORM=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# SAFETENSORS_FAST_GPU=1 \
# lm_eval --model vllm --model_args pretrained=Qwen/Qwen3-235B-A22B-FP8,tensor_parallel_size=4,max_model_len=10000 --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto \
# > pr_gsm8k-Qwen_Qwen3-235B-A22B-FP8-aiter-fused-moe-grouped-topk-PR-v1-3.log 2>&1


# echo "Qwen/Qwen3-235B-A22B-FP8 with aiter fused moe v0 kvcachedtype auto quant auto"

# VLLM_USE_V1=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_RMSNORM=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# SAFETENSORS_FAST_GPU=1 \
# lm_eval --model vllm --model_args pretrained=Qwen/Qwen3-235B-A22B-FP8,tensor_parallel_size=4,max_model_len=10000 --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto \
# > pr_gsm8k-Qwen_Qwen3-235B-A22B-FP8-aiter-fused-moe-grouped-topk-PR-v0-2.log 2>&1


# echo "Qwen/Qwen3-235B-A22B-FP8 triton v0"

# VLLM_USE_V1=0 \
# VLLM_ROCM_USE_AITER=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# SAFETENSORS_FAST_GPU=1 \
# lm_eval --model vllm --model_args pretrained=Qwen/Qwen3-235B-A22B-FP8,tensor_parallel_size=4,max_model_len=10000,quantization=fp8,kv_cache_dtype=fp8 --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto \
# > pr_gsm8k-Qwen_Qwen3-235B-A22B-FP8-triton-v0-1.log 2>&1


















# echo "Qwen/Qwen3-32B no aiter"

# VLLM_USE_V1=0 \
# VLLM_ROCM_USE_AITER=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER_MOE=0 \
# SAFETENSORS_FAST_GPU=1 \
# lm_eval --model vllm --model_args pretrained=Qwen/Qwen3-32B,tensor_parallel_size=1,max_model_len=10000 --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto \
# > pr_gsm8k-Qwen_Qwen3-32B-no-aiter-v0-1.log 2>&1


# echo "Qwen/Qwen3-32B with aiter and rms norm, triton fused moe"

# VLLM_USE_V1=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER_MOE=0 \
# SAFETENSORS_FAST_GPU=1 \
# lm_eval --model vllm --model_args pretrained=Qwen/Qwen3-32B,tensor_parallel_size=4,max_model_len=10000 --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto \
# > pr_gsm8k-Qwen_Qwen3-32B-aiter-rmsnorm-triton-fused-moe-v0-1.log 2>&1






# echo "Qwen/Qwen3-32B no aiter"

# VLLM_USE_V1=0 \
# VLLM_ROCM_USE_AITER=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER_MOE=0 \
# SAFETENSORS_FAST_GPU=1 \
# lm_eval --model vllm --model_args pretrained=Qwen/Qwen3-32B,tensor_parallel_size=1,max_model_len=10000 --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto \
# > pr_gsm8k-Qwen_Qwen3-32B-no-aiter-v0-1.log 2>&1


# echo "Qwen/Qwen3-32B with aiter and rms norm, triton fused moe"

# VLLM_USE_V1=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER_MOE=0 \
# SAFETENSORS_FAST_GPU=1 \
# lm_eval --model vllm --model_args pretrained=Qwen/Qwen3-32B,tensor_parallel_size=4,max_model_len=10000 --trust_remote_code --tasks gsm8k --num_fewshot 5 --batch_size auto \
# > pr_gsm8k-Qwen_Qwen3-32B-aiter-rmsnorm-triton-fused-moe-v0-1.log 2>&1
