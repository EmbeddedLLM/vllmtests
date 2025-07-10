#!/bin/bash
export HF_TOKEN=""
rm -rf /root/.cache/vllm

LOG_DIR=logs_gsm8k_cot_llamafp8_padebug

mkdir -p $LOG_DIR


export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# lm_eval \
#   --model vllm \
#   --model_args pretrained=$MODEL,add_bos_token=True \
#   --tasks gsm8k_cot \
#   --num_fewshot 8 \
#   --batch_size 128 --limit 250





echo "deepseek-r1-FP8-Dynamic-from-BF16-calib1024 with aiter"

# AITER_LOG_MORE=1 \
# HF_HUB_OFFLINE=1 \
VLLM_USE_V1=0 \
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_USE_AITER_ASMMOE=1 \
SAFETENSORS_FAST_GPU=1 \
lm_eval --model vllm --model_args pretrained=/app/model/QuantLLM/deepseek-r1-FP8-Dynamic-from-BF16-calib1024,tensor_parallel_size=8,add_bos_token=True,block_size=1 --trust_remote_code --tasks gsm8k_cot --num_fewshot 8 --batch_size 250 --seed 1234 \
> $LOG_DIR/gsm8k_cot_deepseek-r1-FP8-Dynamic-from-BF16-calib1024-v0-aiter-400eff4-bf16kvcache-tp8-asmmoe-beforebugfix.log 2>&1



# echo "deepseek-ai/DeepSeek-R1 with aiter"

# # VLLM_ROCM_USE_AITER_MOE=1 \
# # AITER_LOG_MORE=1 \
# HF_HUB_OFFLINE=1 \
# VLLM_USE_V1=1 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_LINEAR=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER_RMSNORM=0 \
# VLLM_ROCM_USE_AITER_MOE=1 \
# VLLM_ROCM_USE_AITER_PAGED_ATTN=0 \
# SAFETENSORS_FAST_GPU=1 \
# lm_eval --model vllm --model_args pretrained=deepseek-ai/DeepSeek-R1,tensor_parallel_size=8,enable_expert_parallel=True,add_bos_token=True --trust_remote_code --tasks gsm8k_cot --num_fewshot 8 --batch_size 250 --seed 1234 \
# > $LOG_DIR/gsm8k_cot_deepseek-ai_DeepSeek-R1-v1-noaiter-400eff4-bf16kvcache-tp8-ep8-250-fusedmoe.log 2>&1


# echo "RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic with aiter"

# # VLLM_ROCM_USE_AITER_MOE=1 \
# # AITER_LOG_MORE=1 \
# HF_HUB_OFFLINE=1 \
# VLLM_USE_V1=1 \
# VLLM_ROCM_USE_AITER=0 \
# VLLM_ROCM_USE_AITER_LINEAR=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER_RMSNORM=0 \
# VLLM_ROCM_USE_AITER_MHA=0 \
# VLLM_ROCM_USE_AITER_MOE=1 \
# VLLM_ROCM_USE_AITER_PAGED_ATTN=0 \
# VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 \
# SAFETENSORS_FAST_GPU=1 \
# lm_eval --model vllm --model_args pretrained=RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic,tensor_parallel_size=8,enable_expert_parallel=True,add_bos_token=True,max_model_len=10000 --trust_remote_code --tasks gsm8k_cot --num_fewshot 8 --batch_size 250 --seed 1234 \
# > $LOG_DIR/gsm8k_cot_RedHatAI_Llama-4-Scout-17B-16E-Instruct-FP8-dynamic-v1-noaiter-400eff4-bf16kvcache-tp8-ep8-250-fusedmoe.log 2>&1


# echo "meta-llama/Llama-4-Scout-17B-16E-Instruct with aiter"

# # VLLM_ROCM_USE_AITER_MOE=1 \
# # AITER_LOG_MORE=1 \
# HF_HUB_OFFLINE=1 \
# VLLM_USE_V1=1 \
# VLLM_ROCM_USE_AITER=0 \
# VLLM_ROCM_USE_AITER_LINEAR=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER_RMSNORM=0 \
# VLLM_ROCM_USE_AITER_MHA=0 \
# VLLM_ROCM_USE_AITER_MOE=1 \
# VLLM_ROCM_USE_AITER_PAGED_ATTN=0 \
# VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 \
# SAFETENSORS_FAST_GPU=1 \
# lm_eval --model vllm --model_args pretrained=meta-llama/Llama-4-Scout-17B-16E-Instruct,tensor_parallel_size=8,enable_expert_parallel=True,add_bos_token=True,max_model_len=10000 --trust_remote_code --tasks gsm8k_cot --num_fewshot 8 --batch_size 250 --seed 1234 \
# > $LOG_DIR/gsm8k_cot_meta-llama_Llama-4-Scout-17B-16E-Instruct-v1-noaiter-400eff4-bf16kvcache-tp8-ep8-250-fusedmoe.log 2>&1



# echo "Qwen/Qwen3-235B-A22B-FP8 with aiter"

# # VLLM_ROCM_USE_AITER_MOE=1 \
# # AITER_LOG_MORE=1 \
# HF_HUB_OFFLINE=1 \
# VLLM_USE_V1=1 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_LINEAR=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER_RMSNORM=0 \
# VLLM_ROCM_USE_AITER_MHA=0 \
# VLLM_ROCM_USE_AITER_MOE=1 \
# VLLM_ROCM_USE_AITER_PAGED_ATTN=0 \
# SAFETENSORS_FAST_GPU=1 \
# lm_eval --model vllm --model_args pretrained=Qwen/Qwen3-235B-A22B-FP8,tensor_parallel_size=8,enable_expert_parallel=True,add_bos_token=True --trust_remote_code --tasks gsm8k_cot --num_fewshot 8 --batch_size 250 --seed 1234 \
# > $LOG_DIR/gsm8k_cot_Qwen_Qwen3-235B-A22B-FP8-v1-aiter-400eff4-bf16kvcache-tp8-ep8-250-fusedmoe.log 2>&1



# echo "RedHatAI/Qwen3-235B-A22B-FP8-dynamic with aiter"

# # AITER_LOG_MORE=1 \
# HF_HUB_OFFLINE=1 \
# VLLM_USE_V1=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_LINEAR=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER_RMSNORM=0 \
# VLLM_ROCM_USE_AITER_MOE=1 \
# VLLM_ROCM_USE_AITER_PAGED_ATTN=0 \
# SAFETENSORS_FAST_GPU=1 \
# lm_eval --model vllm --model_args pretrained=RedHatAI/Qwen3-235B-A22B-FP8-dynamic,tensor_parallel_size=8,enable_expert_parallel=True,add_bos_token=True --trust_remote_code --tasks gsm8k_cot --num_fewshot 8 --batch_size 500 --seed 1234 \
# > $LOG_DIR/gsm8k_cot_RedHatAI_Qwen3-235B-A22B-FP8-dynamic-v0-aiter-400eff4-bf16kvcache-tp8-ep8-500-expertmask-fusedmoe-afterbugfix.log 2>&1


# echo "RedHatAI/Qwen3-235B-A22B-FP8-dynamic with aiter"

# # AITER_LOG_MORE=1 \
# HF_HUB_OFFLINE=1 \
# VLLM_USE_V1=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_LINEAR=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER_RMSNORM=0 \
# VLLM_ROCM_USE_AITER_MOE=1 \
# VLLM_ROCM_USE_AITER_PAGED_ATTN=0 \
# SAFETENSORS_FAST_GPU=1 \
# lm_eval --model vllm --model_args pretrained=RedHatAI/Qwen3-235B-A22B-FP8-dynamic,tensor_parallel_size=8,enable_expert_parallel=True,add_bos_token=True --trust_remote_code --tasks gsm8k_cot --num_fewshot 8 --batch_size 250 --seed 1234 \
# > $LOG_DIR/gsm8k_cot_RedHatAI_Qwen3-235B-A22B-FP8-dynamic-v0-aiter-400eff4-bf16kvcache-tp8-ep8-250-expertmask.log 2>&1



# echo "EmbeddedLLM/Qwen2.5-1.5B-FP8-Dynamic with aiter"

# # AITER_LOG_MORE=1 \
# HF_HUB_OFFLINE=1 \
# VLLM_USE_V1=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_LINEAR=1 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER_RMSNORM=0 \
# VLLM_ROCM_USE_AITER_PAGED_ATTN=0 \
# SAFETENSORS_FAST_GPU=1 \
# lm_eval --model vllm --model_args pretrained=EmbeddedLLM/Qwen2.5-1.5B-FP8-Dynamic,tensor_parallel_size=1,gpu_memory_utilization=0.8,add_bos_token=True --trust_remote_code --tasks gsm8k_cot --num_fewshot 8 --batch_size 500 --seed 1234 \
# > $LOG_DIR/gsm8k_cot_EmbeddedLLM_Qwen2.5-1.5B-FP8-Dynamic-v0-aiter-2-400eff4-bf16kvcache-500.log 2>&1


# echo "EmbeddedLLM/Qwen2.5-1.5B-Instruct-FP8-Dynamic with aiter"

# # AITER_LOG_MORE=1 \
# HF_HUB_OFFLINE=1 \
# VLLM_USE_V1=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_LINEAR=1 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER_RMSNORM=0 \
# VLLM_ROCM_USE_AITER_PAGED_ATTN=1 \
# SAFETENSORS_FAST_GPU=1 \
# lm_eval --model vllm --model_args pretrained=EmbeddedLLM/Qwen2.5-1.5B-Instruct-FP8-Dynamic,tensor_parallel_size=1,gpu_memory_utilization=0.8,add_bos_token=True --trust_remote_code --tasks gsm8k_cot --num_fewshot 8 --batch_size 400 --seed 1234 \
# > $LOG_DIR/gsm8k_cot_EmbeddedLLM_Qwen2.5-1.5B-Instruct-FP8-Dynamic-v0-aiter-2-400eff4-bf16kvcache-400.log 2>&1


# echo "mistralai/Mistral-7B-Instruct-v0.3 with aiter"

# # AITER_LOG_MORE=1 \
# HF_HUB_OFFLINE=1 \
# VLLM_USE_V1=0 \
# VLLM_ROCM_USE_AITER=0 \
# VLLM_ROCM_USE_AITER_LINEAR=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER_RMSNORM=0 \
# VLLM_ROCM_USE_AITER_PAGED_ATTN=1 \
# SAFETENSORS_FAST_GPU=1 \
# lm_eval --model vllm --model_args pretrained=mistralai/Mistral-7B-Instruct-v0.3,tensor_parallel_size=1,gpu_memory_utilization=0.8,add_bos_token=True --trust_remote_code --tasks gsm8k_cot --num_fewshot 8 --batch_size 400 --seed 1234 \
# > $LOG_DIR/gsm8k_cot_mistralai_Mistral-7B-Instruct-v0.3-v0-noaiter-2-400eff4-fp16kvcache-400.log 2>&1


# echo "mistralai/Mistral-7B-Instruct-v0.1 with aiter"

# # AITER_LOG_MORE=1 \
# # HF_HUB_OFFLINE=1 \
# VLLM_USE_V1=0 \
# VLLM_ROCM_USE_AITER=0 \
# VLLM_ROCM_USE_AITER_LINEAR=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER_RMSNORM=0 \
# VLLM_ROCM_USE_AITER_PAGED_ATTN=1 \
# SAFETENSORS_FAST_GPU=1 \
# lm_eval --model vllm --model_args pretrained=mistralai/Mistral-7B-Instruct-v0.1,tensor_parallel_size=1,gpu_memory_utilization=0.8,add_bos_token=True,kv_cache_dtype=fp8 --trust_remote_code --tasks gsm8k_cot --num_fewshot 8 --batch_size 400 --seed 1234 \
# > $LOG_DIR/gsm8k_cot_mistralai_Mistral-7B-Instruct-v0.1-v0-noaiter-2-400eff4-fp8kvcache-400.log 2>&1

# mistralai/Mistral-7B-Instruct-v0.1
# echo "EmbeddedLLM/Qwen2.5-32B-Instruct-FP8-Dynamic with aiter"

# # AITER_LOG_MORE=1 \
# HF_HUB_OFFLINE=1 \
# VLLM_USE_V1=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_LINEAR=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER_RMSNORM=0 \
# VLLM_ROCM_USE_AITER_PAGED_ATTN=1 \
# SAFETENSORS_FAST_GPU=1 \
# lm_eval --model vllm --model_args pretrained=EmbeddedLLM/Qwen2.5-32B-Instruct-FP8-Dynamic,tensor_parallel_size=8 --trust_remote_code --tasks gsm8k_cot --num_fewshot 8 --batch_size 128 --seed 1234 \
# > $LOG_DIR/gsm8k_cot_EmbeddedLLM_Qwen2.5-32B-Instruct-FP8-Dynamic-v0-1-400eff4-lmeval048.log 2>&1


# echo "EmbeddedLLM/Qwen2.5-32B-Instruct-FP8-Dynamic with aiter"

# # AITER_LOG_MORE=1 \
# HF_HUB_OFFLINE=1 \
# VLLM_USE_V1=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_LINEAR=1 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER_RMSNORM=0 \
# VLLM_ROCM_USE_AITER_PAGED_ATTN=1 \
# SAFETENSORS_FAST_GPU=1 \
# lm_eval --model vllm --model_args pretrained=EmbeddedLLM/Qwen2.5-32B-Instruct-FP8-Dynamic,tensor_parallel_size=4,kv_cache_dtype=fp8 --trust_remote_code --tasks gsm8k_cot --num_fewshot 8 --batch_size 128 --seed 1234 \
# > $LOG_DIR/gsm8k_cot_EmbeddedLLM_Qwen2.5-32B-Instruct-FP8-Dynamic-v0-aiter-1-400eff4-fp8kvcache-noncontiguousquery.log 2>&1


# echo "EmbeddedLLM/Qwen2.5-32B-Instruct-FP8-Dynamic with aiter"

# # AITER_LOG_MORE=1 \
# HF_HUB_OFFLINE=1 \
# VLLM_USE_V1=0 \
# VLLM_ROCM_USE_AITER=0 \
# VLLM_ROCM_USE_AITER_LINEAR=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER_RMSNORM=0 \
# VLLM_ROCM_USE_AITER_PAGED_ATTN=1 \
# SAFETENSORS_FAST_GPU=1 \
# lm_eval --model vllm --model_args pretrained=EmbeddedLLM/Qwen2.5-32B-Instruct-FP8-Dynamic,tensor_parallel_size=8 --trust_remote_code --tasks gsm8k_cot --num_fewshot 8 --batch_size 128 --seed 1234 \
# > $LOG_DIR/gsm8k_cot_EmbeddedLLM_Qwen2.5-32B-Instruct-FP8-Dynamic-v0-noaiter-1-400eff4-bf16kvcache-128.log 2>&1

# echo "Qwen/Qwen3-235B-A22B-FP8 with aiter"

# # AITER_LOG_MORE=1 \
# HF_HUB_OFFLINE=1 \
# VLLM_USE_V1=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_LINEAR=0 \
# VLLM_ROCM_USE_AITER_MOE=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER_RMSNORM=0 \
# VLLM_ROCM_USE_AITER_PAGED_ATTN=1 \
# SAFETENSORS_FAST_GPU=1 \
# lm_eval --model vllm --model_args pretrained=Qwen/Qwen3-235B-A22B-FP8,tensor_parallel_size=4,dtype=half --trust_remote_code --tasks gsm8k_cot --num_fewshot 8 --batch_size 128 --seed 1234 \
# > $LOG_DIR/gsm8k_cot_Qwen_Qwen3-235B-A22B-FP8-v0-aiter-1-400eff4-fp16kvcache-128.log 2>&1


# echo "Qwen/Qwen3-235B-A22B-FP8 with aiter"

# # AITER_LOG_MORE=1 \
# HF_HUB_OFFLINE=1 \
# VLLM_USE_V1=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_LINEAR=0 \
# VLLM_ROCM_USE_AITER_MOE=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER_RMSNORM=0 \
# VLLM_ROCM_USE_AITER_PAGED_ATTN=1 \
# SAFETENSORS_FAST_GPU=1 \
# lm_eval --model vllm --model_args pretrained=Qwen/Qwen3-235B-A22B-FP8,tensor_parallel_size=4,kv_cache_dtype=fp8 --trust_remote_code --tasks gsm8k_cot --num_fewshot 8 --batch_size 128 --seed 1234 \
# > logs_gsm8k_cot/gsm8k_cot_Qwen_Qwen3-235B-A22B-FP8-v0-aiter-1-400eff4-fp8kvcache.log 2>&1


# echo "EmbeddedLLM/Qwen2.5-1.5B-Instruct-FP8-Dynamic with aiter"

# # AITER_LOG_MORE=1 \
# HF_HUB_OFFLINE=1 \
# VLLM_USE_V1=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_LINEAR=0 \
# VLLM_ROCM_USE_AITER_MOE=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER_RMSNORM=0 \
# VLLM_ROCM_USE_AITER_PAGED_ATTN=1 \
# SAFETENSORS_FAST_GPU=1 \
# lm_eval --model vllm --model_args pretrained=EmbeddedLLM/Qwen2.5-1.5B-Instruct-FP8-Dynamic,tensor_parallel_size=1,kv_cache_dtype=fp8,add_bos_token=True --trust_remote_code --tasks gsm8k_cot --num_fewshot 8 --batch_size 128 --seed 1234 \
# > logs_gsm8k_cot/gsm8k_cot_EmbeddedLLM_Qwen2.5-1.5B-Instruct-FP8-Dynamic-v0-aiter-1-400eff4-fp8kvcache.log 2>&1


# echo "EmbeddedLLM/Qwen2.5-1.5B-Instruct-FP8-Dynamic with aiter"

# # AITER_LOG_MORE=1 \
# HF_HUB_OFFLINE=1 \
# VLLM_USE_V1=0 \
# VLLM_ROCM_USE_AITER=0 \
# VLLM_ROCM_USE_AITER_LINEAR=0 \
# VLLM_ROCM_USE_AITER_MOE=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER_RMSNORM=0 \
# VLLM_ROCM_USE_AITER_PAGED_ATTN=0 \
# SAFETENSORS_FAST_GPU=1 \
# lm_eval --model vllm --model_args pretrained=EmbeddedLLM/Qwen2.5-1.5B-Instruct-FP8-Dynamic,tensor_parallel_size=1,kv_cache_dtype=fp8,add_bos_token=True --trust_remote_code --tasks gsm8k_cot --num_fewshot 8 --batch_size 128 --seed 1234 \
# > logs_gsm8k_cot/vllm_upstream_gsm8k_cot_EmbeddedLLM_Qwen2.5-1.5B-Instruct-FP8-Dynamic-v0-aiter-1-400eff4-fp8kvcache.log 2>&1





# echo "meta-llama/Llama-3.3-70B-Instruct with aiter"

# # AITER_LOG_MORE=1 \
# HF_HUB_OFFLINE=1 \
# VLLM_USE_V1=0 \
# VLLM_ROCM_USE_AITER=1 \
# VLLM_ROCM_USE_AITER_LINEAR=0 \
# VLLM_ROCM_USE_AITER_MOE=0 \
# VLLM_USE_TRITON_FLASH_ATTN=0 \
# VLLM_ROCM_USE_AITER_RMSNORM=0 \
# VLLM_ROCM_USE_AITER_PAGED_ATTN=1 \
# SAFETENSORS_FAST_GPU=1 \
# lm_eval --model vllm --model_args pretrained=meta-llama/Llama-3.3-70B-Instruct,tensor_parallel_size=2 --trust_remote_code --tasks gsm8k_cot --num_fewshot 8 --batch_size 128 --seed 1234 \
# > logs_gsm8k_cot/vllm_upstream_gsm8k_cot_meta-llama_Llama-3.3-70B-Instruct-v0-aiter-1-400eff4-bf16kvcache.log 2>&1