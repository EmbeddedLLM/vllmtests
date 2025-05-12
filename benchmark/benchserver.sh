#!/bin/bash

# declare -a in_len=(128 2048)
# declare -a out_len=(128 2048)

declare -a in_len=(1000)
declare -a out_len=(1000)

help() {
    echo "Usage:"
    echo "bash $0 --dataset /path/to/dataset [ADDITIONAL_VLLM_FLAGS]"
    echo "If no dataset path is specified, the script will only perform benchmarking on random prompts"
    echo ""
}

# Define test cases
declare -a test_cases=(
    # "mistralai/Mixtral-8x7B-v0.1"
    # "mistralai/Mixtral-8x7B-v0.1 fp8"
    # "mistralai/Mixtral-8x7B-v0.1 aiter 1stage"
    # "mistralai/Mixtral-8x7B-v0.1 fp8 aiter 1stage"
    # "Qwen/Qwen2.5-72B-Instruct"
    # "Qwen/Qwen2.5-72B-Instruct fp8"
    # "Qwen/Qwen2.5-72B-Instruct aiter"
    # "Qwen/Qwen2.5-72B-Instruct aiter fp8"
    # "Qwen/Qwen3-235B-A22B-FP8 aiter triton moe"
    # "Qwen/Qwen3-235B-A22B-FP8 aiter moe v0"
    # "Qwen/Qwen3-235B-A22B-FP8 aiter moe v1"
    # "Qwen/Qwen3-235B-A22B-FP8 aiter moe autoquant"
    # "Qwen/Qwen3-235B-A22B-FP8 aiter moe TP8"
    # "deepseek-ai/DeepSeek-V3 v0"
    "deepseek-ai/DeepSeek-V3 v1"
    # "deepseek-ai/DeepSeek-V3 new"
    # "deepseek-ai/DeepSeek-V3 arthur"
)

# Define environment variables for each test case
declare -A env_vars=(    
    # ["mistralai/Mixtral-8x7B-v0.1"]="VLLM_USE_TRITON_FLASH_ATTN=0"
    # ["mistralai/Mixtral-8x7B-v0.1 fp8"]="VLLM_USE_TRITON_FLASH_ATTN=0"
    # ["mistralai/Mixtral-8x7B-v0.1 aiter 1stage"]="VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 VLLM_ROCM_USE_AITER_2STAGE_MOE=0 VLLM_USE_TRITON_FLASH_ATTN=0"
    # ["mistralai/Mixtral-8x7B-v0.1 fp8 aiter 1stage"]="VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 VLLM_ROCM_USE_AITER_2STAGE_MOE=0 VLLM_USE_TRITON_FLASH_ATTN=0"
    # ["Qwen/Qwen2.5-72B-Instruct"]="VLLM_USE_TRITON_FLASH_ATTN=0"
    # ["Qwen/Qwen2.5-72B-Instruct fp8"]="VLLM_USE_TRITON_FLASH_ATTN=0"
    # ["Qwen/Qwen2.5-72B-Instruct aiter"]="VLLM_ROCM_USE_AITER=1 VLLM_USE_TRITON_FLASH_ATTN=0"
    # ["Qwen/Qwen2.5-72B-Instruct aiter fp8"]="VLLM_ROCM_USE_AITER=1 VLLM_USE_TRITON_FLASH_ATTN=0"
    # ["Qwen/Qwen3-235B-A22B-FP8 aiter triton moe"]="VLLM_USE_V1=0 SAFETENSORS_FAST_GPU=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=0 VLLM_USE_TRITON_FLASH_ATTN=0"
    # ["Qwen/Qwen3-235B-A22B-FP8 aiter moe v0"]="VLLM_USE_V1=0 SAFETENSORS_FAST_GPU=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 VLLM_USE_TRITON_FLASH_ATTN=0"
    # ["Qwen/Qwen3-235B-A22B-FP8 aiter moe v1"]="VLLM_USE_V1=1 SAFETENSORS_FAST_GPU=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 VLLM_USE_TRITON_FLASH_ATTN=0"
    # ["Qwen/Qwen3-235B-A22B-FP8 aiter moe autoquant"]="VLLM_USE_V1=0 SAFETENSORS_FAST_GPU=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 VLLM_USE_TRITON_FLASH_ATTN=0"
    # ["Qwen/Qwen3-235B-A22B-FP8 aiter moe TP8"]="VLLM_USE_V1=0 SAFETENSORS_FAST_GPU=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 VLLM_USE_TRITON_FLASH_ATTN=0"
    # ["deepseek-ai/DeepSeek-V3 v0"]="VLLM_RPC_TIMEOUT=1800000 SAFETENSORS_FAST_GPU=1 VLLM_USE_V1=0 VLLM_USE_TRITON_FLASH_ATTN=0 VLLM_ROCM_USE_AITER=1"
    ["deepseek-ai/DeepSeek-V3 v1"]="VLLM_RPC_TIMEOUT=1800000 SAFETENSORS_FAST_GPU=1 VLLM_USE_V1=1 VLLM_USE_TRITON_FLASH_ATTN=0 VLLM_ROCM_USE_AITER=1"
    # ["deepseek-ai/DeepSeek-V3 new"]="SAFETENSORS_FAST_GPU=1 VLLM_USE_V1=0 VLLM_USE_TRITON_FLASH_ATTN=0 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MLA=1 VLLM_ROCM_USE_AITER_BLOCK_GEMM=1 VLLM_ROCM_USE_AITER_FP8_BLOCK_MOE=1"
    # ["deepseek-ai/DeepSeek-V3 arthur"]="SAFETENSORS_FAST_GPU=1 VLLM_USE_V1=0 VLLM_USE_TRITON_FLASH_ATTN=0 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MLA=1 VLLM_ROCM_USE_AITER_BLOCK_GEMM=1 VLLM_ROCM_USE_AITER_FP8_BLOCK_MOE=1"
)


# Define model parameters for each test case
declare -A model_params=(
    # ["mistralai/Mixtral-8x7B-v0.1"]="mistralai/Mixtral-8x7B-v0.1 -tp 1 --max-model-len 16384"
    # ["mistralai/Mixtral-8x7B-v0.1 fp8"]="mistralai/Mixtral-8x7B-v0.1 -tp 1 --quantization fp8 --kv-cache-dtype fp8 --max-model-len 16384"
    # ["mistralai/Mixtral-8x7B-v0.1 aiter 1stage"]="mistralai/Mixtral-8x7B-v0.1 -tp 1 --max-model-len 16384"
    # ["mistralai/Mixtral-8x7B-v0.1 fp8 aiter 1stage"]="mistralai/Mixtral-8x7B-v0.1 -tp 1 --quantization fp8 --kv-cache-dtype fp8 --max-model-len 16384"
    # ["Qwen/Qwen2.5-72B-Instruct"]="Qwen/Qwen2.5-72B-Instruct -tp 4 --max-model-len 32768"
    # ["Qwen/Qwen2.5-72B-Instruct fp8"]="Qwen/Qwen2.5-72B-Instruct -tp 4 --quantization fp8 --kv-cache-dtype fp8 --max-model-len 32768"
    # ["Qwen/Qwen2.5-72B-Instruct aiter"]="Qwen/Qwen2.5-72B-Instruct -tp 4 --max-model-len 32768"
    # ["Qwen/Qwen2.5-72B-Instruct aiter fp8"]="Qwen/Qwen2.5-72B-Instruct -tp 4 --quantization fp8 --kv-cache-dtype fp8 --max-model-len 32768"
    # ["Qwen/Qwen3-235B-A22B-FP8 aiter triton moe"]="Qwen/Qwen3-235B-A22B-FP8 --trust-remote-code --disable-log-requests --max-model-len 32768 --gpu_memory_utilization 0.95 --tensor-parallel-size 4 --max_seq_len_to_capture 32768 --num-scheduler-steps 10 --quantization fp8 --kv-cache-dtype fp8"
    # ["Qwen/Qwen3-235B-A22B-FP8 aiter moe v0"]="Qwen/Qwen3-235B-A22B-FP8 --trust-remote-code --disable-log-requests --max-model-len 32768 --gpu_memory_utilization 0.95 --tensor-parallel-size 4 --max_seq_len_to_capture 32768 --num-scheduler-steps 10 --quantization fp8 --kv-cache-dtype fp8"
    # ["Qwen/Qwen3-235B-A22B-FP8 aiter moe v1"]="Qwen/Qwen3-235B-A22B-FP8 --trust-remote-code --disable-log-requests --max-model-len 32768 --gpu_memory_utilization 0.95 --tensor-parallel-size 4 --max_seq_len_to_capture 32768 --quantization fp8 --no-enable-prefix-caching --max_num_batched_tokens 32768"
    # ["Qwen/Qwen3-235B-A22B-FP8 aiter moe autoquant"]="Qwen/Qwen3-235B-A22B-FP8 --trust-remote-code --disable-log-requests --max-model-len 32768 --gpu_memory_utilization 0.95 --tensor-parallel-size 4 --max_seq_len_to_capture 32768 --num-scheduler-steps 10"
    # ["Qwen/Qwen3-235B-A22B-FP8 aiter moe TP8"]="Qwen/Qwen3-235B-A22B-FP8 --trust-remote-code --disable-log-requests --max-model-len 32768 --gpu_memory_utilization 0.95 --tensor-parallel-size 8 --max_seq_len_to_capture 32768 --num-scheduler-steps 10 --quantization fp8 --kv-cache-dtype fp8"
    # ["deepseek-ai/DeepSeek-V3 v0"]="deepseek-ai/DeepSeek-V3 -tp 8 --max-model-len 32768 --block-size 1 --max_seq_len_to_capture 32768 --no-enable-chunked-prefill"
    ["deepseek-ai/DeepSeek-V3 v1"]="deepseek-ai/DeepSeek-V3 -tp 8 --max-model-len 32768 --block-size 1 --max_seq_len_to_capture 32768 --no-enable-prefix-caching --max-num-batched-tokens 32768"
    # ["deepseek-ai/DeepSeek-V3 new"]="deepseek-ai/DeepSeek-V3 -tp 8 --max-model-len 32768 --block-size 1 --max_seq_len_to_capture 32768"
    # ["deepseek-ai/DeepSeek-V3 arthur"]="deepseek-ai/DeepSeek-V3 -tp 8 --max-model-len 32768 --block-size 1 --max_seq_len_to_capture 32768 --enable-chunked-prefill=False"
)

# Convert test case name to valid filename
get_log_filename() {
    local name="$1"
    echo "${name//[\/ ]/_}.log"
}

# Wait until the server is ready
wait_for_server() {
    local retries=30000
    local wait_time=10
    for ((i=0; i<retries; i++)); do
        if curl -s http://localhost:8007/v1/models > /dev/null; then
            echo "Server is ready."
            return 0
        fi
        echo "Waiting for server to be ready..."
        sleep $wait_time
    done
    echo "Server did not become ready in time."
    return 1
}

main() {
    local DATASET=""
    local SUFFIX=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            --dataset|-dataset)
                DATASET=$2
                shift 2
                ;;
            *)
                SUFFIX="$SUFFIX $1"
                shift
                ;;
        esac
    done

    mkdir -p logs

    for test_case in "${test_cases[@]}"; do
        echo "Running online serving benchmark for: $test_case"
        log_file="logs/$(get_log_filename "$test_case")"
        server_log_file="logs/serverlog_$(get_log_filename "$test_case")"
        echo "Logging output to: $log_file"

        rm -rf /root/.cache/vllm/

        base_env=""
        test_env="${env_vars[$test_case]}"
        full_env="$base_env $test_env"

        # Start vLLM server
        echo "Starting vLLM server..."
        eval "$full_env vllm serve ${model_params[$test_case]} --trust-remote-code --distributed-executor-backend mp --swap-space 16 --disable-log-requests --port 8007 $SUFFIX >> $server_log_file &"
        server_pid=$!
        # --no-enable-chunked-prefill
        # Wait for server to be ready
        sleep 3
        if ! wait_for_server; then
            echo "Server failed to start. Skipping test case: $test_case"
            kill $server_pid
            continue
        fi

        # Extract model name from model_params
        model_name=$(echo "${model_params[$test_case]}" | awk '{print $1}')

        # Loop 1: Dataset-based benchmark
        if [ -n "$DATASET" ]; then
            echo "$log_file"
            echo "Running dataset benchmark..." | tee -a "$log_file"
            python3 ../../aiter-biased-group-topk/benchmarks/benchmark_serving.py \
                --backend vllm \
                --model "$model_name" \
                --dataset-name sharegpt \
                --dataset-path "$DATASET" \
                --num-prompts 500 \
                --port 8007 \
                --goodput "ttft:3000" "tpot:50" >> "$log_file" 2>&1
            sleep 10
        fi

        # Loop 2: Random prompt benchmarks
        for _in_len in "${in_len[@]}"; do
            for _out_len in "${out_len[@]}"; do
                echo "Running random benchmark: in_len=$_in_len, out_len=$_out_len" | tee -a "$log_file"
                python3 ../../aiter-biased-group-topk/benchmarks/benchmark_serving.py \
                    --backend vllm \
                    --model "$model_name" \
                    --dataset-name random \
                    --num-prompts 500 \
                    --goodput "ttft:3000" "tpot:50" \
                    --random-input-len "$_in_len" \
                    --random-range-ratio 0.9 \
                    --port 8007 \
                    --random-output-len "$_out_len" >> "$log_file" 2>&1
                sleep 10
            done
        done

        # Kill the server
        echo "Stopping vLLM server..."
        kill $server_pid
        wait $server_pid 2>/dev/null
        sleep 30
    done
}

main "$@"