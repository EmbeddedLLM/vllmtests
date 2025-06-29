#!/bin/bash

declare -a in_len=(2000 1000 5000 10000 3200)
declare -a out_len=(150 1000 1000 1000 800)

help() {
    echo "Usage:"
    echo "bash $0 --dataset /path/to/dataset [ADDITIONAL_VLLM_FLAGS]"
    echo "If no dataset path is specified, the script will only perform benchmarking on random prompts"
    echo ""
}

# Define test cases
declare -a test_cases=(
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct v1 Chunked Size 32768 no prefix caching no aiter"
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct v1 Chunked Size 32768 no prefix caching yes aiter"
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct v1 Chunked Size 32768 yes prefix caching no aiter"
    "meta-llama/Llama-4-Maverick-17B-128E-Instruct v1 Chunked Size 32768 yes prefix caching yes aiter"
)

# Define environment variables for each test case
declare -A env_vars=(
    ["meta-llama/Llama-4-Maverick-17B-128E-Instruct v1 Chunked Size 32768 no prefix caching no aiter"]="VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 SAFETENSORS_FAST_GPU=1 VLLM_USE_V1=1"
    ["meta-llama/Llama-4-Maverick-17B-128E-Instruct v1 Chunked Size 32768 no prefix caching yes aiter"]="VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 SAFETENSORS_FAST_GPU=1 VLLM_USE_V1=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MHA=0 VLLM_ROCM_USE_AITER_RMSNORM=0"
    ["meta-llama/Llama-4-Maverick-17B-128E-Instruct v1 Chunked Size 32768 yes prefix caching no aiter"]="VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 SAFETENSORS_FAST_GPU=1 VLLM_USE_V1=1"
    ["meta-llama/Llama-4-Maverick-17B-128E-Instruct v1 Chunked Size 32768 yes prefix caching yes aiter"]="VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 SAFETENSORS_FAST_GPU=1 VLLM_USE_V1=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MHA=0 VLLM_ROCM_USE_AITER_RMSNORM=0"

)


# Define model parameters for each test case
declare -A model_params=(
    ["meta-llama/Llama-4-Maverick-17B-128E-Instruct v1 Chunked Size 32768 no prefix caching no aiter"]="meta-llama/Llama-4-Maverick-17B-128E-Instruct -tp 8  --max_num_batched_tokens 32768 --max-num-seqs 1024 --max-model-len 36000  --gpu_memory_utilization 0.95 --no-enable-prefix-caching"
    ["meta-llama/Llama-4-Maverick-17B-128E-Instruct v1 Chunked Size 32768 no prefix caching yes aiter"]="meta-llama/Llama-4-Maverick-17B-128E-Instruct -tp 8  --max_num_batched_tokens 32768 --max-num-seqs 1024 --max-model-len 36000  --gpu_memory_utilization 0.95 --no-enable-prefix-caching"
    ["meta-llama/Llama-4-Maverick-17B-128E-Instruct v1 Chunked Size 32768 yes prefix caching no aiter"]="meta-llama/Llama-4-Maverick-17B-128E-Instruct -tp 8 --max_num_batched_tokens 32768 --max-num-seqs 1024 --max-model-len 36000  --gpu_memory_utilization 0.95"
    ["meta-llama/Llama-4-Maverick-17B-128E-Instruct v1 Chunked Size 32768 yes prefix caching yes aiter"]="meta-llama/Llama-4-Maverick-17B-128E-Instruct -tp 8 --max_num_batched_tokens 32768 --max-num-seqs 1024 --max-model-len 36000  --gpu_memory_utilization 0.95"

)

declare -A folder_name=(
    ["meta-llama/Llama-4-Maverick-17B-128E-Instruct v1 Chunked Size 32768 no prefix caching no aiter"]="Llama-4-Maverick-17B-128E-Instruct-v1-Chunked-Size-32768-no-prefix-caching-no-aiter"
    ["meta-llama/Llama-4-Maverick-17B-128E-Instruct v1 Chunked Size 32768 no prefix caching yes aiter"]="Llama-4-Maverick-17B-128E-Instruct-v1-Chunked-Size-32768-no-prefix-caching-yes-aiter"
    ["meta-llama/Llama-4-Maverick-17B-128E-Instruct v1 Chunked Size 32768 yes prefix caching no aiter"]="Llama-4-Maverick-17B-128E-Instruct-v1-Chunked-Size-32768-yes-prefix-caching-no-aiter"
    ["meta-llama/Llama-4-Maverick-17B-128E-Instruct v1 Chunked Size 32768 yes prefix caching yes aiter"]="Llama-4-Maverick-17B-128E-Instruct-v1-Chunked-Size-32768-yes-prefix-caching-yes-aiter"

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
    local DATASET="/path/to//ShareGPT_V3_unfiltered_cleaned_split.json"
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

    mkdir -p logs_4

    for test_case in "${test_cases[@]}"; do
        res_dir="llama4_blogpost/prefil_decoder_attn_true/${folder_name[$test_case]}"

        echo "Running online serving benchmark for: $test_case"
        log_file="logs_4/$(get_log_filename "$test_case")"
        server_log_file="logs_4/serverlog_$(get_log_filename "$test_case")"
        echo "Logging output to: $log_file"

        mkdir -p $res_dir
        echo "save results in: $res_dir"

        base_env=""
        test_env="${env_vars[$test_case]}"
        full_env="$base_env $test_env"

        # Start vLLM server
        echo "Starting vLLM server..."
        eval "$full_env vllm serve ${model_params[$test_case]} --trust-remote-code --distributed-executor-backend mp --swap-space 16 --disable-log-requests --compilation-config '{\"full_cuda_graph\": true}' --port 8007 $SUFFIX >> $server_log_file &"
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
        # if [ -n "$DATASET" ]; then
        #     echo "$log_file"
        #     echo "Running dataset benchmark..." | tee -a "$log_file"
        #     python3 /path/to//vllm/benchmarks/benchmark_serving.py \
        #         --backend vllm \
        #         --model "$model_name" \
        #         --dataset-name sharegpt \
        #         --dataset-path "$DATASET" \
        #         --save-result \
        #         --max-concurrency 64 \
        #         --num-prompts 640 \
        #         --ignore-eos \
        #         --percentile_metrics ttft,tpot,itl,e2el \
        #         --result-filename "sharegpt.json" \
        #         --result-dir "/path/to//$res_dir"\
        #         --port 8007 >> "$log_file" 2>&1
        #     sleep 10
        # fi

        # Loop 2: Random prompt benchmarks
        for i in "${!in_len[@]}"; do
            _in_len=${in_len[i]}
            _out_len=${out_len[i]}
            echo "Running random benchmark: in_len=$_in_len, out_len=$_out_len" | tee -a "$log_file"
            python3 /path/to//vllm/benchmarks/benchmark_serving.py \
                --backend vllm \
                --model "$model_name" \
                --dataset-name random \
                --random-input-len "$_in_len" \
                --save-result \
                --result-filename "ISL-$_in_len-OSL-$_out_len.json" \
                --result-dir "/path/to//$res_dir" \
                --port 8007 \
                --max-concurrency 64 \
                --num-prompts 640 \
                --ignore-eos \
                --percentile_metrics ttft,tpot,itl,e2el \
                --random-output-len "$_out_len" >> "$log_file" 2>&1
            sleep 10
        done

        # Kill the server
        echo "Stopping vLLM server..."
        kill $server_pid
        wait $server_pid 2>/dev/null
        sleep 30
    done
}

main "$@"