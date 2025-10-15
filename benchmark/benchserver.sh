#!/bin/bash

# Default configuration file
CONFIG_FILE="${CONFIG_FILE:-}"

# Input/output lengths for random benchmarks
# These can be overridden by the configuration file
in_len=(2000 1000 5000 10000 3200)
out_len=(150 1000 1000 1000 800)

# Basic configuration
# update these
VLLM_PORT=11169
LOGS_PATH="logs"
RESULT_DIR="results"
HIP_VISIBLE_DEVICES=0

# Initialize server_pid
server_pid=""

# Declare associative arrays that will be populated by configuration
declare -A env_vars
declare -A model_params
declare -A folder_name

# Function to load configuration file
load_config() {
    local config_file="$1"

    if [ -f "$config_file" ]; then
        echo "Loading configuration from $config_file"
        # Source the config file in the current shell context
        . "$config_file"
    else
        echo "ERROR: Configuration file $config_file not found!"
        exit 1
    fi
}

# Load configuration if provided
if [ -n "$CONFIG_FILE" ]; then
    load_config "$CONFIG_FILE"
fi

help() {
    echo "Usage:"
    echo "bash $0 --config /path/to/config.sh [--stop-on-server-failure]"
    echo "Options:"
    echo "  --config /path/to/config.sh     Path to configuration script (required)"
    echo "  --stop-on-server-failure        Stop the entire script if server dies (default: continue)"
    echo "  --help                          Show this help message"
    echo ""
    echo "The configuration script should define the following arrays/associative arrays:"
    echo "  - test_cases[]             : Array of test case definitions"
    echo "  - env_vars[test_case]      : Associative array of environment variables"
    echo "  - model_params[test_case]  : Associative array of model parameters"
    echo "  - folder_name[test_case]   : Associative array of folder names"
    echo "  - VLLM_PORT                : VLLM server port for benchmark"
    echo "  - LOGS_PATH                : Path to store run logs"
    echo "  - RESULT_DIR               : Path to store benchmark results"
    echo ""
    echo "Example: bash $0 --config ./my_benchmark_config.sh"
    echo ""
}

# Convert test case name to valid filename
get_log_filename() {
    local name="$1"
    echo "${name//[\/ ]/_}.log"
}

# Check if server process is still running
is_server_running() {
    if [ -z "$server_pid" ]; then
        return 1
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "Server process $server_pid is no longer running."
        return 1
    fi
    return 0
}

# Check server health and get error details
check_server_health() {
    if ! is_server_running; then
        echo "ERROR: Server process $server_pid is not running."
        if [ -f "$server_log_file" ]; then
            echo "Last 10 lines of server log:"
            tail -10 "$server_log_file"
            echo ""
            echo "Checking for error patterns in server log..."
            if grep -i "error\|failed\|exception\|traceback" "$server_log_file" | tail -5; then
                echo "Found errors in server log:"
                grep -i "error\|failed\|exception\|traceback" "$server_log_file" | tail -5
            fi
        fi
        return 1
    fi

    # Check if server is responding to HTTP requests
    if ! curl -s http://localhost:${VLLM_PORT}/v1/models > /dev/null; then
        echo "Server process is running but not responding on port $VLLM_PORT, continue waiting..."
        return 2
    fi

    return 0
}

# Wait until the server is ready
wait_for_server() {
    local retries=300
    local wait_time=10
    for ((i=0; i<retries; i++)); do
        check_server_health
        result=$?
        if [ "$result" -eq 2 ]; then
            echo "Waiting for server to be ready... (attempt $((i+1))/$retries)"
        fi

        if [ "$result" -eq 1 ]; then
            echo "Server failed to start!"
            return 1
        fi

        if [ "$result" -eq 0 ]; then
            echo "Server is ready!"
            return 0
        fi
        sleep $wait_time
    done
    echo "Server did not become ready in time."
    return 1
}



# make sure we clean up the vllm server
cleanup() {
    echo "Caught Ctrl+C. Killing background process with PID $server_pid..."
    kill "$server_pid"
    exit 1
}
trap cleanup SIGINT

# Generate summary report
generate_summary() {
    echo ""
    echo "=========================================="
    echo "BENCHMARK SUMMARY REPORT"
    echo "=========================================="
    echo "Total test cases: ${#test_cases[@]}"

    local success_count=0
    local failure_count=0

    for test_case in "${test_cases[@]}"; do
        if [ "${test_results[$test_case]}" == "SUCCESS" ]; then
            success_count=$((success_count + 1))
            elif [ "${test_results[$test_case]}" == "FAILED" ]; then
            failure_count=$((failure_count + 1))
        fi
    done

    echo "Successful test cases: $success_count"
    echo "Failed test cases: $failure_count"
    echo ""

    if [ $failure_count -gt 0 ]; then
        echo "FAILED TEST CASES:"
        for test_case in "${test_cases[@]}"; do
            if [ "${test_results[$test_case]}" == "FAILED" ]; then
                echo "  - $test_case"
            fi
        done
        echo ""
    fi

    if [ $success_count -gt 0 ]; then
        echo "SUCCESSFUL TEST CASES:"
        for test_case in "${test_cases[@]}"; do
            if [ "${test_results[$test_case]}" == "SUCCESS" ]; then
                echo "  - $test_case"
            fi
        done
    fi

    echo "=========================================="
}

# Global variable to control script behavior on server failure
STOP_ON_SERVER_FAILURE=false
# Track test case results (this will be populated during execution)
declare -A test_results
main() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --config)
                CONFIG_FILE="$2"
                shift 2
            ;;
            --stop-on-server-failure)
                STOP_ON_SERVER_FAILURE=true
                shift
            ;;
            --help)
                help
                exit 0
            ;;
            *)
                echo "ERROR: Unknown option: $1"
                help
                exit 1
            ;;
        esac
    done

    # Check if configuration file is provided
    if [ -z "$CONFIG_FILE" ]; then
        echo "ERROR: Configuration file is required!"
        echo ""
        help
        exit 1
    fi

    # Load the configuration
    load_config "$CONFIG_FILE"

    echo "Using configuration file: $CONFIG_FILE"
    echo "Loaded ${#test_cases[@]} test cases from configuration"
    echo "Server failure behavior: $([ "$STOP_ON_SERVER_FAILURE" = true ] && echo "STOP ON FAILURE" || echo "CONTINUE ON FAILURE")"

    mkdir -p ${LOGS_PATH}

    # Initialize a global benchmark log file using timestamp runtime format: benchmark_YYYYMMDD_HHMMSS.log
    benchmark_runtime="$(date +%Y%m%d_%H%M%S)"
    benchmark_log="${LOGS_PATH}/benchmark_${benchmark_runtime}.log"
    echo "Initializing benchmark log: $benchmark_log"
    # Create / truncate the benchmark log file so subsequent appends are safe
    : > "$benchmark_log"

    for test_case in "${test_cases[@]}"; do
        # Skip test cases marked as FAILED
        if [[ "$test_case" == *"FAILED"* ]]; then
            echo "Skipping failed test case: $test_case"
            continue
        fi

        res_dir="${RESULT_DIR}/${folder_name[$test_case]}"
        echo "Running online serving benchmark for: $test_case"
        log_file="${LOGS_PATH}/$(get_log_filename "$test_case")"
        server_log_file="${LOGS_PATH}/serverlog_$(get_log_filename "$test_case")"
        echo "Logging output to: $log_file"
        mkdir -p $res_dir
        echo "save results in: $res_dir"
        base_env="HIP_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES"
        test_env="${env_vars[$test_case]}"
        full_env="$base_env $test_env"

        echo "running using base_env $base_env"
        echo "running using test_env $test_env"
        echo "running using model_params ${model_params[$test_case]}"

        rm -rf /root/.cache/vllm

        # Initialize test case result as unknown
        test_results["$test_case"]="UNKNOWN"

        # Start vLLM server with different command structure for the new test case
        echo "Starting vLLM server..."
        eval "$full_env  vllm serve ${model_params[$test_case]} --trust-remote-code --distributed-executor-backend mp --swap-space 16 --disable-log-requests --port ${VLLM_PORT} >> $server_log_file &"
        server_pid=$!

        # Wait for server to be ready
        sleep 3
        if ! wait_for_server; then
            echo "Server failed to start. Skipping test case: $test_case" | tee -a "$log_file"
            test_results["$test_case"]="FAILED"
            if is_server_running; then
                kill $server_pid
            fi

            # Check if we should stop the entire script
            if [ "$STOP_ON_SERVER_FAILURE" = true ]; then
                echo "STOP ON FAILURE mode enabled. Aborting script." | tee -a "$log_file"
                generate_summary
                exit 1
            fi
            continue
        fi

        # Extract model name from model_params
        model_name=$(echo "${model_params[$test_case]}" | awk '{print $1}')

        # Loop 2: Random prompt benchmarks
        local benchmark_success=true
        for i in "${!in_len[@]}"; do
            _in_len=${in_len[i]}
            _out_len=${out_len[i]}

            # Check if server is still running before each benchmark
            if ! is_server_running; then
                echo "Server died before benchmark. Skipping test case: $test_case" | tee -a "$log_file"
                benchmark_success=false
                break

                # Check if we should stop the entire script
                if [ "$STOP_ON_SERVER_FAILURE" = true ]; then
                    echo "STOP ON FAILURE mode enabled. Aborting script." | tee -a "$log_file"
                    test_results["$test_case"]="FAILED"
                    generate_summary
                    exit 1
                fi
            fi

            echo "Running random benchmark: in_len=$_in_len, out_len=$_out_len" | tee -a "$log_file"

            # Run benchmark in background to monitor server during execution
            python3 /app/benchmarkaiter/llama_fp8_03122025_sync_upstream_20250812/benchmarks/benchmark_serving.py \
            --backend vllm \
            --model "$model_name" \
            --dataset-name random \
            --random-input-len "$_in_len" \
            --save-result \
            --result-filename "ISL-$_in_len-OSL-$_out_len.json" \
            --result-dir "$res_dir" \
            --port ${VLLM_PORT} \
            --max-concurrency 64 \
            --num-prompts 640 \
            --ignore-eos \
            --percentile_metrics ttft,tpot,itl,e2el \
            --random-output-len "$_out_len" >> "$log_file" 2>&1 &

            benchmark_pid=$!

            # Give benchmark time to start
            sleep 60

            # Monitor both the benchmark and server process
            while true; do
                # Check if benchmark completed
                if ! kill -0 "$benchmark_pid" 2>/dev/null; then
                    break
                fi

                # Check if server is still running
                if ! is_server_running; then
                    echo "Server died during benchmark (PID: $server_pid). Killing benchmark (PID: $benchmark_pid)..." | tee -a "$log_file"
                    kill $benchmark_pid 2>/dev/null
                    benchmark_success=false
                    break

                    # Check if we should stop the entire script
                    if [ "$STOP_ON_SERVER_FAILURE" = true ]; then
                        echo "STOP ON FAILURE mode enabled. Aborting script." | tee -a "$log_file"
                        test_results["$test_case"]="FAILED"
                        generate_summary
                        exit 1
                    fi
                fi

                sleep 5
            done

            wait $benchmark_pid 2>/dev/null
            sleep 10
        done

        # Determine test case result
        if [ "$benchmark_success" = true ] && is_server_running; then
            echo "Test case completed successfully: $test_case" | tee -a "$log_file"
            test_results["$test_case"]="SUCCESS"
        else
            echo "Test case failed: $test_case" | tee -a "$log_file"
            test_results["$test_case"]="FAILED"
        fi

        # After each test case, append an updated benchmark summary to the global benchmark log.
        # This uses the same output as generate_summary() so the file accumulates a running report.
        if [ -n "${benchmark_log:-}" ]; then
            {
                echo ""
                echo "=========================================="
                echo "PER-RUN BENCHMARK SUMMARY (after test case: $test_case)"
                echo "Runtime: $benchmark_runtime"
                echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
                echo "------------------------------------------"
                # Call generate_summary to print the current global summary and append it
                generate_summary
                echo "=========================================="
                echo ""
            } >> "$benchmark_log"
            echo "Appended updated summary to $benchmark_log"
        fi

        # Kill the server
        echo "Stopping vLLM server..."
        if is_server_running; then
            kill $server_pid
            # Wait for server to actually terminate
            local wait_count=0
            while is_server_running && [ $wait_count -lt 60 ]; do
                sleep 1
                wait_count=$((wait_count + 1))
            done

            if is_server_running; then
                echo "Server did not terminate gracefully, forcing kill..." | tee -a "$log_file"
                kill -9 $server_pid 2>/dev/null
            else
                echo "Server terminated gracefully." | tee -a "$log_file"
            fi
        else
            echo "Server process already died." | tee -a "$log_file"
        fi

        sleep 30
    done

    # Generate summary report
    generate_summary
}

main "$@"
