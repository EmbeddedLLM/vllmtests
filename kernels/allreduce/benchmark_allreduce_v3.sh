#!/bin/bash
# run_benchmarks.sh

set -e

export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT8

echo "Running All-Reduce Benchmarks"

# Check if we have enough GPUs
NUM_GPUS=8
echo "Detected $NUM_GPUS GPUs"

# Run benchmarks for different world sizes
for WS in 2 4 8; do
    if [ $WS -le $NUM_GPUS ]; then
        echo "Running benchmark for world size $WS"
        torchrun --nproc_per_node=$WS benchmark_allreduce_v3.py --world-size $WS --output benchmark_results_v3.md  --mode cudagraph
    else
        echo "Skipping world size $WS (not enough GPUs)"
    fi
done

# Combine results into a single markdown file
echo "# All-Reduce Benchmark Results" > combined_results.md
echo "" >> combined_results.md
echo "Benchmark comparing AITER custom all-reduce, vLLM custom all-reduce, and vLLM quick all-reduce." >> combined_results.md
echo "" >> combined_results.md

for WS in 2 4 8; do
    if [ -f "benchmark_results_ws${WS}.md" ]; then
        cat "benchmark_results_ws${WS}.md" >> combined_results.md
        echo "" >> combined_results.md
    fi
done

echo "Combined results saved to combined_results.md"