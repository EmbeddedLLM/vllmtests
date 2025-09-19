#!/bin/bash
# run_benchmark.sh

# Set CUDA visible devices to use all 8 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Set environment variables for optimal performance
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # Disable InfiniBand if not available
export NCCL_P2P_DISABLE=0  # Enable P2P if available

# Run the benchmark
python3 benchmark_allreduce_multi_gpu.py \
    --tp-sizes 2 4 8 \
    --output-dir ./benchmark_results \
    --master-addr localhost \
    --master-port 12355

echo "Benchmarking completed!"
echo "Results are saved in ./benchmark_results/"
echo "- tp2_results.json: Results for TP2"
echo "- tp4_results.json: Results for TP4" 
echo "- tp8_results.json: Results for TP8"