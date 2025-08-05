import argparse
import time
import torch
from typing import List, Dict, Any
import json
from datetime import datetime

from kernels import (
    mrope_forward_native, 
    mrope_forward_liger_kernel,
    mrope_forward_liger_kernel_adapted_to_vllm_input
)
from utils import (
    DEVICE, MODEL_TP_DICT, DTYPE_LIST,
    prepare_test_inputs, get_common_args, validate_inputs, print_config_info
)

# Benchmark-specific constants
WARMUP_ITER = 5
BENCHMARK_ITER = 100
NUM_TOKENS_LIST = [2**i for i in range(0, 18)]

def benchmark_function(func, *args, warmup_iter=WARMUP_ITER, benchmark_iter=BENCHMARK_ITER) -> float:
    """Benchmark a function with warmup."""
    # Warm up
    for _ in range(warmup_iter):
        func(*args)
    
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(benchmark_iter):
        func(*args)
    torch.cuda.synchronize()
    
    return (time.time() - start_time) / benchmark_iter

def benchmark_all_implementations(
    model_config: Dict[str, Any],
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Dict[str, float]:
    """Benchmark all MRoPE implementations."""
    args = get_common_args(positions, query, key, cos, sin, model_config)
    
    # Benchmark implementations
    ref_time = benchmark_function(
        lambda *a: mrope_forward_native(*a),
        *args
    )
    
    liger_time = benchmark_function(
        lambda *a: mrope_forward_liger_kernel(*a),
        *args
    )
    
    vllm_adapted_time = benchmark_function(
        lambda *a: mrope_forward_liger_kernel_adapted_to_vllm_input(*a),
        *args
    )
    
    return {
        'ref_time': ref_time,
        'liger_time': liger_time,
        'vllm_adapted_time': vllm_adapted_time
    }

def run_single_benchmark(
    model_name: str, 
    tp_size: int, 
    dtype: torch.dtype, 
    num_tokens: int,
    verbose: bool = True
) -> Dict[str, Any]:
    """Run benchmark for a single configuration."""
    if verbose:
        print(f"\nBenchmarking:")
        print_config_info(model_name, tp_size, dtype, num_tokens, "  ")
    
    # Validate inputs
    validate_inputs(model_name, tp_size, num_tokens)
    
    # Prepare test inputs
    model_config, positions, query, key, cos, sin = prepare_test_inputs(
        model_name, tp_size, dtype, num_tokens
    )
    
    # Run benchmarks
    timing_results = benchmark_all_implementations(
        model_config, positions, query, key, cos, sin
    )
    
    # Calculate speedups
    ref_time = timing_results['ref_time']
    liger_speedup = ref_time / timing_results['liger_time'] if timing_results['liger_time'] > 0 else 0
    vllm_speedup = ref_time / timing_results['vllm_adapted_time'] if timing_results['vllm_adapted_time'] > 0 else 0
    
    results = {
        'model_name': model_name,
        'tp_size': tp_size,
        'dtype': str(dtype),
        'num_tokens': num_tokens,
        'num_heads': model_config['num_heads'],
        'num_kv_heads': model_config['num_kv_heads'],
        'head_dim': model_config['head_dim'],
        'liger_speedup': liger_speedup,
        'vllm_speedup': vllm_speedup,
        **timing_results
    }
    
    if verbose:
        print_benchmark_results(results)
    
    return results

def print_benchmark_results(results: Dict[str, Any]):
    """Print benchmark results for a single configuration."""
    print(f"Performance comparison for config ({results['num_tokens']}, {results['num_heads']}, {results['num_kv_heads']}):")
    print(f"  Reference implementation: {results['ref_time']:.8f}s")
    print(f"  Liger Kernel implementation: {results['liger_time']:.8f}s")
    print(f"  Liger Kernel Adapted to vLLM Input implementation: {results['vllm_adapted_time']:.8f}s")
    print(f"  Liger Kernel Speedup over vLLM native: {results['liger_speedup']:.4f}x")
    print(f"  Liger Kernel Adapted to vLLM Input Speedup over vLLM native: {results['vllm_speedup']:.4f}x")

def run_comprehensive_benchmark(
    models: List[str] = None,
    token_sizes: List[int] = None,
    output_file: str = None,
    verbose: bool = True
) -> List[Dict[str, Any]]:
    """Run comprehensive benchmark across all configurations."""
    if models is None:
        models = list(MODEL_TP_DICT.keys())
    
    if token_sizes is None:
        token_sizes = NUM_TOKENS_LIST
    
    all_results = []
    total_configs = sum(len(MODEL_TP_DICT.get(model, [])) for model in models) * len(DTYPE_LIST) * len(token_sizes)
    current_config = 0
    
    for model_name in models:
        if model_name not in MODEL_TP_DICT:
            print(f"Warning: Model {model_name} not found in MODEL_TP_DICT")
            continue
            
        for tp_size in MODEL_TP_DICT[model_name]:
            for dtype in DTYPE_LIST:
                for num_tokens in token_sizes:
                    current_config += 1
                    if verbose:
                        print(f"\nProgress: {current_config}/{total_configs}")
                        print("=" * 80)
                    
                    try:
                        result = run_single_benchmark(
                            model_name, tp_size, dtype, num_tokens, verbose
                        )
                        all_results.append(result)
                    except Exception as e:
                        print(f"Error benchmarking {model_name} tp={tp_size} tokens={num_tokens}: {e}")
                        continue
    
    # Save results if output file specified
    if output_file:
        save_results(all_results, output_file)
    
    return all_results

def save_results(results: List[Dict[str, Any]], filename: str):
    """Save benchmark results to JSON file."""
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'device': str(DEVICE),
        'warmup_iter': WARMUP_ITER,
        'benchmark_iter': BENCHMARK_ITER,
        'results': results
    }
    
    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {filename}")

def analyze_results(results: List[Dict[str, Any]]):
    """Analyze and print summary of benchmark results."""
    if not results:
        print("No results to analyze")
        return
    
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    
    # Overall statistics
    liger_speedups = [r['liger_speedup'] for r in results if r['liger_speedup'] > 0]
    vllm_speedups = [r['vllm_speedup'] for r in results if r['vllm_speedup'] > 0]
    
    if liger_speedups:
        print(f"Liger Kernel Speedup - Mean: {sum(liger_speedups)/len(liger_speedups):.4f}x, "
              f"Max: {max(liger_speedups):.4f}x, Min: {min(liger_speedups):.4f}x")
    
    if vllm_speedups:
        print(f"vLLM Adapted Speedup - Mean: {sum(vllm_speedups)/len(vllm_speedups):.4f}x, "
              f"Max: {max(vllm_speedups):.4f}x, Min: {min(vllm_speedups):.4f}x")
    
    # Best performing configurations
    best_liger = max(results, key=lambda x: x['liger_speedup'])
    best_vllm = max(results, key=lambda x: x['vllm_speedup'])
    
    print(f"\nBest Liger Kernel Performance:")
    print(f"  Model: {best_liger['model_name']}, TP: {best_liger['tp_size']}, "
          f"Tokens: {best_liger['num_tokens']}, Speedup: {best_liger['liger_speedup']:.4f}x")
    
    print(f"\nBest vLLM Adapted Performance:")
    print(f"  Model: {best_vllm['model_name']}, TP: {best_vllm['tp_size']}, "
          f"Tokens: {best_vllm['num_tokens']}, Speedup: {best_vllm['vllm_speedup']:.4f}x")

def main():
    """Main function for command-line interface."""
    # Update global constants
    global WARMUP_ITER, BENCHMARK_ITER
    parser = argparse.ArgumentParser(description="MRoPE Benchmark Suite")
    parser.add_argument("--models", nargs="+", help="Models to benchmark", 
                       choices=list(MODEL_TP_DICT.keys()))
    parser.add_argument("--tokens", nargs="+", type=int, help="Token sizes to benchmark")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("--warmup", type=int, default=WARMUP_ITER, help="Warmup iterations")
    parser.add_argument("--benchmark", type=int, default=BENCHMARK_ITER, help="Benchmark iterations")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark with reduced configurations")
    
    args = parser.parse_args()
    
    WARMUP_ITER = args.warmup
    BENCHMARK_ITER = args.benchmark
    
    # Set up configurations
    models = args.models if args.models else list(MODEL_TP_DICT.keys())
    
    if args.quick:
        token_sizes = [256, 1024, 4096]
    else:
        token_sizes = args.tokens if args.tokens else NUM_TOKENS_LIST
    
    verbose = not args.quiet
    
    # Run benchmark
    print(f"Starting benchmark on device: {DEVICE}")
    print(f"Models: {models}")
    print(f"Token sizes: {token_sizes}")
    print(f"Warmup iterations: {WARMUP_ITER}")
    print(f"Benchmark iterations: {BENCHMARK_ITER}")
    
    results = run_comprehensive_benchmark(
        models=models,
        token_sizes=token_sizes,
        output_file=args.output,
        verbose=verbose
    )
    
    # Analyze results
    analyze_results(results)

if __name__ == "__main__":
    main()