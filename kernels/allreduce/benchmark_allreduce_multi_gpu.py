#!/usr/bin/env python3
"""
Multi-GPU All-Reduce Benchmarking Script for TP2, TP4, and TP8
Supports testing different tensor parallelism configurations on a single server.
"""

import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
import json
from typing import List, Dict, Optional, Tuple
import numpy as np
from contextlib import contextmanager

# Set environment variables for Quick AllReduce configurations
os.environ['VLLM_ROCM_QUICK_REDUCE_QUANTIZATION'] = 'FP'
os.environ['VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16'] = '1'
os.environ['VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB'] = '2048'

# Import the communicator classes
try:
    from aiter.dist.custom_all_reduce import CustomAllreduce
    from vllm.distributed.device_communicators.quick_all_reduce import QuickAllReduce, QuickReduceRegime
    custom_ar_available = True
except ImportError as e:
    print(f"Warning: Could not import custom all-reduce modules: {e}")
    custom_ar_available = False

class MultiGPUAllReduceBenchmark:
    def __init__(self, rank: int, world_size: int, master_addr: str = "localhost", master_port: str = "12355"):
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{rank}")
        
        # Set CUDA device
        torch.cuda.set_device(self.device)
        
        # Initialize distributed process group
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        
        # Initialize with NCCL for device group and GLOO for CPU operations
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        
        # Create a separate GLOO group for CPU operations (required by custom allreduce)
        self.cpu_group = dist.new_group(backend='gloo')
        
        # Initialize communicators
        self.custom_ar = None
        self.quick_ar = None
        
        if custom_ar_available:
            try:
                self.custom_ar = CustomAllreduce(
                    group=self.cpu_group,
                    device=self.device
                )
                if self.rank == 0:
                    print(f"Custom AllReduce initialized: disabled={self.custom_ar.disabled}")
            except Exception as e:
                if self.rank == 0:
                    print(f"Failed to initialize Custom AllReduce: {e}")
            
            try:
                self.quick_ar = QuickAllReduce(
                    group=self.cpu_group,
                    device=self.device
                )
                if self.rank == 0:
                    print(f"Quick AllReduce initialized: disabled={self.quick_ar.disabled}")
            except Exception as e:
                if self.rank == 0:
                    print(f"Failed to initialize Quick AllReduce: {e}")

    @contextmanager
    def set_quick_reduce_regime(self, regime: QuickReduceRegime):
        """Context manager to temporarily set Quick AllReduce quantization regime"""
        if self.quick_ar and not self.quick_ar.disabled:
            old_regime = getattr(self.quick_ar, 'qr_quant_level', None)
            self.quick_ar.qr_quant_level = regime
            try:
                yield
            finally:
                if old_regime is not None:
                    self.quick_ar.qr_quant_level = old_regime
        else:
            yield

    def create_tensor(self, size_bytes: int, dtype: torch.dtype) -> torch.Tensor:
        """Create a tensor of specified size in bytes"""
        element_size = torch.tensor([], dtype=dtype).element_size()
        num_elements = size_bytes // element_size
        
        # Ensure the tensor size is multiple of 16 bytes for custom all-reduce
        if size_bytes % 16 != 0:
            num_elements = (size_bytes + 15) // 16 * 16 // element_size
        
        tensor = torch.randn(num_elements, dtype=dtype, device=self.device)
        return tensor

    def benchmark_method(self, tensor: torch.Tensor, method: str, 
                        regime: Optional[QuickReduceRegime] = None, 
                        warmup_runs: int = 10, benchmark_runs: int = 50) -> float:
        """Benchmark a specific all-reduce method"""
        
        # Synchronize all processes before starting
        dist.barrier()
        
        # Warmup
        for _ in range(warmup_runs):
            if method == "custom_ar" and self.custom_ar and not self.custom_ar.disabled:
                result = self.custom_ar.custom_all_reduce(tensor.clone())
                if result is None:
                    # Fall back to NCCL
                    tensor_copy = tensor.clone()
                    dist.all_reduce(tensor_copy)
            elif method == "quick_ar" and self.quick_ar and not self.quick_ar.disabled:
                with self.set_quick_reduce_regime(regime):
                    if regime and self.quick_ar.should_quick_allreduce(tensor):
                        _ = self.quick_ar.quick_all_reduce(tensor.clone())
                    else:
                        # Fall back to NCCL
                        tensor_copy = tensor.clone()
                        dist.all_reduce(tensor_copy)
            elif method == "nccl_baseline":
                tensor_copy = tensor.clone()
                dist.all_reduce(tensor_copy)
            elif method == "gloo_baseline":
                tensor_copy = tensor.clone()
                dist.all_reduce(tensor_copy, group=self.cpu_group)
        
        # Synchronize before timing
        dist.barrier()
        torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.perf_counter()
        
        for _ in range(benchmark_runs):
            if method == "custom_ar" and self.custom_ar and not self.custom_ar.disabled:
                result = self.custom_ar.custom_all_reduce(tensor.clone())
                if result is None:
                    tensor_copy = tensor.clone()
                    dist.all_reduce(tensor_copy)
            elif method == "quick_ar" and self.quick_ar and not self.quick_ar.disabled:
                with self.set_quick_reduce_regime(regime):
                    if regime and self.quick_ar.should_quick_allreduce(tensor):
                        _ = self.quick_ar.quick_all_reduce(tensor.clone())
                    else:
                        tensor_copy = tensor.clone()
                        dist.all_reduce(tensor_copy)
            elif method == "nccl_baseline":
                tensor_copy = tensor.clone()
                dist.all_reduce(tensor_copy)
            elif method == "gloo_baseline":
                tensor_copy = tensor.clone()
                dist.all_reduce(tensor_copy, group=self.cpu_group)
        
        torch.cuda.synchronize()
        dist.barrier()
        end_time = time.perf_counter()
        
        avg_time_ms = (end_time - start_time) * 1000 / benchmark_runs
        return avg_time_ms

    def run_benchmark_suite(self, dtypes: List[torch.dtype] = None) -> Dict:
        """Run comprehensive benchmark suite"""
        if dtypes is None:
            dtypes = [torch.float16, torch.bfloat16]
        
        # Test sizes from 2KB to 2GB
        test_sizes = [
            ("2.0KB", 2 * 1024),
            ("32.0KB", 32 * 1024),
            ("256.0KB", 256 * 1024),
            ("512.0KB", 512 * 1024),
            ("1.0MB", 1024 * 1024),
            ("2.0MB", 2 * 1024 * 1024),
            ("4.0MB", 4 * 1024 * 1024),
            ("8.0MB", 8 * 1024 * 1024),
            ("16.0MB", 16 * 1024 * 1024),
            ("32.0MB", 32 * 1024 * 1024),
            ("64.0MB", 64 * 1024 * 1024),
            ("128.0MB", 128 * 1024 * 1024),
            ("256.0MB", 256 * 1024 * 1024),
            ("512.0MB", 512 * 1024 * 1024),
            ("1GB", 1024 * 1024 * 1024),
            ("2GB", 2 * 1024 * 1024 * 1024),
        ]
        
        results = {}
        
        for dtype in dtypes:
            dtype_name = "fp16" if dtype == torch.float16 else "bf16"
            results[dtype_name] = {}
            
            if self.rank == 0:
                print(f"\nBenchmarking {dtype_name} on TP{self.world_size}...")
            
            for size_name, size_bytes in test_sizes:
                if self.rank == 0:
                    print(f"  Testing {size_name}...")
                
                try:
                    tensor = self.create_tensor(size_bytes, dtype)
                    
                    # Benchmark baseline (use NCCL for < 16MB, keep NCCL for >= 16MB)
                    baseline_time = self.benchmark_method(tensor, "nccl_baseline")
                    
                    # Benchmark Custom AllReduce
                    custom_ar_time = None
                    if self.custom_ar and not self.custom_ar.disabled:
                        custom_ar_time = self.benchmark_method(tensor, "custom_ar")
                    
                    # Benchmark Quick AllReduce with different quantization levels
                    qr_fp_time = None
                    qr_int8_time = None
                    qr_int6_time = None
                    qr_int4_time = None
                    
                    if self.quick_ar and not self.quick_ar.disabled:
                        qr_fp_time = self.benchmark_method(tensor, "quick_ar", QuickReduceRegime.FP)
                        qr_int8_time = self.benchmark_method(tensor, "quick_ar", QuickReduceRegime.INT8)
                        qr_int6_time = self.benchmark_method(tensor, "quick_ar", QuickReduceRegime.INT6)
                        qr_int4_time = self.benchmark_method(tensor, "quick_ar", QuickReduceRegime.INT4)
                    
                    results[dtype_name][size_name] = {
                        'baseline': baseline_time,
                        'custom_ar': custom_ar_time,
                        'qr_fp': qr_fp_time,
                        'qr_int8': qr_int8_time,
                        'qr_int6': qr_int6_time,
                        'qr_int4': qr_int4_time,
                    }
                    
                except Exception as e:
                    if self.rank == 0:
                        print(f"    Error testing {size_name}: {e}")
                    results[dtype_name][size_name] = {
                        'baseline': None,
                        'custom_ar': None,
                        'qr_fp': None,
                        'qr_int8': None,
                        'qr_int6': None,
                        'qr_int4': None,
                    }
        
        return results

    def print_results_table(self, results: Dict):
        """Print results in table format similar to PR description"""
        if self.rank != 0:
            return
            
        print("\n" + "="*140)
        print(f"BENCHMARK RESULTS FOR TP{self.world_size} (Time in microseconds)")
        print("="*140)
        
        # Print header
        header = f"{'msg size':<10} {'baseline':<10} {'custom_ar':<10} {'QR FP':<10} {'QR int8':<10} {'QR int6':<10} {'QR int4':<10} {'QR FP bf16':<12} {'QR int8 bf16':<12} {'QR int6 bf16':<12} {'QR int4 bf16':<12}"
        print(header)
        print("-" * len(header))
        
        # Get all size names from fp16 results
        if 'fp16' in results and results['fp16']:
            size_names = list(results['fp16'].keys())
            
            for size_name in size_names:
                fp16_data = results.get('fp16', {}).get(size_name, {})
                bf16_data = results.get('bf16', {}).get(size_name, {})
                
                def format_time(time_ms):
                    if time_ms is None:
                        return "N/A"
                    return f"{time_ms:.2f}"
                
                row = f"{size_name:<10} "
                row += f"{format_time(fp16_data.get('baseline')):<10} "
                row += f"{format_time(fp16_data.get('custom_ar')):<10} "
                row += f"{format_time(fp16_data.get('qr_fp')):<10} "
                row += f"{format_time(fp16_data.get('qr_int8')):<10} "
                row += f"{format_time(fp16_data.get('qr_int6')):<10} "
                row += f"{format_time(fp16_data.get('qr_int4')):<10} "
                row += f"{format_time(bf16_data.get('qr_fp')):<12} "
                row += f"{format_time(bf16_data.get('qr_int8')):<12} "
                row += f"{format_time(bf16_data.get('qr_int6')):<12} "
                row += f"{format_time(bf16_data.get('qr_int4')):<12}"
                
                print(row)

    def cleanup(self):
        """Clean up resources"""
        if self.custom_ar:
            self.custom_ar.close()
        if self.quick_ar:
            self.quick_ar.close()
        dist.destroy_process_group()

def run_benchmark_process(rank: int, world_size: int, output_file: str = None):
    """Function to run benchmark in a single process"""
    try:
        # Initialize benchmark
        benchmark = MultiGPUAllReduceBenchmark(rank=rank, world_size=world_size)
        
        if rank == 0:
            print(f"Starting benchmark suite for TP{world_size}...")
        
        # Run benchmark suite
        results = benchmark.run_benchmark_suite()
        
        # Print and save results (only rank 0)
        if rank == 0:
            benchmark.print_results_table(results)
            
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\nResults saved to {output_file}")
        
        # Cleanup
        benchmark.cleanup()
        
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description='Multi-GPU All-Reduce Benchmark')
    parser.add_argument('--tp-sizes', nargs='+', type=int, default=[2, 4, 8], 
                       help='Tensor parallelism sizes to test (default: 2 4 8)')
    parser.add_argument('--output-dir', type=str, default='./benchmark_results', 
                       help='Directory to save results')
    parser.add_argument('--master-addr', type=str, default='localhost', 
                       help='Master address for distributed training')
    parser.add_argument('--master-port', type=str, default='12355', 
                       help='Master port for distributed training')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Test each TP size
    for tp_size in args.tp_sizes:
        if tp_size > 8:
            print(f"Warning: TP{tp_size} requires more than 8 GPUs, skipping...")
            continue
            
        print(f"\n{'='*60}")
        print(f"TESTING TP{tp_size}")
        print(f"{'='*60}")
        
        output_file = os.path.join(args.output_dir, f'tp{tp_size}_results.json')
        
        # Set environment variables for this run
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = str(int(args.master_port) + tp_size)  # Different port for each TP size
        
        try:
            # Spawn processes for multi-GPU testing
            mp.spawn(
                run_benchmark_process,
                args=(tp_size, output_file),
                nprocs=tp_size,
                join=True
            )
        except Exception as e:
            print(f"Error running TP{tp_size} benchmark: {e}")
            continue
    
    print(f"\nAll benchmarks completed. Results saved in {args.output_dir}/")

if __name__ == "__main__":
    main()