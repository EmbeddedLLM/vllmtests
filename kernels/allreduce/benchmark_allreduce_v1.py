#!/usr/bin/env python3
"""
Benchmark script comparing AITER custom all-reduce, vLLM custom all-reduce, 
and vLLM quick all-reduce across different tensor parallel sizes and message sizes.
"""

import os
import sys
import time
import argparse
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from typing import List, Dict, Optional, Tuple
import numpy as np
from contextlib import contextmanager
from enum import Enum

# Test sizes in bytes
TEST_SIZES = [
    2 * 1024,           # 2KB
    32 * 1024,          # 32KB  
    256 * 1024,         # 256KB
    512 * 1024,         # 512KB
    1024 * 1024,        # 1MB
    2 * 1024 * 1024,    # 2MB
    4 * 1024 * 1024,    # 4MB
    8 * 1024 * 1024,    # 8MB
    16 * 1024 * 1024,   # 16MB
    32 * 1024 * 1024,   # 32MB
    64 * 1024 * 1024,   # 64MB
    128 * 1024 * 1024,  # 128MB
    256 * 1024 * 1024,  # 256MB
    512 * 1024 * 1024,  # 512MB
    1024 * 1024 * 1024, # 1GB
    2 * 1024 * 1024 * 1024, # 2GB
]

WORLD_SIZES = [2, 4, 8]
WARMUP_ITERS = 10
BENCHMARK_ITERS = 100

class QuickReduceRegime(Enum):
    FP = 0
    INT8 = 1
    INT6 = 2
    INT4 = 3

def format_size(size_bytes: int) -> str:
    """Format size in bytes to human readable format."""
    if size_bytes >= 1024**3:
        return f"{size_bytes // (1024**3)}GB"
    elif size_bytes >= 1024**2:
        return f"{size_bytes // (1024**2)}MB"
    elif size_bytes >= 1024:
        return f"{size_bytes // 1024}KB"
    else:
        return f"{size_bytes}B"

@contextmanager
def timer():
    """Context manager for timing operations."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    torch.cuda.synchronize()
    end = time.perf_counter()
    return end - start

class BenchmarkRunner:
    def __init__(self, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.results = {}
        
        # Initialize process groups
        self.init_process_groups()
        
        # Initialize all-reduce implementations
        self.init_implementations()
    
    def init_process_groups(self):
        """Initialize process groups for different backends."""
        # NCCL group for baseline
        self.nccl_group = dist.new_group(backend="nccl")
        
        # Gloo group for custom implementations
        self.gloo_group = dist.new_group(backend="gloo")
    
    def init_implementations(self):
        """Initialize all all-reduce implementations."""
        self.implementations = {}
        
        # Try to initialize AITER custom all-reduce
        try:
            from aiter.dist.custom_all_reduce import CustomAllreduce as AITERCustomAllreduce
            self.implementations['aiter'] = AITERCustomAllreduce(
                group=self.gloo_group,
                device=self.device,
                max_size=8192 * 1024 * 8
            )
            print(f"Rank {self.rank}: AITER custom all-reduce initialized")
        except Exception as e:
            print(f"Rank {self.rank}: Failed to initialize AITER: {e}")
            self.implementations['aiter'] = None
        
        # Try to initialize vLLM custom all-reduce
        try:
            from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce as VLLMCustomAllreduce
            self.implementations['vllm_custom'] = VLLMCustomAllreduce(
                group=self.gloo_group,
                device=self.device,
                max_size=8192 * 1024
            )
            print(f"Rank {self.rank}: vLLM custom all-reduce initialized")
        except Exception as e:
            print(f"Rank {self.rank}: Failed to initialize vLLM custom: {e}")
            self.implementations['vllm_custom'] = None
        
        # Try to initialize vLLM quick all-reduce
        try:
            from vllm.distributed.device_communicators.quick_all_reduce import QuickAllReduce
            self.implementations['vllm_quick'] = QuickAllReduce(
                group=self.gloo_group,
                device=self.device
            )
            print(f"Rank {self.rank}: vLLM quick all-reduce initialized")
        except Exception as e:
            print(f"Rank {self.rank}: Failed to initialize vLLM quick: {e}")
            self.implementations['vllm_quick'] = None
    
    def create_test_tensor(self, size_bytes: int, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """Create a test tensor of specified size."""
        element_size = torch.tensor([], dtype=dtype).element_size()
        num_elements = size_bytes // element_size
        
        # Ensure size is multiple of 16 for custom all-reduce requirements
        if num_elements * element_size % 16 != 0:
            num_elements = ((num_elements * element_size + 15) // 16) * 16 // element_size
        
        tensor = torch.randn(num_elements, dtype=dtype, device=self.device)
        return tensor
    
    def benchmark_nccl(self, tensor: torch.Tensor) -> float:
        """Benchmark NCCL all-reduce."""
        # Warmup
        for _ in range(WARMUP_ITERS):
            test_tensor = tensor.clone()
            dist.all_reduce(test_tensor, group=self.nccl_group)
        
        # Benchmark
        times = []
        for _ in range(BENCHMARK_ITERS):
            test_tensor = tensor.clone()
            torch.cuda.synchronize()
            start = time.perf_counter()
            dist.all_reduce(test_tensor, group=self.nccl_group)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
        
        return np.mean(times) * 1000  # Convert to milliseconds
    
    def benchmark_aiter(self, tensor: torch.Tensor) -> Dict[str, float]:
        """Benchmark AITER custom all-reduce."""
        if self.implementations['aiter'] is None or self.implementations['aiter'].disabled:
            return {'all_reduce_reg': -1, 'all_reduce_unreg': -1}
        
        impl = self.implementations['aiter']
        results = {}
        
        # # Test registered version
        # if impl.should_custom_ar(tensor):
        # Register buffer
        impl.register_buffer(tensor)
        
        # Warmup
        for _ in range(WARMUP_ITERS):
            _ = impl.all_reduce_reg(tensor)
        
        # Benchmark
        times = []
        for _ in range(BENCHMARK_ITERS):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = impl.all_reduce_reg(tensor)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
        
        results['all_reduce_reg'] = np.mean(times) * 1000
        # else:
        #     results['all_reduce_reg'] = -1
        
        # Test unregistered version
        # if impl.should_custom_ar(tensor):
        # Warmup
        for _ in range(WARMUP_ITERS):
            _ = impl.all_reduce_unreg(tensor)
        
        # Benchmark
        times = []
        for _ in range(BENCHMARK_ITERS):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = impl.all_reduce_unreg(tensor)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
        
        results['all_reduce_unreg'] = np.mean(times) * 1000
        # else:
        #     results['all_reduce_unreg'] = -1
        
        return results
    
    def benchmark_vllm_custom(self, tensor: torch.Tensor) -> float:
        """Benchmark vLLM custom all-reduce."""
        if self.implementations['vllm_custom'] is None or self.implementations['vllm_custom'].disabled:
            return -1
        
        impl = self.implementations['vllm_custom']
        
        # if not impl.should_custom_ar(tensor):
        #     return -1
        
        # Warmup
        for _ in range(WARMUP_ITERS):
            _ = impl.custom_all_reduce(tensor)
        
        # Benchmark
        times = []
        for _ in range(BENCHMARK_ITERS):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = impl.custom_all_reduce(tensor)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
        
        return np.mean(times) * 1000
    
    def benchmark_vllm_quick(self, tensor: torch.Tensor) -> Dict[str, float]:
        """Benchmark vLLM quick all-reduce with different quantization levels."""
        if self.implementations['vllm_quick'] is None or self.implementations['vllm_quick'].disabled:
            return {f'quick_{regime.name}': -1 for regime in QuickReduceRegime}
        
        impl = self.implementations['vllm_quick']
        results = {}
        
        for regime in QuickReduceRegime:
            # Set quantization level
            impl.qr_quant_level = regime
            
            # if not impl.should_quick_allreduce(tensor):
            #     results[f'quick_{regime.name}'] = -1
            #     continue
            
            # Warmup
            for _ in range(WARMUP_ITERS):
                _ = impl.quick_all_reduce(tensor)
            
            # Benchmark
            times = []
            for _ in range(BENCHMARK_ITERS):
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = impl.quick_all_reduce(tensor)
                torch.cuda.synchronize()
                end = time.perf_counter()
                times.append(end - start)
            
            results[f'quick_{regime.name}'] = np.mean(times) * 1000
        
        return results
    
    def run_benchmark(self):
        """Run the complete benchmark suite."""
        print(f"Rank {self.rank}: Starting benchmark for world size {self.world_size}")
        
        for size_bytes in TEST_SIZES:
            if self.rank == 0:
                print(f"Testing size: {format_size(size_bytes)}")
            
            # Create test tensor
            tensor = self.create_test_tensor(size_bytes)
            
            # Store results for this size
            size_key = format_size(size_bytes)
            self.results[size_key] = {}
            
            # Benchmark NCCL
            nccl_time = self.benchmark_nccl(tensor)
            self.results[size_key]['nccl'] = nccl_time
            
            # Benchmark AITER
            aiter_results = self.benchmark_aiter(tensor)
            self.results[size_key].update(aiter_results)
            
            # Benchmark vLLM custom
            vllm_custom_time = self.benchmark_vllm_custom(tensor)
            self.results[size_key]['vllm_custom'] = vllm_custom_time
            
            # Benchmark vLLM quick
            vllm_quick_results = self.benchmark_vllm_quick(tensor)
            self.results[size_key].update(vllm_quick_results)
            
            # Synchronize all processes
            dist.barrier()
    
    def save_results(self, output_file: str):
        """Save results to markdown file."""
        if self.rank != 0:
            return
        
        with open(output_file, 'w') as f:
            f.write(f"# All-Reduce Benchmark Results (World Size: {self.world_size})\n\n")
            f.write("All times are in milliseconds. -1 indicates the method is not supported/available for that size.\n\n")
            
            # Create table header
            headers = ['Size', 'NCCL', 'AITER Reg', 'AITER Unreg', 'vLLM Custom', 
                      'Quick FP', 'Quick INT8', 'Quick INT6', 'Quick INT4']
            f.write('| ' + ' | '.join(headers) + ' |\n')
            f.write('|' + '|'.join(['---'] * len(headers)) + '|\n')
            
            # Write results
            for size_key in self.results:
                row = [size_key]
                result = self.results[size_key]
                
                # Format values
                row.append(f"{result.get('nccl', -1):.3f}" if result.get('nccl', -1) > 0 else "N/A")
                row.append(f"{result.get('all_reduce_reg', -1):.3f}" if result.get('all_reduce_reg', -1) > 0 else "N/A")
                row.append(f"{result.get('all_reduce_unreg', -1):.3f}" if result.get('all_reduce_unreg', -1) > 0 else "N/A")
                row.append(f"{result.get('vllm_custom', -1):.3f}" if result.get('vllm_custom', -1) > 0 else "N/A")
                row.append(f"{result.get('quick_FP', -1):.3f}" if result.get('quick_FP', -1) > 0 else "N/A")
                row.append(f"{result.get('quick_INT8', -1):.3f}" if result.get('quick_INT8', -1) > 0 else "N/A")
                row.append(f"{result.get('quick_INT6', -1):.3f}" if result.get('quick_INT6', -1) > 0 else "N/A")
                row.append(f"{result.get('quick_INT4', -1):.3f}" if result.get('quick_INT4', -1) > 0 else "N/A")
                
                f.write('| ' + ' | '.join(row) + ' |\n')
            
            f.write(f"\n## Performance Summary\n\n")
            f.write("### Speedup vs NCCL\n\n")
            
            # Calculate speedups
            speedup_headers = ['Size', 'AITER Reg', 'AITER Unreg', 'vLLM Custom', 
                              'Quick FP', 'Quick INT8', 'Quick INT6', 'Quick INT4']
            f.write('| ' + ' | '.join(speedup_headers) + ' |\n')
            f.write('|' + '|'.join(['---'] * len(speedup_headers)) + '|\n')
            
            for size_key in self.results:
                result = self.results[size_key]
                nccl_time = result.get('nccl', -1)
                
                if nccl_time <= 0:
                    continue
                
                row = [size_key]
                
                for method in ['all_reduce_reg', 'all_reduce_unreg', 'vllm_custom', 
                              'quick_FP', 'quick_INT8', 'quick_INT6', 'quick_INT4']:
                    method_time = result.get(method, -1)
                    if method_time > 0:
                        speedup = nccl_time / method_time
                        row.append(f"{speedup:.2f}x")
                    else:
                        row.append("N/A")
                
                f.write('| ' + ' | '.join(row) + ' |\n')

def main():
    parser = argparse.ArgumentParser(description='Benchmark all-reduce implementations')
    parser.add_argument('--world-size', type=int, required=True, choices=WORLD_SIZES,
                       help='World size for distributed training')
    parser.add_argument('--output', type=str, default='benchmark_results.md',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Initialize distributed
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    assert world_size == args.world_size, f"Expected world size {args.world_size}, got {world_size}"
    
    # Set device
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    # Run benchmark
    runner = BenchmarkRunner(rank, world_size, device)
    runner.run_benchmark()
    
    # Save results
    output_file = f"{args.output.split('.')[0]}_ws{world_size}.md"
    runner.save_results(output_file)
    
    if rank == 0:
        print(f"Benchmark completed. Results saved to {output_file}")
    
    # Cleanup
    for impl in runner.implementations.values():
        if impl is not None and hasattr(impl, 'close'):
            impl.close()
    
    dist.destroy_process_group()

if __name__ == "__main__":
    main()