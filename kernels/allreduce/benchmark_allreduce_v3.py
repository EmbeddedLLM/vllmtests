#!/usr/bin/env python3
"""
Benchmark script comparing AITER custom all-reduce, vLLM custom all-reduce, 
and vLLM quick all-reduce across different tensor parallel sizes and message sizes.
Supports both Eager and CudaGraph execution modes.
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
import multiprocessing
import random

# Import vLLM components
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import (get_tensor_model_parallel_group,
                                             get_tp_group, graph_capture, ensure_model_parallel_initialized, init_distributed_environment)

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

class ExecutionMode(Enum):
    EAGER = "eager"
    CUDAGRAPH = "cudagraph"

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
    def __init__(self, rank: int, world_size: int, device: torch.device, mode: ExecutionMode):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.mode = mode
        self.results = {}
        
        # Initialize vLLM distributed environment
        self.init_vllm_environment()
        
        # Initialize implementations
        self.init_implementations()
    
    def init_vllm_environment(self):
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "49373"
        """Initialize vLLM distributed environment."""
        # Initialize test distributed environment
        init_distributed_environment(
            world_size=self.world_size,
            rank=self.rank,
            backend="cpu:gloo,cuda:nccl",
            local_rank=self.rank
        )
        ensure_model_parallel_initialized(self.world_size, 1)
        
        # Get groups
        self.tp_group = get_tp_group()
        self.nccl_group = get_tensor_model_parallel_group()
    
    def init_implementations(self):
        """Initialize all all-reduce implementations."""
        self.implementations = {}
        
        # Try to initialize AITER custom all-reduce
        try:
            from aiter.dist.custom_all_reduce import CustomAllreduce as AITERCustomAllreduce
            max_size = max(TEST_SIZES)
            self.implementations['aiter'] = AITERCustomAllreduce(
                group=self.nccl_group,
                device=self.device,
                max_size=max_size
            )
            print(f"Rank {self.rank}: AITER custom all-reduce initialized")
        except Exception as e:
            print(f"Rank {self.rank}: Failed to initialize AITER: {e}")
            self.implementations['aiter'] = None
        
        # Try to initialize vLLM custom all-reduce
        try:
            from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce as VLLMCustomAllreduce
            max_size = max(TEST_SIZES)
            self.implementations['vllm_custom'] = VLLMCustomAllreduce(
                group=self.nccl_group,
                device=self.device,
                max_size=max_size
            )
            print(f"Rank {self.rank}: vLLM custom all-reduce initialized")
        except Exception as e:
            print(f"Rank {self.rank}: Failed to initialize vLLM custom: {e}")
            self.implementations['vllm_custom'] = None
        
        # vLLM quick all-reduce is accessed through tp_group
        if hasattr(self.tp_group, 'device_communicator') and hasattr(self.tp_group.device_communicator, 'qr_comm'):
            self.quick_comm = self.tp_group.device_communicator.qr_comm
            print(f"Rank {self.rank}: vLLM quick all-reduce initialized")
        else:
            self.quick_comm = None
            print(f"Rank {self.rank}: vLLM quick all-reduce not available")
    
    def create_test_tensor(self, size_bytes: int, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """Create a test tensor of specified size."""
        element_size = torch.tensor([], dtype=dtype).element_size()
        num_elements = size_bytes // element_size
        
        # Ensure size is multiple of 16 for custom all-reduce requirements
        if num_elements * element_size % 16 != 0:
            num_elements = ((num_elements * element_size + 15) // 16) * 16 // element_size
        
        tensor = torch.randn(num_elements, dtype=dtype, device=self.device)
        return tensor
    
    def benchmark_nccl_eager(self, tensor: torch.Tensor) -> float:
        """Benchmark NCCL all-reduce in eager mode."""
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
        
        return np.mean(times) * 1000
    
    def benchmark_nccl_cudagraph(self, tensor: torch.Tensor) -> float:
        """Benchmark NCCL all-reduce in CudaGraph mode."""
        try:
            with graph_capture(device=self.device) as graph_capture_context:
                test_tensor = tensor.clone()
                torch.cuda.synchronize()
                
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                    dist.all_reduce(test_tensor, group=self.nccl_group)
            
            # Warmup
            for _ in range(WARMUP_ITERS):
                graph.replay()
            
            # Benchmark
            times = []
            for _ in range(BENCHMARK_ITERS):
                torch.cuda.synchronize()
                start = time.perf_counter()
                graph.replay()
                torch.cuda.synchronize()
                end = time.perf_counter()
                times.append(end - start)
            
            return np.mean(times) * 1000
        except Exception as e:
            print(f"Rank {self.rank}: NCCL CudaGraph failed: {e}")
            return -1
    
    def benchmark_tensor_model_parallel_all_reduce_eager(self, tensor: torch.Tensor) -> float:
        """Benchmark tensor_model_parallel_all_reduce in eager mode."""
        try:
            # Warmup
            for _ in range(WARMUP_ITERS):
                test_tensor = tensor.clone()
                _ = tensor_model_parallel_all_reduce(test_tensor)
            
            # Benchmark
            times = []
            for _ in range(BENCHMARK_ITERS):
                test_tensor = tensor.clone()
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = tensor_model_parallel_all_reduce(test_tensor)
                torch.cuda.synchronize()
                end = time.perf_counter()
                times.append(end - start)
            
            return np.mean(times) * 1000
        except Exception as e:
            print(f"Rank {self.rank}: tensor_model_parallel_all_reduce eager failed: {e}")
            return -1
    
    def benchmark_tensor_model_parallel_all_reduce_cudagraph(self, tensor: torch.Tensor) -> float:
        """Benchmark tensor_model_parallel_all_reduce in CudaGraph mode."""
        try:
            with graph_capture(device=self.device) as graph_capture_context:
                test_tensor = tensor.clone()
                torch.cuda.synchronize()
                
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                    result = tensor_model_parallel_all_reduce(test_tensor)
            
            # Warmup
            for _ in range(WARMUP_ITERS):
                graph.replay()
            
            # Benchmark
            times = []
            for _ in range(BENCHMARK_ITERS):
                torch.cuda.synchronize()
                start = time.perf_counter()
                graph.replay()
                torch.cuda.synchronize()
                end = time.perf_counter()
                times.append(end - start)
            
            return np.mean(times) * 1000
        except Exception as e:
            print(f"Rank {self.rank}: tensor_model_parallel_all_reduce CudaGraph failed: {e}")
            return -1
    
    def benchmark_quick_all_reduce_eager(self, tensor: torch.Tensor) -> Dict[str, float]:
        """Benchmark quick all-reduce in eager mode."""
        if self.quick_comm is None:
            return {f'quick_{regime.name}': -1 for regime in QuickReduceRegime}
        
        results = {}
        original_quant_level = getattr(self.quick_comm, 'qr_quant_level', None)
        
        for regime in QuickReduceRegime:
            try:
                # Set quantization level
                if hasattr(self.quick_comm, 'qr_quant_level'):
                    self.quick_comm.qr_quant_level = regime
                
                # Warmup
                for _ in range(WARMUP_ITERS):
                    test_tensor = tensor.clone()
                    _ = self.quick_comm.quick_all_reduce(test_tensor)
                
                # Benchmark
                times = []
                for _ in range(BENCHMARK_ITERS):
                    test_tensor = tensor.clone()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    _ = self.quick_comm.quick_all_reduce(test_tensor)
                    torch.cuda.synchronize()
                    end = time.perf_counter()
                    times.append(end - start)
                
                results[f'quick_{regime.name}'] = np.mean(times) * 1000
            except Exception as e:
                print(f"Rank {self.rank}: Quick all-reduce {regime.name} eager failed: {e}")
                results[f'quick_{regime.name}'] = -1
        
        # Restore original quantization level
        if original_quant_level is not None and hasattr(self.quick_comm, 'qr_quant_level'):
            self.quick_comm.qr_quant_level = original_quant_level
        
        return results
    
    def benchmark_quick_all_reduce_cudagraph(self, tensor: torch.Tensor) -> Dict[str, float]:
        """Benchmark quick all-reduce in CudaGraph mode."""
        if self.quick_comm is None:
            return {f'quick_{regime.name}': -1 for regime in QuickReduceRegime}
        
        results = {}
        original_quant_level = getattr(self.quick_comm, 'qr_quant_level', None)
        
        for regime in QuickReduceRegime:
            try:
                # Set quantization level
                if hasattr(self.quick_comm, 'qr_quant_level'):
                    self.quick_comm.qr_quant_level = regime
                
                with graph_capture(device=self.device) as graph_capture_context:
                    test_tensor = tensor.clone()
                    torch.cuda.synchronize()
                    
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=graph_capture_context.stream):
                        result = self.quick_comm.quick_all_reduce(test_tensor)
                
                # Warmup
                for _ in range(WARMUP_ITERS):
                    graph.replay()
                
                # Benchmark
                times = []
                for _ in range(BENCHMARK_ITERS):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    graph.replay()
                    torch.cuda.synchronize()
                    end = time.perf_counter()
                    times.append(end - start)
                
                results[f'quick_{regime.name}'] = np.mean(times) * 1000
            except Exception as e:
                print(f"Rank {self.rank}: Quick all-reduce {regime.name} CudaGraph failed: {e}")
                results[f'quick_{regime.name}'] = -1
        
        # Restore original quantization level
        if original_quant_level is not None and hasattr(self.quick_comm, 'qr_quant_level'):
            self.quick_comm.qr_quant_level = original_quant_level
        
        return results
    
    def run_benchmark(self):
        """Run the complete benchmark suite."""
        print(f"Rank {self.rank}: Starting benchmark for world size {self.world_size} in {self.mode.value} mode")
        
        for size_bytes in TEST_SIZES:
            if self.rank == 0:
                print(f"Testing size: {format_size(size_bytes)}")
            
            # Create test tensor
            tensor = self.create_test_tensor(size_bytes)
            
            # Store results for this size
            size_key = format_size(size_bytes)
            self.results[size_key] = {}
            
            # Benchmark NCCL
            if self.mode == ExecutionMode.EAGER:
                nccl_time = self.benchmark_nccl_eager(tensor)
            else:
                nccl_time = self.benchmark_nccl_cudagraph(tensor)
            self.results[size_key]['nccl'] = nccl_time
            
            # Benchmark tensor_model_parallel_all_reduce
            if self.mode == ExecutionMode.EAGER:
                tmp_all_reduce_time = self.benchmark_tensor_model_parallel_all_reduce_eager(tensor)
            else:
                tmp_all_reduce_time = self.benchmark_tensor_model_parallel_all_reduce_cudagraph(tensor)
            self.results[size_key]['tensor_mp_all_reduce'] = tmp_all_reduce_time
            
            # Benchmark quick all-reduce
            if self.mode == ExecutionMode.EAGER:
                quick_results = self.benchmark_quick_all_reduce_eager(tensor)
            else:
                quick_results = self.benchmark_quick_all_reduce_cudagraph(tensor)
            self.results[size_key].update(quick_results)
            
            # Synchronize all processes
            dist.barrier()
    
    def save_results(self, output_file: str):
        """Save results to markdown file."""
        if self.rank != 0:
            return
        
        with open(output_file, 'w') as f:
            f.write(f"# All-Reduce Benchmark Results (World Size: {self.world_size}, Mode: {self.mode.value})\n\n")
            f.write("All times are in milliseconds. -1 indicates the method is not supported/available for that size.\n\n")
            
            # Create table header
            headers = ['Size', 'NCCL', 'Tensor MP All-Reduce', 
                      'Quick FP', 'Quick INT8', 'Quick INT6', 'Quick INT4']
            f.write('| ' + ' | '.join(headers) + ' |\n')
            f.write('|' + '|'.join(['---'] * len(headers)) + '|\n')
            
            # Write results
            for size_key in self.results:
                row = [size_key]
                result = self.results[size_key]
                
                # Format values
                row.append(f"{result.get('nccl', -1):.3f}" if result.get('nccl', -1) > 0 else "N/A")
                row.append(f"{result.get('tensor_mp_all_reduce', -1):.3f}" if result.get('tensor_mp_all_reduce', -1) > 0 else "N/A")
                row.append(f"{result.get('quick_FP', -1):.3f}" if result.get('quick_FP', -1) > 0 else "N/A")
                row.append(f"{result.get('quick_INT8', -1):.3f}" if result.get('quick_INT8', -1) > 0 else "N/A")
                row.append(f"{result.get('quick_INT6', -1):.3f}" if result.get('quick_INT6', -1) > 0 else "N/A")
                row.append(f"{result.get('quick_INT4', -1):.3f}" if result.get('quick_INT4', -1) > 0 else "N/A")
                
                f.write('| ' + ' | '.join(row) + ' |\n')
            
            f.write(f"\n## Performance Summary\n\n")
            f.write("### Speedup vs NCCL\n\n")
            
            # Calculate speedups
            speedup_headers = ['Size', 'Tensor MP All-Reduce', 
                              'Quick FP', 'Quick INT8', 'Quick INT6', 'Quick INT4']
            f.write('| ' + ' | '.join(speedup_headers) + ' |\n')
            f.write('|' + '|'.join(['---'] * len(speedup_headers)) + '|\n')
            
            for size_key in self.results:
                result = self.results[size_key]
                nccl_time = result.get('nccl', -1)
                
                if nccl_time <= 0:
                    continue
                
                row = [size_key]
                
                for method in ['tensor_mp_all_reduce', 'quick_FP', 'quick_INT8', 'quick_INT6', 'quick_INT4']:
                    method_time = result.get(method, -1)
                    if method_time > 0:
                        speedup = nccl_time / method_time
                        row.append(f"{speedup:.2f}x")
                    else:
                        row.append("N/A")
                
                f.write('| ' + ' | '.join(row) + ' |\n')

def run_benchmark_process(rank: int, world_size: int, mode: ExecutionMode, output_prefix: str):
    """Run benchmark in a single process."""
    # Set device
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    # Set random seeds
    torch.manual_seed(42)
    random.seed(44)
    
    # Run benchmark
    runner = BenchmarkRunner(rank, world_size, device, mode)
    runner.run_benchmark()
    
    # Save results
    output_file = f"{output_prefix}_ws{world_size}_{mode.value}.md"
    runner.save_results(output_file)
    
    if rank == 0:
        print(f"Benchmark completed. Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Benchmark all-reduce implementations')
    parser.add_argument('--world-size', type=int, required=True, choices=WORLD_SIZES,
                       help='World size for distributed training')
    parser.add_argument('--mode', type=str, choices=['eager', 'cudagraph', 'both'], 
                       default='both', help='Execution mode')
    parser.add_argument('--output', type=str, default='benchmark_results',
                       help='Output file prefix for results')
    
    args = parser.parse_args()
    
    # Set multiprocessing start method
    multiprocessing.set_start_method('spawn', force=True)
    
    modes_to_test = []
    if args.mode == 'both':
        modes_to_test = [ExecutionMode.EAGER, ExecutionMode.CUDAGRAPH]
    elif args.mode == 'eager':
        modes_to_test = [ExecutionMode.EAGER]
    elif args.mode == 'cudagraph':
        modes_to_test = [ExecutionMode.CUDAGRAPH]
    
    for mode in modes_to_test:
        print(f"Running benchmark in {mode.value} mode...")
        
        # Start processes
        processes = []
        for rank in range(args.world_size):
            p = multiprocessing.Process(
                target=run_benchmark_process,
                args=(rank, args.world_size, mode, args.output)
            )
            p.start()
            processes.append(p)
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        print(f"Completed benchmark in {mode.value} mode")

if __name__ == "__main__":
    main()