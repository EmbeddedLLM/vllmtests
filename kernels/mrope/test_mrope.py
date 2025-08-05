"""
Expert-level MRoPE kernel testing and benchmarking suite.
Provides comprehensive testing, performance analysis, and statistical validation.
"""

import torch
import time
import logging
import json
import csv
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, NamedTuple
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from collections import defaultdict
import statistics
import pytest
import torch.nn as nn
from transformers import AutoConfig

from kernels import (
    mrope_forward_native, 
    mrope_forward_liger_kernel,
    mrope_forward_liger_kernel_adapted_to_vllm_input
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestConfig:
    """Configuration for test parameters."""
    model_name: str
    tp_size: int
    num_tokens: int
    dtype: torch.dtype
    atol: float
    rtol: float
    num_heads: int
    num_kv_heads: int
    head_dim: int
    max_position: int
    rope_theta: float
    mrope_section: List[int]
    is_neox_style: bool = True

@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    config: TestConfig
    native_time: float
    liger_kernel_time: float
    vllm_adapted_time: float
    liger_speedup: float
    vllm_adapted_speedup: float
    memory_usage: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['config'] = asdict(self.config)
        result['config']['dtype'] = str(self.config.dtype)
        return result

class TestData(NamedTuple):
    """Container for test input data."""
    positions: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor

class MRoPETestSuite:
    """Expert-level MRoPE testing and benchmarking suite."""
    
    # Test configurations
    MODEL_TP_CONFIG = {
        "Qwen/Qwen2-VL-2B-Instruct": [1],
        "Qwen/Qwen2-VL-7B-Instruct": [1],
        "Qwen/Qwen2-VL-72B-Instruct": [2, 4, 8],
        "Qwen/Qwen2.5-VL-3B-Instruct": [1, 2, 4, 8],
        "Qwen/Qwen2.5-VL-7B-Instruct": [1, 2, 4, 8],
        "Qwen/Qwen2.5-VL-72B-Instruct": [2, 4, 8]
    }
    
    DTYPE_TOLERANCES = [
        (torch.bfloat16, 1e-5, 1.6e-2),
        (torch.float16, 1e-4, 1e-3),
        (torch.float32, 1e-6, 1e-5),
    ]
    
    TOKEN_SIZES = [2**i for i in range(0, 18)]
    
    def __init__(self, 
                 warmup_iterations: int = 10,
                 benchmark_iterations: int = 100,
                 output_dir: str = "benchmark_results",
                 device: Optional[torch.device] = None):
        """Initialize the test suite."""
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available. Running on CPU may be slow.")
        
        self.results: List[BenchmarkResult] = []
        
    @contextmanager
    def cuda_timer(self):
        """Context manager for accurate CUDA timing."""
        if self.device.type == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            torch.cuda.synchronize()
            start_event.record()
            yield lambda: start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
            end_event.record()
            torch.cuda.synchronize()
        else:
            start_time = time.perf_counter()
            yield lambda: time.perf_counter() - start_time

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        if self.device.type == 'cuda':
            return {
                'allocated_gb': torch.cuda.memory_allocated() / 1e9,
                'reserved_gb': torch.cuda.memory_reserved() / 1e9,
                'max_allocated_gb': torch.cuda.max_memory_allocated() / 1e9,
            }
        return {'allocated_gb': 0.0, 'reserved_gb': 0.0, 'max_allocated_gb': 0.0}

    def create_mrope_helper(self, config: TestConfig) -> 'SimpleMRoPEClass':
        """Create MRoPE helper class with given configuration."""
        return SimpleMRoPEClass(
            head_size=config.head_dim,
            rotary_dim=config.head_dim,
            max_position_embeddings=config.max_position,
            base=config.rope_theta,
            is_neox_style=config.is_neox_style,
            dtype=config.dtype,
            mrope_section=config.mrope_section,
        )

    def generate_test_data(self, config: TestConfig) -> TestData:
        """Generate test data for given configuration."""
        # Create 2D positions (3, num_tokens) for multimodal case
        positions = torch.randint(
            0, config.max_position // 4, 
            (3, config.num_tokens), 
            device=self.device
        )
        
        # Create query and key tensors
        query = torch.randn(
            config.num_tokens, config.num_heads * config.head_dim,
            dtype=config.dtype, device=self.device
        )
        key = torch.randn(
            config.num_tokens, config.num_kv_heads * config.head_dim,
            dtype=config.dtype, device=self.device
        )
        
        # Generate cos/sin cache
        mrope_helper = self.create_mrope_helper(config)
        cos_sin = mrope_helper.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        
        return TestData(positions, query, key, cos, sin)

    def validate_correctness(self, config: TestConfig, test_data: TestData) -> bool:
        """Validate correctness of kernel implementations."""
        try:
            # Get reference results
            query_native, key_native = mrope_forward_native(
                test_data.positions,
                test_data.query.clone(),
                test_data.key.clone(),
                test_data.cos,
                test_data.sin,
                config.mrope_section,
                config.is_neox_style,
                config.head_dim,
                config.head_dim,
            )
            
            # Test Liger kernel
            query_liger, key_liger = mrope_forward_liger_kernel(
                test_data.positions,
                test_data.query.clone(),
                test_data.key.clone(),
                test_data.cos,
                test_data.sin,
                config.mrope_section,
                config.is_neox_style,
                config.head_dim,
                config.head_dim,
            )
            
            # Test vLLM adapted kernel
            query_vllm, key_vllm = mrope_forward_liger_kernel_adapted_to_vllm_input(
                test_data.positions,
                test_data.query.clone(),
                test_data.key.clone(),
                test_data.cos,
                test_data.sin,
                config.mrope_section,
                config.is_neox_style,
                config.head_dim,
                config.head_dim,
            )
            
            # Validate results
            torch.testing.assert_close(query_native, query_liger, atol=config.atol, rtol=config.rtol)
            torch.testing.assert_close(key_native, key_liger, atol=config.atol, rtol=config.rtol)
            torch.testing.assert_close(query_native, query_vllm, atol=config.atol, rtol=config.rtol)
            torch.testing.assert_close(key_native, key_vllm, atol=config.atol, rtol=config.rtol)
            
            return True
            
        except Exception as e:
            logger.error(f"Correctness validation failed: {e}")
            return False

    def benchmark_implementation(self, func, test_data: TestData, config: TestConfig) -> float:
        """Benchmark a single implementation with statistical analysis."""
        # Warmup
        for _ in range(self.warmup_iterations):
            func(
                test_data.positions,
                test_data.query.clone(),
                test_data.key.clone(),
                test_data.cos,
                test_data.sin,
                config.mrope_section,
                config.is_neox_style,
                config.head_dim,
                config.head_dim,
            )
        
        # Benchmark
        times = []
        for _ in range(self.benchmark_iterations):
            with self.cuda_timer() as timer:
                func(
                    test_data.positions,
                    test_data.query.clone(),
                    test_data.key.clone(),
                    test_data.cos,
                    test_data.sin,
                    config.mrope_section,
                    config.is_neox_style,
                    config.head_dim,
                    config.head_dim,
                )
            times.append(timer())
        
        # Return median time for robustness
        return statistics.median(times)

    def run_benchmark(self, config: TestConfig) -> Optional[BenchmarkResult]:
        """Run complete benchmark for a configuration."""
        logger.info(f"Benchmarking: {config.model_name}, TP={config.tp_size}, "
                   f"tokens={config.num_tokens}, dtype={config.dtype}")
        
        try:
            # Generate test data
            test_data = self.generate_test_data(config)
            
            # Validate correctness first
            if not self.validate_correctness(config, test_data):
                logger.error(f"Correctness validation failed for config: {config}")
                return None
            
            # Benchmark implementations
            native_time = self.benchmark_implementation(mrope_forward_native, test_data, config)
            liger_time = self.benchmark_implementation(mrope_forward_liger_kernel, test_data, config)
            vllm_time = self.benchmark_implementation(
                mrope_forward_liger_kernel_adapted_to_vllm_input, test_data, config
            )
            
            # Calculate speedups
            liger_speedup = native_time / liger_time if liger_time > 0 else 0
            vllm_speedup = native_time / vllm_time if vllm_time > 0 else 0
            
            # Get memory usage
            memory_usage = self.get_memory_usage()
            
            result = BenchmarkResult(
                config=config,
                native_time=native_time,
                liger_kernel_time=liger_time,
                vllm_adapted_time=vllm_time,
                liger_speedup=liger_speedup,
                vllm_adapted_speedup=vllm_speedup,
                memory_usage=memory_usage
            )
            
            self.log_result(result)
            return result
            
        except Exception as e:
            logger.error(f"Benchmark failed for config {config}: {e}")
            return None

    def log_result(self, result: BenchmarkResult):
        """Log benchmark result."""
        config = result.config
        logger.info(f"Results for {config.model_name} (TP={config.tp_size}, tokens={config.num_tokens}):")
        logger.info(f"  Native: {result.native_time:.6f}s")
        logger.info(f"  Liger:  {result.liger_kernel_time:.6f}s (speedup: {result.liger_speedup:.2f}x)")
        logger.info(f"  vLLM:   {result.vllm_adapted_time:.6f}s (speedup: {result.vllm_adapted_speedup:.2f}x)")
        logger.info(f"  Memory: {result.memory_usage['allocated_gb']:.2f}GB allocated")

    def save_results(self):
        """Save benchmark results to files."""
        if not self.results:
            logger.warning("No results to save")
            return
        
        # Save as JSON
        json_path = self.output_dir / "benchmark_results.json"
        with open(json_path, 'w') as f:
            json.dump([result.to_dict() for result in self.results], f, indent=2)
        
        # Save as CSV
        csv_path = self.output_dir / "benchmark_results.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow([
                'model_name', 'tp_size', 'num_tokens', 'dtype', 'num_heads', 'num_kv_heads',
                'head_dim', 'native_time', 'liger_time', 'vllm_time', 
                'liger_speedup', 'vllm_speedup', 'memory_allocated_gb'
            ])
            # Data
            for result in self.results:
                config = result.config
                writer.writerow([
                    config.model_name, config.tp_size, config.num_tokens, str(config.dtype),
                    config.num_heads, config.num_kv_heads, config.head_dim,
                    result.native_time, result.liger_kernel_time, result.vllm_adapted_time,
                    result.liger_speedup, result.vllm_adapted_speedup,
                    result.memory_usage['allocated_gb']
                ])
        
        logger.info(f"Results saved to {json_path} and {csv_path}")

    def run_full_benchmark_suite(self):
        """Run the complete benchmark suite."""
        logger.info("Starting comprehensive MRoPE benchmark suite")
        
        total_configs = sum(
            len(tp_list) * len(self.DTYPE_TOLERANCES) * len(self.TOKEN_SIZES)
            for tp_list in self.MODEL_TP_CONFIG.values()
        )
        
        completed = 0
        
        for model_name, tp_list in self.MODEL_TP_CONFIG.items():
            try:
                config_obj = AutoConfig.from_pretrained(model_name)
            except Exception as e:
                logger.error(f"Failed to load config for {model_name}: {e}")
                continue
                
            for tp_size in tp_list:
                # Calculate model dimensions
                total_num_kv_heads = config_obj.num_key_value_heads
                total_num_heads = config_obj.num_attention_heads
                num_heads = total_num_heads // tp_size
                num_kv_heads = max(1, total_num_kv_heads // tp_size)
                head_dim = config_obj.hidden_size // total_num_heads
                
                for dtype, atol, rtol in self.DTYPE_TOLERANCES:
                    for num_tokens in self.TOKEN_SIZES:
                        config = TestConfig(
                            model_name=model_name,
                            tp_size=tp_size,
                            num_tokens=num_tokens,
                            dtype=dtype,
                            atol=atol,
                            rtol=rtol,
                            num_heads=num_heads,
                            num_kv_heads=num_kv_heads,
                            head_dim=head_dim,
                            max_position=config_obj.max_position_embeddings,
                            rope_theta=config_obj.rope_theta,
                            mrope_section=config_obj.rope_scaling["mrope_section"],
                        )
                        
                        result = self.run_benchmark(config)
                        if result:
                            self.results.append(result)
                        
                        completed += 1
                        logger.info(f"Progress: {completed}/{total_configs} ({100*completed/total_configs:.1f}%)")
        
        self.save_results()
        self.generate_summary_report()

    def generate_summary_report(self):
        """Generate a summary report of benchmark results."""
        if not self.results:
            return
        
        # Group results by model and calculate statistics
        model_stats = defaultdict(list)
        for result in self.results:
            model_stats[result.config.model_name].append(result)
        
        report_path = self.output_dir / "summary_report.txt"
        with open(report_path, 'w') as f:
            f.write("MRoPE Kernel Benchmark Summary Report\n")
            f.write("=" * 50 + "\n\n")
            
            for model_name, results in model_stats.items():
                f.write(f"Model: {model_name}\n")
                f.write("-" * 30 + "\n")
                
                liger_speedups = [r.liger_speedup for r in results]
                vllm_speedups = [r.vllm_adapted_speedup for r in results]
                
                f.write(f"Liger Kernel Speedups:\n")
                f.write(f"  Mean: {statistics.mean(liger_speedups):.2f}x\n")
                f.write(f"  Median: {statistics.median(liger_speedups):.2f}x\n")
                f.write(f"  Max: {max(liger_speedups):.2f}x\n")
                f.write(f"  Min: {min(liger_speedups):.2f}x\n")
                
                f.write(f"vLLM Adapted Speedups:\n")
                f.write(f"  Mean: {statistics.mean(vllm_speedups):.2f}x\n")
                f.write(f"  Median: {statistics.median(vllm_speedups):.2f}x\n")
                f.write(f"  Max: {max(vllm_speedups):.2f}x\n")
                f.write(f"  Min: {min(vllm_speedups):.2f}x\n\n")
        
        logger.info(f"Summary report saved to {report_path}")


class SimpleMRoPEClass(nn.Module):
    """Simplified MRoPE class for testing."""
    
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        mrope_section: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings * 4  # for video
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        
        cache = self._compute_cos_sin_cache()
        cache = cache.to(dtype)
        self.register_buffer("cos_sin_cache", cache, persistent=False)
        
        self.mrope_section = mrope_section
        if self.mrope_section:
            assert sum(self.mrope_section) == rotary_dim // 2
        
    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        """Compute the inverse frequency."""
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cuda')
        inv_freq = 1.0 / (base ** (torch.arange(
            0, self.rotary_dim, 2, dtype=torch.float, device=device) / self.rotary_dim))
        return inv_freq
    
    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        device = inv_freq.device
        t = torch.arange(self.max_position_embeddings, dtype=torch.float, device=device)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache


# Pytest integration
class TestMRoPEKernels:
    """Pytest test class for MRoPE kernels."""
    
    @pytest.fixture
    def test_suite(self):
        """Create test suite fixture."""
        return MRoPETestSuite(warmup_iterations=2, benchmark_iterations=5)
    
    @pytest.mark.parametrize("model_name,tp_size", [
        ("Qwen/Qwen2-VL-2B-Instruct", 1),
        ("Qwen/Qwen2.5-VL-3B-Instruct", 1),
    ])
    @pytest.mark.parametrize("num_tokens", [64, 256, 1024])
    def test_kernel_correctness(self, test_suite, model_name, tp_size, num_tokens):
        """Test kernel correctness for various configurations."""
        config_obj = AutoConfig.from_pretrained(model_name)
        
        config = TestConfig(
            model_name=model_name,
            tp_size=tp_size,
            num_tokens=num_tokens,
            dtype=torch.bfloat16,
            atol=1e-5,
            rtol=1.6e-2,
            num_heads=config_obj.num_attention_heads // tp_size,
            num_kv_heads=max(1, config_obj.num_key_value_heads // tp_size),
            head_dim=config_obj.hidden_size // config_obj.num_attention_heads,
            max_position=config_obj.max_position_embeddings,
            rope_theta=config_obj.rope_theta,
            mrope_section=config_obj.rope_scaling["mrope_section"],
        )
        
        test_data = test_suite.generate_test_data(config)
        assert test_suite.validate_correctness(config, test_data)


def main():
    """Main function to run the benchmark suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MRoPE Kernel Benchmark Suite")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--benchmark", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Output directory")
    parser.add_argument("--models", nargs="+", help="Specific models to test")
    
    args = parser.parse_args()
    
    suite = MRoPETestSuite(
        warmup_iterations=args.warmup,
        benchmark_iterations=args.benchmark,
        output_dir=args.output_dir
    )
    
    if args.models:
        # Filter models
        suite.MODEL_TP_CONFIG = {
            k: v for k, v in suite.MODEL_TP_CONFIG.items() 
            if any(model in k for model in args.models)
        }
    
    suite.run_full_benchmark_suite()


if __name__ == "__main__":
    main()