import torch
import torch.nn.functional as F
import aiter
from aiter.test_common import run_perftest, checkAllclose, benchmark
from aiter import dtypes
import pandas as pd
import argparse
import os
from datetime import datetime
import vllm._custom_ops as custom_ops

def torch_silu_and_mul(input: torch.Tensor) -> torch.Tensor:
    d = input.shape[-1] // 2
    x, y = input.split([d, d], dim=-1)
    out = F.silu(x) * y
    return out

def torch_scaled_silu_and_mul(input: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    d = input.shape[-1] // 2
    x, y = input.split([d, d], dim=-1)
    out = F.silu(x) * y / scale
    return out.to(dtypes.fp8)

# Check if vLLM is available
try:
    import vllm
    # Try to access the ops to see if they're available
    torch.ops._C.silu_and_mul
    torch.ops._C.scaled_silu_and_mul
    VLLM_AVAILABLE = True
    VLLM_SCALED_AVAILABLE = True
except (ImportError, AttributeError):
    VLLM_AVAILABLE = False
    VLLM_SCALED_AVAILABLE = False
    try:
        import vllm
        torch.ops._C.silu_and_mul
        VLLM_AVAILABLE = True
    except (ImportError, AttributeError):
        pass
    try:
        torch.ops._C.scaled_silu_and_mul
        VLLM_SCALED_AVAILABLE = True
    except AttributeError:
        pass
    
    if not VLLM_AVAILABLE:
        print("Warning: vLLM not available or torch.ops._C.silu_and_mul not found")
    if not VLLM_SCALED_AVAILABLE:
        print("Warning: torch.ops._C.scaled_silu_and_mul not found")

@benchmark()
def test_silu_and_mul(m, n, dtype):
    input = torch.randn(m, n, dtype=dtype, device="cuda")
    out_aiter = torch.empty((m, n // 2), dtype=dtype, device="cuda")
    
    # Reference implementation
    ref, us_torch = run_perftest(
        torch_silu_and_mul,
        input,
        num_warmup=2,
        num_iters=3,
    )
    
    # Test aiter implementation
    _, us_aiter = run_perftest(
        aiter.silu_and_mul,
        out_aiter,
        input,
        num_warmup=10,
        num_iters=100,
    )
    
    # Check if aiter results are close to reference
    checkAllclose(ref, out_aiter, rtol=1e-3, atol=1e-3)
    
    results = {
        "us_torch": us_torch,
        "us_aiter": us_aiter,
        "speedup_aiter": us_torch / us_aiter if us_aiter > 0 else 0
    }
    
    # Test vLLM implementation if available
    if VLLM_AVAILABLE:
        try:
            out_vllm = torch.empty((m, n // 2), dtype=dtype, device="cuda")
            _, us_vllm = run_perftest(
                torch.ops._C.silu_and_mul,
                out_vllm,
                input,
                num_warmup=10,
                num_iters=100,
            )
            
            # Check if vLLM results are close to reference
            checkAllclose(ref, out_vllm, rtol=1e-3, atol=1e-3)
            
            results.update({
                "us_vllm": us_vllm,
                "speedup_vllm": us_torch / us_vllm if us_vllm > 0 else 0,
                "aiter_vs_vllm": us_vllm / us_aiter if us_aiter > 0 else 0
            })
        except Exception as e:
            print(f"Error testing vLLM implementation: {e}")
            results.update({
                "us_vllm": float('nan'),
                "speedup_vllm": float('nan'),
                "aiter_vs_vllm": float('nan')
            })
    else:
        results.update({
            "us_vllm": None,
            "speedup_vllm": None,
            "aiter_vs_vllm": None
        })
    
    return results

@benchmark()
def test_scaled_silu_and_mul(m, n, dtype):
    """Test scaled version with both aiter and vLLM implementations"""
    input = torch.randn(m, n, dtype=dtype, device="cuda")
    scale = torch.max(input).to(torch.float32)
    out_aiter = torch.empty((m, n // 2), dtype=dtypes.fp8, device="cuda")
    
    # Reference implementation
    ref, us_torch = run_perftest(
        torch_scaled_silu_and_mul,
        input,
        scale,
        num_warmup=2,
        num_iters=3,
    )
    
    # Test aiter implementation
    _, us_aiter = run_perftest(
        aiter.scaled_silu_and_mul,
        out_aiter,
        input,
        scale,
        num_warmup=10,
        num_iters=100,
    )
    
    # Check if aiter results are close to reference
    checkAllclose(ref.to(torch.float), out_aiter.to(torch.float), rtol=1e-3, atol=1e-3)
    
    results = {
        "us_torch": us_torch,
        "us_aiter": us_aiter,
        "speedup_aiter": us_torch / us_aiter if us_aiter > 0 else 0
    }
    
    # Test vLLM scaled implementation if available
    if VLLM_SCALED_AVAILABLE:
        try:
            out_vllm = torch.empty((m, n // 2), dtype=dtypes.fp8, device="cuda")
            _, us_vllm = run_perftest(
                torch.ops._C.scaled_silu_and_mul,
                out_vllm,
                input,
                scale,
                num_warmup=10,
                num_iters=100,
            )
            
            # Check if vLLM results are close to reference
            checkAllclose(ref.to(torch.float), out_vllm.to(torch.float), rtol=1e-3, atol=1e-3)
            
            results.update({
                "us_vllm": us_vllm,
                "speedup_vllm": us_torch / us_vllm if us_vllm > 0 else 0,
                "aiter_vs_vllm": us_vllm / us_aiter if us_aiter > 0 else 0
            })
        except Exception as e:
            print(f"Error testing vLLM scaled implementation: {e}")
            results.update({
                "us_vllm": float('nan'),
                "speedup_vllm": float('nan'),
                "aiter_vs_vllm": float('nan')
            })
    else:
        results.update({
            "us_vllm": None,
            "speedup_vllm": None,
            "aiter_vs_vllm": None
        })
    
    return results

def save_results_to_csv(df, filename, output_dir="benchmark_results"):
    """Save DataFrame to CSV with timestamp and metadata"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename with timestamp
    base_name = filename.replace('.csv', '')
    csv_filename = f"{base_name}_{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    # Also save a "latest" version for easy access
    latest_path = os.path.join(output_dir, f"{base_name}_latest.csv")
    df.to_csv(latest_path, index=False)
    print(f"Latest results saved to: {latest_path}")
    
    return csv_path

def add_system_info(df):
    """Add system information to the DataFrame"""
    # Get GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
    else:
        gpu_name = "N/A"
        gpu_memory = 0
    
    # Add system info columns
    df['gpu_name'] = gpu_name
    df['gpu_memory_gb'] = gpu_memory
    df['vllm_available'] = VLLM_AVAILABLE
    df['vllm_scaled_available'] = VLLM_SCALED_AVAILABLE
    df['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df['torch_version'] = torch.__version__
    
    return df

# Configuration
l_dtype = ["fp16", "bf16"]
l_m = [1, 32, 64, 128, 256, 512, 1024, 4096, 8192]
l_n = [1024, 4096, 8192]

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Benchmark silu_and_mul operations against aiter and vLLM implementations",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=l_dtype,
    nargs="?",
    const=None,
    default=None,
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-m",
    type=int,
    nargs="?",
    const=None,
    default=None,
    help="""M of mnk.
    e.g.: -m 32""",
)
parser.add_argument(
    "-n",
    type=int,
    nargs="?",
    const=None,
    default=None,
    help="""N of mnk.
    e.g.: -n 1024""",
)
parser.add_argument(
    "--test-scaled",
    action="store_true",
    help="Also run scaled silu_and_mul tests"
)
parser.add_argument(
    "--only-scaled",
    action="store_true",
    help="Only run scaled silu_and_mul tests"
)
parser.add_argument(
    "--output-dir",
    type=str,
    default="benchmark_results",
    help="Directory to save CSV results (default: benchmark_results)"
)
parser.add_argument(
    "--prefix",
    type=str,
    default="",
    help="Prefix for output filenames"
)

if __name__ == "__main__":
    args = parser.parse_args()
    
    if args.dtype is None:
        l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
    else:
        l_dtype = [dtypes.d_dtypes[args.dtype]]
    if args.m is not None:
        l_m = [args.m]
    if args.n is not None:
        l_n = [args.n]
    
    print(f"vLLM silu_and_mul available: {VLLM_AVAILABLE}")
    print(f"vLLM scaled_silu_and_mul available: {VLLM_SCALED_AVAILABLE}")
    print(f"Testing configurations: dtypes={[str(d) for d in l_dtype]}, m={l_m}, n={l_n}")
    print(f"Results will be saved to: {args.output_dir}")
    
    filename_prefix = f"{args.prefix}_" if args.prefix else ""
    
    # Test regular silu_and_mul (unless only-scaled is specified)
    if not args.only_scaled:
        df_results = []
        total_tests = len(l_dtype) * len(l_m) * len(l_n)
        current_test = 0
        
        print("\n" + "="*60)
        print("TESTING REGULAR SILU_AND_MUL")
        print("="*60)
        
        for dtype in l_dtype:
            for m in l_m:
                for n in l_n:
                    current_test += 1
                    print(f"Testing silu_and_mul [{current_test}/{total_tests}]: m={m}, n={n}, dtype={dtype}")
                    ret = test_silu_and_mul(m, n, dtype)
                    ret.update({"m": m, "n": n, "dtype": str(dtype)})
                    df_results.append(ret)
        
        df = pd.DataFrame(df_results)
        df = add_system_info(df)
        
        # Reorder columns for better readability
        column_order = ['timestamp', 'm', 'n', 'dtype', 'us_torch', 'us_aiter', 'us_vllm', 
                       'speedup_aiter', 'speedup_vllm', 'aiter_vs_vllm', 
                       'gpu_name', 'gpu_memory_gb', 'vllm_available', 'vllm_scaled_available', 'torch_version']
        df = df.reindex(columns=[col for col in column_order if col in df.columns])
        
        print(f"\nsilu_and_mul benchmark results:")
        print(df[['m', 'n', 'dtype', 'us_torch', 'us_aiter', 'us_vllm', 'speedup_aiter', 'speedup_vllm', 'aiter_vs_vllm']].to_string(index=False))
        
        # Save to CSV
        csv_path = save_results_to_csv(df, f"{filename_prefix}silu_and_mul_benchmark.csv", args.output_dir)
        aiter.logger.info(f"silu_and_mul results saved to: {csv_path}")
    
    # Test scaled version if requested or if only-scaled is specified
    if args.test_scaled or args.only_scaled:
        df_scaled_results = []
        total_scaled_tests = len(l_dtype) * len(l_m) * len(l_n)
        current_scaled_test = 0
        
        print("\n" + "="*60)
        print("TESTING SCALED SILU_AND_MUL")
        print("="*60)
        
        for dtype in l_dtype:
            for m in l_m:
                for n in l_n:
                    current_scaled_test += 1
                    print(f"Testing scaled_silu_and_mul [{current_scaled_test}/{total_scaled_tests}]: m={m}, n={n}, dtype={dtype}")
                    ret = test_scaled_silu_and_mul(m, n, dtype)
                    ret.update({"m": m, "n": n, "dtype": str(dtype)})
                    df_scaled_results.append(ret)
        
        df_scaled = pd.DataFrame(df_scaled_results)
        df_scaled = add_system_info(df_scaled)
        
        # Reorder columns
        scaled_column_order = ['timestamp', 'm', 'n', 'dtype', 'us_torch', 'us_aiter', 'us_vllm',
                              'speedup_aiter', 'speedup_vllm', 'aiter_vs_vllm',
                              'gpu_name', 'gpu_memory_gb', 'vllm_available', 'vllm_scaled_available', 'torch_version']
        df_scaled = df_scaled.reindex(columns=[col for col in scaled_column_order if col in df_scaled.columns])
        
        print(f"\nscaled_silu_and_mul benchmark results:")
        print(df_scaled[['m', 'n', 'dtype', 'us_torch', 'us_aiter', 'us_vllm', 'speedup_aiter', 'speedup_vllm', 'aiter_vs_vllm']].to_string(index=False))
        
        # Save scaled results to CSV
        scaled_csv_path = save_results_to_csv(df_scaled, f"{filename_prefix}scaled_silu_and_mul_benchmark.csv", args.output_dir)
        aiter.logger.info(f"scaled_silu_and_mul results saved to: {scaled_csv_path}")
    
    print(f"\nAll results saved to directory: {args.output_dir}")
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"vLLM silu_and_mul available: {VLLM_AVAILABLE}")
    print(f"vLLM scaled_silu_and_mul available: {VLLM_SCALED_AVAILABLE}")
    
    if not args.only_scaled and 'df' in locals():
        print(f"\nRegular silu_and_mul tests completed: {len(df)} configurations")
        if VLLM_AVAILABLE:
            avg_aiter_speedup = df['speedup_aiter'].mean()
            avg_vllm_speedup = df['speedup_vllm'].mean()
            avg_comparison = df['aiter_vs_vllm'].mean()
            print(f"Average aiter speedup vs PyTorch: {avg_aiter_speedup:.2f}x")
            print(f"Average vLLM speedup vs PyTorch: {avg_vllm_speedup:.2f}x")
            print(f"Average aiter vs vLLM ratio: {avg_comparison:.2f}")
    
    if (args.test_scaled or args.only_scaled) and 'df_scaled' in locals():
        print(f"\nScaled silu_and_mul tests completed: {len(df_scaled)} configurations")
        if VLLM_SCALED_AVAILABLE:
            avg_aiter_speedup = df_scaled['speedup_aiter'].mean()
            avg_vllm_speedup = df_scaled['speedup_vllm'].mean()
            avg_comparison = df_scaled['aiter_vs_vllm'].mean()
            print(f"Average aiter speedup vs PyTorch: {avg_aiter_speedup:.2f}x")
            print(f"Average vLLM speedup vs PyTorch: {avg_vllm_speedup:.2f}x")
            print(f"Average aiter vs vLLM ratio: {avg_comparison:.2f}")