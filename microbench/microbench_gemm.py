# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional
import pandas as pd

import torch
import time
from vllm.utils import direct_register_custom_op

# Implement your functions here
###############################

def _rocm_aiter_tuned_gemm_impl(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        out_dtype: Optional[torch.dtype] = None,
        scale_a: Optional[torch.Tensor] = None,
        scale_b: Optional[torch.Tensor] = None) -> torch.Tensor:

    from aiter.tuned_gemm import tgemm as aiter_tgemm
    
    return aiter_tgemm.mm(input,
                          weight,
                          otype=out_dtype,
                          scale_a=scale_a,
                          scale_b=scale_b,
                          scale_c=None,
                          bias=bias)


def _rocm_aiter_tuned_gemm_fake(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        out_dtype: Optional[torch.dtype] = None,
        scale_a: Optional[torch.Tensor] = None,
        scale_b: Optional[torch.Tensor] = None) -> torch.Tensor:

    m = input.shape[0]
    n = weight.shape[0]
    if out_dtype is None:
        out_dtype = input.dtype
    return torch.empty((m, n), dtype=out_dtype, device=input.device)

direct_register_custom_op(
    op_name="rocm_aiter_tuned_gemm",
    op_func=_rocm_aiter_tuned_gemm_impl,
    mutates_args=[],
    fake_impl=_rocm_aiter_tuned_gemm_fake,
)

def rocm_aiter_tuned_gemm(
        input: torch.Tensor,  # [M, K]
        weight: torch.Tensor,  # [N, K]
        bias: Optional[torch.Tensor] = None,
        out_dtype: Optional[torch.dtype] = None,
        scale_a: Optional[torch.Tensor] = None,
        scale_b: Optional[torch.Tensor] = None) -> torch.Tensor:

    return torch.ops.vllm.rocm_aiter_tuned_gemm(
        input,
        weight,
        bias=bias,
        out_dtype=out_dtype,
        scale_a=scale_a,
        scale_b=scale_b,
    )

rocm_aiter_tuned_gemm_compiled = torch.compile(rocm_aiter_tuned_gemm,
                            fullgraph=True,
                            backend="inductor",
                            mode="reduce-overhead",
                            dynamic=True)


# ----------- User-configurable: path to CSV files -----------
INPUT_CSV = "aiter/aiter/configs/untuned_gemm.csv"
OUTPUT_CSV = "latency/untuned_gemm_latency_results.csv"
DEVICE = "cuda"  # or "cpu"
REPEAT = 100  # Number of iterations for measuring latency
WARMUP = 10   # Warmup iterations

# Utility to safely parse dtype strings from CSV
def parse_torch_dtype(dtype_str):
    """
    Converts a string like 'torch.bfloat16' into torch.bfloat16.
    """
    try:
        # Ensure string is prepended with 'torch.'
        if not dtype_str.startswith("torch."):
            dtype_str = "torch." + dtype_str
        return eval(dtype_str)
    except Exception as e:
        raise ValueError(f"Invalid dtype string: {dtype_str}") from e

def rocm_aiter_tuned_gemm_input_factory(m, k, n, in_dtype_str, out_dtype_str, device=DEVICE):
    in_dtype = parse_torch_dtype(in_dtype_str)
    out_dtype = parse_torch_dtype(out_dtype_str)
    input = torch.randn((m, k), dtype=in_dtype, device=device)
    weight = torch.randn((n, k), dtype=in_dtype, device=device)
    bias = None
    scale_a = None
    scale_b = None
    return input, weight, bias, out_dtype, scale_a, scale_b


# ----------- Latency measurement function -----------
def measure_latency(fn, input_generator, warmup=WARMUP, repeat=REPEAT):
    # Warmup
    for _ in range(warmup):
        fn(*input_generator())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn(*input_generator())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    return sum(times) / repeat

# ----------- Main benchmarking loop -----------
def main():
    df = pd.read_csv(INPUT_CSV)
    rows = []
    for idx, row in df.iterrows():
        m, n, k = int(row["M"]), int(row["N"]), int(row["K"])
        in_dtype = row["dtype"]
        out_dtype = row["outdtype"]
        input_gen = lambda: rocm_aiter_tuned_gemm_input_factory(m, k, n, in_dtype, out_dtype)
        latency = measure_latency(rocm_aiter_tuned_gemm, input_gen) * 1000  # ms
        print(f"M={m}, N={n}, K={k}, dtype={in_dtype}, outdtype={out_dtype}: latency={latency:.4f} ms")
        result = {
            "M": m, "N": n, "K": k,
            "dtype": in_dtype,
            "outdtype": out_dtype,
            "latency_ms": latency
        }
        rows.append(result)
    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
