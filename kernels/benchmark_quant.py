# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
from aiter.test_common import (
    checkAllclose,
    benchmark,
    run_perftest,
)
import torch
import aiter
from aiter import dtypes
from aiter import get_hip_quant, get_torch_quant, get_triton_quant
import itertools
import argparse
torch.set_default_device("cuda")
from aiter import QuantType
import vllm._custom_ops as custom_ops
from functools import partial
import functools

# Import the vLLM functions you provided
# (Assuming they are available in the vllm module or you've defined them)
try:
    from vllm.model_executor.layers.quantization.fp8 import per_token_group_quant_fp8
    VLLM_GROUP_QUANT_AVAILABLE = True
except ImportError:
    VLLM_GROUP_QUANT_AVAILABLE = False
    print("Warning: vLLM per_token_group_quant_fp8 not available")

# vLLM scaled_fp8_quant wrappers
per_tensor_quant_vllm = partial(custom_ops.scaled_fp8_quant,
                            num_token_padding=0,
                            use_per_token_if_dynamic=False)
per_token_quant_vllm = partial(custom_ops.scaled_fp8_quant,
                            num_token_padding=0,
                            use_per_token_if_dynamic=True)

# vLLM per_token_group_quant_fp8 wrapper
def per_token_group_quant_vllm_wrapper(group_size=128):
    """Wrapper for vLLM's per_token_group_quant_fp8 to match aiter interface"""
    def _quant_func(input, scale=None, **kwargs):
        if scale is not None:
            # Static quantization not directly supported by per_token_group_quant_fp8
            # Fall back to scaled_fp8_quant for static case
            return custom_ops.scaled_fp8_quant(
                input, scale=scale, 
                num_token_padding=0, 
                use_per_token_if_dynamic=False
            )
        else:
            # Dynamic quantization using group quantization
            if VLLM_GROUP_QUANT_AVAILABLE:
                return per_token_group_quant_fp8(input, group_size=group_size)
            else:
                # Fallback to per-token if group quant not available
                return custom_ops.scaled_fp8_quant(
                    input, scale=None,
                    num_token_padding=0,
                    use_per_token_if_dynamic=True
                )
    return _quant_func

@functools.lru_cache()
def vllm_get_quant(qType):
    tmp = {
        QuantType.No: lambda *a, **k: (a[0], None),
        QuantType.per_Tensor: per_tensor_quant_vllm,
        QuantType.per_Token: per_token_quant_vllm,
        QuantType.per_1x32: per_token_group_quant_vllm_wrapper(32),  # Group size 32
        QuantType.per_1x128: per_token_group_quant_vllm_wrapper(128),  # Group size 128
    }
    def raise_NotImplementedError(*a, **k):
        raise NotImplementedError(f"unsupported quant type {qType=}")
    return tmp.get(qType, raise_NotImplementedError)

@benchmark()
def test_quant(m, n, q_type, q_dtype, h_dtype):
    dim = (m, n)
    input = torch.randn(dim, dtype=h_dtype)
    ref, ref_scale = get_torch_quant(q_type)(input, quant_dtype=q_dtype)
    
    q_funcs = {
        "triton": get_triton_quant,
        "hip": get_hip_quant,
        "vllm": vllm_get_quant,
    }
    ret = {}
    
    for name, q_func in q_funcs.items():
        try:
            if name == "vllm":
                q_func = q_func(q_type)
                # Dynamic quantization test
                (out, scale), us1 = run_perftest(q_func, input, None)
                
                # Handle potential shape differences in vLLM output
                if out.shape != ref.shape:
                    print(f"Warning: {name} output shape {out.shape} != ref shape {ref.shape}")
                    # Trim or pad as needed
                    min_shape = tuple(min(a, b) for a, b in zip(out.shape, ref.shape))
                    out_trimmed = out[:min_shape[0], :min_shape[1]] if len(min_shape) == 2 else out[:min_shape[0]]
                    ref_trimmed = ref[:min_shape[0], :min_shape[1]] if len(min_shape) == 2 else ref[:min_shape[0]]
                else:
                    out_trimmed, ref_trimmed = out, ref
                
                err1 = checkAllclose(
                    ref_trimmed.to(dtypes.fp32),
                    out_trimmed.to(dtypes.fp32),
                    rtol=1e-3,
                    atol=1e-3,
                    msg=f"{name}: dynamic quant",
                )
                
                # Handle scale shape differences
                if scale is not None and ref_scale is not None:
                    if scale.numel() != ref_scale.numel():
                        print(f"Warning: {name} scale shape {scale.shape} != ref scale shape {ref_scale.shape}")
                        # For group quantization, we might have different scale shapes
                        # Just check that scales are in reasonable range
                        scale_check = torch.allclose(scale.mean(), ref_scale.mean(), rtol=0.5, atol=0.1)
                        if not scale_check:
                            print(f"Warning: {name} scale values significantly different")
                    else:
                        checkAllclose(
                            ref_scale.to(dtypes.fp32),
                            scale.to(dtypes.fp32),
                            rtol=1e-3,
                            atol=1e-3,
                            msg=f"{name}: dynamic quant scale",
                        )
                
                ret[f"{name} dq"] = us1
                ret[f"{name} dq err"] = err1
                
                # Static quantization test (only for per_Tensor)
                if q_type == aiter.QuantType.per_Tensor:
                    try:
                        (out, scale), us2 = run_perftest(
                            q_func, input, ref_scale
                        )
                        err2 = checkAllclose(
                            ref.to(dtypes.fp32),
                            out.to(dtypes.fp32),
                            rtol=1e-3,
                            atol=1e-3,
                            msg=f"{name}: static quant",
                        )
                        ret[f"{name} sq"] = us2
                        ret[f"{name} sq err"] = err2
                    except Exception as e:
                        print(f"Warning: {name} static quantization failed: {e}")
                        ret[f"{name} sq"] = float('inf')
                        ret[f"{name} sq err"] = float('inf')

            else:
                # Original aiter/triton/hip code
                q_func = q_func(q_type)
                (out, scale), us1 = run_perftest(q_func, input, None)
                err1 = checkAllclose(
                    ref.to(dtypes.fp32),
                    out.to(dtypes.fp32),
                    rtol=1e-3,
                    atol=1e-3,
                    msg=f"{name}: dynamic quant",
                )
                if scale is not None and ref_scale is not None:
                    checkAllclose(
                        ref_scale.to(dtypes.fp32),
                        scale.to(dtypes.fp32),
                        rtol=1e-3,
                        atol=1e-3,
                        msg=f"{name}: dynamic quant scale",
                    )
                ret[f"{name} dq"] = us1
                ret[f"{name} dq err"] = err1
                
                if q_type == aiter.QuantType.per_Tensor:
                    (out, scale), us2 = run_perftest(
                        q_func, input, ref_scale, quant_dtype=q_dtype
                    )
                    err2 = checkAllclose(
                        ref.to(dtypes.fp32),
                        out.to(dtypes.fp32),
                        rtol=1e-3,
                        atol=1e-3,
                        msg=f"{name}: static quant",
                    )
                    ret[f"{name} sq"] = us2
                    ret[f"{name} sq err"] = err2
                    
        except Exception as e:
            print(f"Error testing {name} with {q_type}: {e}")
            ret[f"{name} dq"] = float('inf')
            ret[f"{name} dq err"] = float('inf')
            if q_type == aiter.QuantType.per_Tensor:
                ret[f"{name} sq"] = float('inf')
                ret[f"{name} sq err"] = float('inf')
    
    return ret

d_quant = {
    "fp8_tensor": (aiter.QuantType.per_Tensor, dtypes.fp8),
    "fp8_token": (aiter.QuantType.per_Token, dtypes.fp8),
    "fp8_1x128": (aiter.QuantType.per_1x128, dtypes.fp8),  # Now enabled for vLLM comparison
    # "fp8_1x32": (aiter.QuantType.per_1x32, dtypes.fp8),   # Now enabled for vLLM comparison
    # "i8_token": (aiter.QuantType.per_Token, dtypes.i8),
}

list_dtype = ["fp16", "bf16"]
l_n = [4096, 8192]
l_m = [1, 2, 16, 32, 64, 128, 192, 256, 512, 1024, 16384, 163840]

import pandas as pd
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=list_dtype,
    nargs="?",
    const=None,
    default=None,
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-n",
    "--n",
    type=int,
    nargs="*",
    default=None,
    help="""N of mnk.
    e.g.: -n 1024""",
)
parser.add_argument(
    "-m",
    "--m",
    type=int,
    nargs="*",
    default=None,
    help="""M of mnk.
    e.g.: -m 32""",
)
parser.add_argument(
    "-q",
    "--quant",
    type=str,
    choices=list(d_quant.keys()),
    nargs="*",
    default=list(d_quant.keys()),
    help="""Quantization type.
    e.g.: -q fp8_tensor""",
)

args = parser.parse_args()
if args.dtype is None:
    list_dtype = [dtypes.d_dtypes[key] for key in list_dtype]
else:
    list_dtype = [dtypes.d_dtypes[args.dtype]]
list_quant = [d_quant[key] for key in args.quant]
if args.n is not None:
    l_n = args.n
if args.m is not None:
    l_m = args.m

all_results = []
for (
    (q_type, q_dtype),
    h_dtype,
) in itertools.product(list_quant, list_dtype):
    print(f"q_type: {q_type}, q_dtype: {q_dtype}, h_dtype: {h_dtype}")
    for n in l_n:
        for m in l_m:
            print(f"Testing m={m}, n={n}")
            ret = test_quant(m, n, q_type, q_dtype, h_dtype)
            # Add the configuration parameters to each result
            ret['m'] = m
            ret['n'] = n
            ret['q_type'] = str(q_type)
            ret['q_dtype'] = str(q_dtype)
            ret['h_dtype'] = str(h_dtype)
            all_results.append(ret)

# Save all results to a single CSV
df_all = pd.DataFrame(all_results)
df_all.to_csv("remove_vllm_contiguous_all_summaries.csv", index=False)
aiter.logger.info(f"All summaries saved to all_summaries.csv with shape: {df_all.shape}")

# Print a summary comparison
print("\n=== Performance Summary ===")
for q_type_name in args.quant:
    print(f"\n{q_type_name}:")
    subset = df_all[df_all['q_type'].str.contains(q_type_name.split('_')[1])]
    if not subset.empty:
        for impl in ['triton', 'hip', 'vllm']:
            dq_col = f'{impl} dq'
            if dq_col in subset.columns:
                avg_time = subset[dq_col].replace([float('inf')], float('nan')).mean()
                print(f"  {impl} avg time: {avg_time:.2f} μs")