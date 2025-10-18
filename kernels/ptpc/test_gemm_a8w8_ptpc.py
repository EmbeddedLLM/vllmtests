# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import random
import aiter
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import checkAllclose, perftest, benchmark
import pandas as pd
import argparse
import itertools
import os
from aiter import hipb_mm, hipb_create_extension
from aiter.jit.utils.chip_info import get_gfx
from functools import lru_cache


TEST_NUM_ITERS = 100

@perftest(num_iters=TEST_NUM_ITERS)
def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=dtypes.bf16):
    x = x.to(dtypes.fp32) * x_scale
    weight = weight.to(dtypes.fp32) * w_scale
    out = F.linear(x, weight)
    if bias is not None:
        out = out.to(bias.dtype) + bias
    return out.to(dtype)

@perftest(num_iters=TEST_NUM_ITERS, needTrace=False, num_warmup=10, testGraph=False, num_rotate_args=0)
def run_gemm_ck(x, weight, x_scale, w_scale, bias=None, dtype=dtypes.bf16):
    return aiter.gemm_a8w8_CK(x, weight, x_scale, w_scale, bias, dtype)

@perftest()
def run_gemm_ck_bpreshuffle(x, weight, x_scale, w_scale, bias=None, dtype=dtypes.bf16):
    return aiter.gemm_a8w8_bpreshuffle(x, weight, x_scale, w_scale, None, dtype)

@perftest()
def run_gemm_asm(x, weightshuffle, x_scale, w_scale, bias=None, dtype=dtypes.bf16):
    return aiter.gemm_a8w8_ASM(x, weightshuffle, x_scale, w_scale, bias)

@perftest(num_iters=TEST_NUM_ITERS)
def run_gemm_skinny(
    x, weight, x_scale, w_scale, bias=None, dtype=dtypes.bf16, cu_count=1
):
    out = torch.empty(x.shape[0], weight.shape[0], dtype=dtype, device=x.device)
    aiter.wvSplitKQ(weight, x, out, w_scale, x_scale, cu_count)
    if bias is not None:
        out = out.to(bias.dtype) + bias
    return out.to(dtype)

@benchmark()
def test_gemm(dtype, m, n, k, quantDtype=dtypes.i8):
    dim = (m, n, k)
    x = torch.randn((m, k), dtype=dtype, device="cuda")
    weight = torch.randn((n, k), dtype=dtype, device="cuda")
    x, x_scale = aiter.pertoken_quant(x, quant_dtype=quantDtype)
    weight, w_scale = aiter.pertoken_quant(weight, quant_dtype=quantDtype)
    weightshuffle = shuffle_weight(weight, layout=(16, 16))
    bias = torch.rand([1, n], dtype=dtype, device="cuda") * 10
    a, avg_a = run_torch(x, weight, x_scale, w_scale, bias, dtype)
    b, avg_b = run_gemm_ck(x, weight, x_scale, w_scale, bias, dtype)
    err_b = checkAllclose(a, b, msg="ck: ", rtol=1e-2, atol=1e-2)
    if quantDtype != dtypes.i8:
        c, avg_c = aiter.gemm_a8w8_bpreshuffle(x, weightshuffle, x_scale, w_scale, None, dtype)
        c = c + bias
        err_c = checkAllclose(a, c, msg="ck bpreshuffle: ", rtol=1e-2, atol=1e-2)
    else:
        avg_c = None; err_c = None
    avg_d = None; err_d = None
    gpu = torch.cuda.current_device(); device_properties = torch.cuda.get_device_properties(gpu)
    cu_num = device_properties.multi_processor_count
    if (dtype == dtypes.bf16 and quantDtype == dtypes.i8 and bias is not None and cu_num == 80):
        weightshuffle_asm = shuffle_weight(weight, layout=(32, 16))
        bias_f32 = bias.to(dtypes.fp32)
        d, avg_d = run_gemm_asm(x, weightshuffle_asm, x_scale, w_scale, bias_f32, dtype)
        if d is not None:
            err_d = checkAllclose(a, d, msg="asm: ", rtol=1e-2, atol=1e-2)
        else:
            avg_d = None
    return { "ck us": avg_b, "ck err": err_b, "ck bpreshuffle us": avg_c, "ck bpreshuffle err": err_c, "asm us": avg_d, "asm err": err_d }

def test_skinny_gemm(dtype, m, n, k, quantDtype=dtypes.fp8, cu_count=80):
    dim = (m, n, k)
    x = torch.randn((m, k), dtype=dtype, device="cuda")
    weight = torch.randn((n, k), dtype=dtype, device="cuda")
    x, x_scale = aiter.per_tensor_quant(x, quant_dtype=quantDtype)
    weight, w_scale = aiter.per_tensor_quant(weight, quant_dtype=quantDtype)
    bias = None
    a, avg_a = run_torch(x, weight, x_scale, w_scale, bias, dtype)
    if m <= 2:
        b, avg_b = run_gemm_skinny(x, weight, x_scale, w_scale, None, dtype, cu_count)
    else:
        b, avg_b = run_gemm_ck(x, weight, x_scale, w_scale, bias, dtype)
    msg = f"[perf] dim: {str(dim):<20} dtype: {dtype}, quantDtype: {quantDtype}, torch avg: {avg_a:<8.2f} us, skinny_gemm avg: {avg_b:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
    checkAllclose(a, b, msg="a,b: " + msg, rtol=1e-2, atol=0.01)


@perftest(num_iters=TEST_NUM_ITERS, needTrace=False, num_warmup=10, testGraph=False, num_rotate_args=0)
def run_torch_scaled_mm(x, weight, x_scale, w_scale, bias=None, out_dtype=dtypes.bf16):
    result = torch._scaled_mm(x, weight.T, out_dtype=out_dtype, scale_a=x_scale, scale_b=w_scale.T)
    if isinstance(result, tuple):
        output = result[0]
    else:
        output = result
    if bias is not None:
        output = output + bias.to(output.dtype)
    return output

# @lru_cache(maxsize=1)
# def init_hipblas():
#     hipb_create_extension()

# @perftest(num_iters=TEST_NUM_ITERS, needTrace=False, num_warmup=10, testGraph=False, num_rotate_args=0)
# def run_gemm_hipbmm(x, weight_shuffled, x_scale, w_scale, bias=None, dtype=dtypes.bf16):
#     weight_shuffled_transposed = weight_shuffled.t()
#     if w_scale is not None:
#         w_scale = w_scale.t()
#     return hipb_mm(
#         x,
#         weight_shuffled_transposed,
#         solution_index=-1,
#         bias=bias,
#         out_dtype=dtype,
#         scaleA=x_scale,
#         scaleB=w_scale,
#         scaleOut=None,
#         bpreshuffle=True,
#     )

@benchmark()
def benchmark_ptpc_kernels(dtype, m, n, k, preshuffle: bool):
    if dtype != dtypes.bf16:
        return {}
    quantDtype = dtypes.fp8
    x_orig = torch.randn((m, k), dtype=dtype, device="cuda")
    weight_orig = torch.randn((n, k), dtype=dtype, device="cuda")
    x, x_scale = aiter.pertoken_quant(x_orig, quant_dtype=quantDtype)
    weight, w_scale = aiter.pertoken_quant(weight_orig, quant_dtype=quantDtype)
    bias = torch.rand([1, n], dtype=dtype, device="cuda")
    ref_out, _ = run_torch(x, weight, x_scale, w_scale, bias, dtype)
    
    us_aiter = float('inf')
    err_aiter = 1.0

    try:
        if preshuffle:
            if n % 16 == 0 and k % 32 == 0:
                weight_shuffled = shuffle_weight(weight, layout=(16, 16))
                aiter_out_no_bias, us_aiter = run_gemm_ck_bpreshuffle(x, weight_shuffled, x_scale, w_scale, None, dtype)
                aiter_out = aiter_out_no_bias + bias
                err_aiter = checkAllclose(ref_out, aiter_out, rtol=1e-2, atol=1e-2)
        else: # not preshuffle
            aiter_out, us_aiter = run_gemm_ck(x, weight, x_scale, w_scale, bias, dtype)
            err_aiter = checkAllclose(ref_out, aiter_out, rtol=1e-2, atol=1e-2)
    except Exception as e:
        print(f"    -> aiter kernel execution failed: {e}")
    
    # calling torch._scaled_mm and Record scaled_mm time
    us_torch_scaled_mm = float('inf')
    err_scaled_mm = 1.0
    try:
        scaled_mm_out, us_torch_scaled_mm = run_torch_scaled_mm(x, weight, x_scale, w_scale, bias, out_dtype=dtype)
        err_scaled_mm = checkAllclose(ref_out, scaled_mm_out, rtol=1e-2, atol=1e-2)
    except Exception as e:
        print(f"    -> torch._scaled_mm execution failed: {e}")


    # us_hipbmm = float('inf')
    # err_hipbmm = 1.0
    # if preshuffle and get_gfx() == "gfx942":
    #     try:
    #        init_hipblas() 
    #        weight_shuffled_for_hipbmm = shuffle_weight(weight, layout=(16, 16))
    #        hipbmm_out, us_hipbmm = run_gemm_hipbmm(x, weight_shuffled_for_hipbmm, x_scale, w_scale, bias, dtype)
    #        err_hipbmm = checkAllclose(ref_out, hipbmm_out, rtol=1e-2, atol=1e-2)
    #     except Exception as e:
    #        print(f"    -> hipbmm kernel execution failed: {e}")

    return {
        "M": m, "N": n, "K": k, "dtype": "bf16", "preshuffle": preshuffle,
        "us_aiter": us_aiter,
        "us_torch_scaled_mm": us_torch_scaled_mm,
        # "us_hipbmm": us_hipbmm,
        "err_aiter": err_aiter,
        "err_scaled_mm": err_scaled_mm,
        # "err_hipbmm": err_hipbmm,
    }


if __name__ == "__main__":
    # Integrate command line parameters of the two modes
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description="PTPC GEMM Kernel Benchmark (Dual Mode)")
    # Mode 1: List mode (for tuned and untuned in parallel)
    parser.add_argument("--start-index", type=int, default=0, help="List Mode: Starting Task Index")
    parser.add_argument("--end-index", type=int, default=None, help="List Mode: End Task Index")
    parser.add_argument("--task-set", type=str, default="all", choices=["all", "true", "false"], help="List mode: which set of preshuffle tasks to execute (all, true, false)")

    # Mode 2: Single-task mode (for untuned_false)
    parser.add_argument("-m", type=int, default=None, help="Single-task mode: M dimension")
    parser.add_argument("-nk", type=dtypes.str2tuple, default=None, help="Single-task mode: N,K dimensions, e.g., 4096,512")
    parser.add_argument("--preshuffle", type=lambda x: (str(x).lower() == 'true'), default=None, help="Single-task mode: Preshuffle status (True/False)")
    parser.add_argument("--output-file", type=str, default=None, help="Single task mode: The CSV file name to append the results to")
    args = parser.parse_args()

    # l_m in tuned scenario
    l_m_tuned = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 264, 272, 280, 288, 296, 304, 312, 320, 328, 336, 344, 352, 360, 368, 376, 384, 392, 400, 408, 416, 424, 432, 440, 448, 456, 464, 472, 480, 488, 496, 504, 512, 1024, 8192]
    # l_m in untuned scenario
    l_m_untuned = [10000, 16384, 20480, 32768, 65536, 128000, 131072, 260000, 262144, 1048576]

    nk_to_model = {
        # Llama-4-Maverick-17B-128E-Instruct-FP8 (tp=8)
        (896, 5120): "Llama-4-Maverick-17B-128E-Instruct-FP8_tp8", 
        (4096, 640): "Llama-4-Maverick-17B-128E-Instruct-FP8_tp8", 
        (5120, 640): "Llama-4-Maverick-17B-128E-Instruct-FP8_tp8", 
        (5120, 2048): "Llama-4-Maverick-17B-128E-Instruct-FP8_tp8",

        # Llama-4-Maverick-17B-128E-Instruct-FP8 (tp=4)
        (1024, 5632): "Llama-4-Maverick-17B-128E-Instruct-FP8_tp4", 
        (1056, 1408): "Llama-4-Maverick-17B-128E-Instruct-FP8_tp4", 
        (1280, 4096): "Llama-4-Maverick-17B-128E-Instruct-FP8_tp4", 
        (1408, 352): "Llama-4-Maverick-17B-128E-Instruct-FP8_tp4", 
        (1408, 1408): "Llama-4-Maverick-17B-128E-Instruct-FP8_tp4", 
        (4096, 1024): "Llama-4-Maverick-17B-128E-Instruct-FP8_tp4",

        # Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic (tp=8)
        (1536, 2048): "Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic_tp8", 
        (1792, 6144): "Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic_tp8", 
        (2048, 1536): "Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic_tp8", 
        (2048, 6144): "Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic_tp8", 
        (6144, 768): "Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic_tp8", 
        (6144, 1024): "Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic_tp8",

        # Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic (tp=4)
        (3072, 4096): "Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic_tp4", 
        (3584, 6144): "Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic_tp4", 
        (4096, 3072): "Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic_tp4", 
        (6144, 1536): "Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic_tp4", 
        (6144, 2048): "Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic_tp4",

        # deepseek-r1-FP8-Dynamic (tp=8)
        (192, 7168): "deepseek-r1-FP8-Dynamic_tp8", 
        (512, 7168): "deepseek-r1-FP8-Dynamic_tp8", 
        (7168, 256): "deepseek-r1-FP8-Dynamic_tp8", 
        (7168, 2304): "deepseek-r1-FP8-Dynamic_tp8",
    }

    l_nk = list(nk_to_model.keys())

    # Determine which mode to enter based on the incoming parameters

    # Mode 2: Single task mode
    if args.m is not None and args.nk is not None and args.preshuffle is not None:
        m, n, k, preshuffle = args.m, args.nk[0], args.nk[1], args.preshuffle
        output_filename = args.output_file if args.output_file else "ptpc_results_single.csv"
        
        print(f"--> [Single Task Mode] Testing: (M={m}, N={n}, K={k}, Preshuffle={preshuffle})")
        res = benchmark_ptpc_kernels(dtypes.bf16, m, n, k, preshuffle)
        if res:
            res['Model'] = nk_to_model.get((n, k), "Unknown")
            df = pd.DataFrame([res])
            # 'us_hipbmm'  'err_hipbmm'
            final_columns = ['Model', 'M', 'N', 'K', 'dtype', 'preshuffle', 'us_aiter', 'us_torch_scaled_mm', 'err_aiter', 'err_scaled_mm']
            df = df[final_columns]
            header = not os.path.exists(output_filename)
            df.to_csv(output_filename, mode='a', header=header, index=False, float_format='%.4f')
            print(f"    -> Result appended to {output_filename}")
        else:
            print(f"    -> Task failed or skipped.")

    # Mode 1: List Mode
    else:
        # Determine whether to use tuned or untuned M list
        # If the environment variable AITER_REBUILD exists, it is assumed to be tuned mode
        is_tuned_mode = 'AITER_REBUILD' in os.environ
        l_m = l_m_tuned if is_tuned_mode else l_m_untuned
        
        if args.task_set == "all": preshuffle_states = [True, False]
        elif args.task_set == "true": preshuffle_states = [True]
        else: preshuffle_states = [False]
        
        all_tasks = list(itertools.product(l_m, l_nk, preshuffle_states))
        start_index = args.start_index
        end_index = args.end_index if args.end_index is not None else len(all_tasks)
        tasks_to_process = all_tasks[start_index:end_index]

        print(f"--> [List Mode] Start PTPC kernel evaluation [task block: {start_index} -> {end_index}]")
        results = []
        total_tasks_in_chunk = len(tasks_to_process)

        for i, (m, (n, k), preshuffle) in enumerate(tasks_to_process):
            print(f"--> Testing ({i+1}/{total_tasks_in_chunk} in chunk): (M={m}, N={n}, K={k}, Preshuffle={preshuffle})")
            res = benchmark_ptpc_kernels(dtypes.bf16, m, n, k, preshuffle)
            if res:
                res['Model'] = nk_to_model.get((n, k), "Unknown")
                results.append(res)
        
        df = pd.DataFrame(results)
        if not df.empty:
            final_columns = ['Model', 'M', 'N', 'K', 'dtype', 'preshuffle', 'us_aiter', 'us_torch_scaled_mm', 'err_aiter', 'err_scaled_mm']
            df = df[final_columns]
        
        output_filename = f"ptpc_results_part_{start_index}_{end_index}.csv"
        df.to_csv(output_filename, index=False, float_format='%.4f')
        print(f"The block evaluation results have been saved to {output_filename}")