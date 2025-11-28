# Motivation
The motivation is to understand the performance benefit or using a ck_gemm_a8w8(Abbreviated as CK) and ck_gemm_a8w8_bpreshuffle(Abbreviated as shuffle)  over  torch_scaled_mm. 

First the AITER has provided tuning guide :
- `csrc/ck_gemm_a8w8`: https://github.com/ROCm/aiter/tree/main/csrc/ck_gemm_a8w8
- `csrc/ck_gemm_a8w8_bpreshuffle`: https://github.com/ROCm/aiter/tree/main/csrc/ck_gemm_a8w8_bpreshuffle

Mainly tuned for the following models (not exhaustive list) :
- meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 (tp=4,8)
- RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic (tp=4,8)
- EmbeddedLLM/Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic (tp=4,8)
- EmbeddedLLM/deepseek-r1-FP8-Dynamic(tp=8)

The configurations that are interested in TP8 (8 GPUs)

Based on the latest format of `aiter/configs/a8w8_bpreshuffle_untuned_gemm.csv`, we adjusted the logic of our `generate_ptpc_shapes_csv.py` file.

Running `python generate_ptpc_shapes_csv.py bpreshuffle` yields the untuned file for `bpreshuffle`,

Running `python generate_ptpc_shapes_csv.py ck` yields the untuned file for `ck`.

Then continue with the Tuning Procedure below.

## Tuning Procedure

The list of M-values in this script is carefully derived from the vLLM default cuda graph capture size 
and the heuristic of AITER PTPC CK gemm.

- Heuristic of AITER PTPC CK GEMM can be derived from https://github.com/ROCm/aiter/blob/main/csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle.cu and https://github.com/ROCm/aiter/blob/be1e8ae23dbf342263130a3d04063bbfdf9a2fba/csrc/py_itfs_cu/gemm_common.cu

The shapes of the block scaled gemm by running: `python3 generate_mnk_shape_csv.py`

Copy the shapes into:

`aiter/configs/a8w8_untuned_gemm.csv`
`aiter/configs/a8w8_bpreshuffle_untuned_gemm.csv`

Run `AITER_REBUILD=1 python3 csrc/ck_gemm_a8w8/gemm_a8w8_tune.py -i aiter/configs/a8w8_untuned_gemm.csv -o aiter/configs/a8w8_tuned_gemm.csv`

Run `AITER_REBUILD=1 python3 csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py -i aiter/configs/a8w8_bpreshuffle_untuned_gemm.csv -o aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv`

NOTE: At the time of writing this tests, the tuned csv files contain duplicate for entry `(cu_num,M,N,K)`.

Update the `CSV_FILE` in `dedup.py` to point to the tuned csv file. Run the following script to deduplicate the file. `python3 dedup.py`

(Optional) You can run the following script to check at kernel level to find out the speed up of each entry `(cu_num,M,N,K)`. Update df_bpreshuffle and df_regular to point to the tuned files. Then run `python3 compare_raw_data.py`. A csv file will be generated.

Run the API level tests: `AITER_REBUILD=1 python3 test_gemm_a8w8_ptpc.py`

NOTE: After tuning, you should at least run once with this flag `AITER_REBUILD=1`. Either `AITER_REBUILD=1 python3 test_gemm_a8w8_ptpc.py` or (if you skip running the test script) `AITER_REBUILD=1 vllm serve ...`

This is a file copied from the aiter repo and modified to test the shapes of interested models.
It is modified to generate results into a csv file for further analysis.

## Findings:
On AITER commit: `a7d3bf8cd47afbaf6a6133c1f12e3b01d2c27b0e`, on MI300X:

Tuned shapes, bpreshuffle block scaled GEMM is around `16%` better than normal block scaled GEMM on average, across all cases.

- Table 1: noshuffle vs scaledmm

    | Model | Multiplier | Boost |
    |-------|------------|-------|
    | Llama-4-Maverick-17B-128E-Instruct-FP8 (tp=8) | 1.542153929 | 15.4% boost |
    | Llama-4-Maverick-17B-128E-Instruct-FP8 (tp=4) | 1.677547653 | 16.7% boost |
    | Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic (tp=8) | 1.37099678 | 13.7% boost |
    | Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic (tp=4) | 1.173175067 | 11.7% boost |
    | deepseek-r1-FP8-Dynamic (tp=8) | 1.309801496 | 13.1% boost |

- Table 2: shuffle vs scaledmm

    | Model | Multiplier | Boost |
    |-------|------------|-------|
    | Llama-4-Maverick-17B-128E-Instruct-FP8 (tp=8) | 1.758673823 | 17.6% boost |
    | Llama-4-Maverick-17B-128E-Instruct-FP8 (tp=4) | 1.697543574 | 17.0% boost |
    | Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic (tp=8) | 1.673015491 | 16.7% boost |
    | Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic (tp=4) | 1.48122014 | 14.8% boost |
    | deepseek-r1-FP8-Dynamic (tp=8) | 1.49825552 | 15.0% boost |

### Finding 1:
Tuned for vLLM default CUDAGraph Capture Sizes: `l_m = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 264, 272, 280, 288, 296, 304, 312, 320, 328, 336, 344, 352, 360, 368, 376, 384, 392, 400, 408, 416, 424, 432, 440, 448, 456, 464, 472, 480, 488, 496, 504, 512, 1024, 8192]`

`1079` out of `1104` (MNK-shape show speed up).

`25` out of `1104` shapes shows slowdown of average `1%` and up to `3.4%`

### Finding 2:
Untuned large shapes bpreshuffle block scaled GEMM around 2 times better than normal block scaled GEMM on average across all cases.

`l_m = [10000, 16384, 20480, 32768, 65536, 128000, 131072, 260000, 262144]`
and there seems to have some numerical issue for size 260000
```csv
dtype,m,n,k,preshuffle,us,failed
torch.bfloat16,260000,7168,256,True,2608.56526,0.00012249581050127745
torch.bfloat16,260000,7168,2048,True,11240.853494949492,0.00012307692668400705
torch.bfloat16,260000,4608,7168,True,25362.158949999986,0.00012307692668400705
torch.bfloat16,260000,7168,2304,True,12459.498252525238,0.00012307692668400705
torch.bfloat16,260000,6144,1536,True,7497.625349999996,0.00012307692668400705
torch.bfloat16,260000,6144,3072,True,13952.130918367338,0.00012307692668400705
```
NOTE: all cases only refers to the M,N,K-shapes that are of interest in current experiments.