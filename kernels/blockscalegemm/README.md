## Motivation
The motivation is to understand the performance benefit or using a preshuffled block scaled gemm over normal block scaled gemm. Both are CK implementation. 

First the AITER has provided tuning guide : 

* Normal block scaled gemm https://github.com/ROCm/aiter/tree/main/csrc/ck_gemm_a8w8_blockscale 
* Preshuffled block scaled gemmhttps://github.com/ROCm/aiter/tree/main/csrc/ck_gemm_a8w8_blockscale_bpreshuffle 

Known to date the models (not exhaustive list) that are using block scaled gemm: 

1. DeepSeek-V3/DeepSeek-R1 
2. Qwen/Qwen3-Next-80B-A3B-Instruct-FP8 
3. Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 
4. Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 
5. Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 
6. Qwen/Qwen3-235B-A22B-Thinking-2507-FP8 
7. Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 
8. Qwen/Qwen3-30B-A3B-Thinking-2507-FP8 
9. Qwen/Qwen3-4B-Thinking-2507-FP8 
10. Qwen/Qwen3-4B-Instruct-2507-FP8

The models that that are of interested: 

1. DeepSeek-V3/DeepSeek-R1 
2. Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8 
3. Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 

The configurations that are interested in TP8 (8 GPUs) 

## Tuning Procedure

1. The shapes of the block scaled gemm by running: `python3 generate_mnk_shape_csv.py`

2. Copy the shapes into:
    * `aiter/configs/a8w8_blockscale_bpreshuffle_untuned_gemm.csv`
    * `aiter/configs/a8w8_blockscale_untuned_gemm.csv`

3. Run `AITER_REBUILD=1 python3 csrc/ck_gemm_a8w8_blockscale_bpreshuffle/gemm_a8w8_blockscale_bpreshuffle_tune.py -i aiter/configs/a8w8_blockscale_bpreshuffle_untuned_gemm.csv -o aiter/configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv`

4. Run `AITER_REBUILD=1 python3 csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py -i aiter/configs/a8w8_blockscale_untuned_gemm.csv -o aiter/configs/a8w8_blockscale_tuned_gemm.csv`


5. NOTE: At the time of writing this tests, the tuned csv files contain duplicate for entry (cu_num,M,N,K).

6. Update the `CSV_FILE` in `dedup.py` to point to the tuned csv file. Run the following script to deduplicate the file. `python3 dedup.py`

7. (Optional) You can run the following script to check at kernel level to find out the speed up of each entry (cu_num,M,N,K). Update `df_bpreshuffle` and `df_regular` to point to the tuned files. Then run `python3 compare_raw_data.py`. A csv file will be generated.

8. Run the API level tests: `AITER_REBUILD=1 python3 test_gemm_a8w8_blockscale.py`
  - This is a file copied from the aiter repo and modified to test the shapes of interested models.
  - It is modified to generate results into a csv file for further analysis.

## Findings:
- On AITER commit: `a7d3bf8cd47afbaf6a6133c1f12e3b01d2c27b0e`, on MI300X:
- Tuned shapes, bpreshuffle block scaled GEMM is around 16% better than normal block scaled GEMM on average, across all cases.
    | Model | Multiplier | Boost |
    |-------|------------|-------|
    | DeepSeek-R1 | 1.167129029x | 16.7% boost |
    | Qwen3-235B-FP8 (TP8) | 1.165692083x | 16.6% boost |
    | Qwen3-Coder-480B-FP8 (TP8) | 1.174530533x | 17% boost |
    | Qwen3-Coder-480B-FP8 (TP4) | 1.17128822x | 17% boost |

- Untuned large shapes bpreshuffle block scaled GEMM around 2 times better than normal block scaled GEMM on average across all cases.
  - `l_m = [10000, 16384, 20480, 32768, 65536, 128000, 131072, 260000, 262144]`


NOTE: `all cases` only refers to the M,N,K-shapes that are of interest in current experiments.