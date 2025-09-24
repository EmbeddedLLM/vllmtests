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
  - Tuned for vLLM default CUDAGraph Capture Sizes: `l_m = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 264, 272, 280, 288, 296, 304, 312, 320, 328, 336, 344, 352, 360, 368, 376, 384, 392, 400, 408, 416, 424, 432, 440, 448, 456, 464, 472, 480, 488, 496, 504, 512, 1024, 8192]`

  - 1079 out of 1104 (MNK-shape show speed up). 
  - 25 out of 1104 shapes shows slowdown of average 1% and up to 3.4% 

- Untuned large shapes bpreshuffle block scaled GEMM around 2 times better than normal block scaled GEMM on average across all cases.
  - `l_m = [10000, 16384, 20480, 32768, 65536, 128000, 131072, 260000, 262144]`
  - and there seems to have some numerical issue for size 260000
    ```csv
    dtype,m,n,k,preshuffle,us,failed
    torch.bfloat16,260000,7168,256,True,2608.56526,0.00012249581050127745
    torch.bfloat16,260000,7168,2048,True,11240.853494949492,0.00012307692668400705
    torch.bfloat16,260000,4608,7168,True,25362.158949999986,0.00012307692668400705
    torch.bfloat16,260000,7168,2304,True,12459.498252525238,0.00012307692668400705
    torch.bfloat16,260000,6144,1536,True,7497.625349999996,0.00012307692668400705
    torch.bfloat16,260000,6144,3072,True,13952.130918367338,0.00012307692668400705
    ```


NOTE: `all cases` only refers to the M,N,K-shapes that are of interest in current experiments.