# Note:

# Script 1: `test_gemm_a8w8.py`

To run with the configuration where the bias addition is fused with gemm op: `python3 test_gemm_a8w8.py -q fp8 -d bf16 --fused_bias`. Otherwise, run `python3 test_gemm_a8w8.py -q fp8 -d bf16`

## Validated on MI300X, using rocm/vllm-dev:nightly + aiter's latest commit 


Accuracy report: Percentage of elements mismatched increases from 0.004 % (no fused bias) to 0.5% error rate (fused bias).

(no fused bias)
```
| # | dtype | m | n | k | quantDtype | ... | ck bpreshuffle err | asm us | asm err | hipmm bpreshuffle us | hipmm bpreshuffle err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | torch.bfloat16 | 16 | 7424 | 8192 | torch.float8_e4m3fnuz | ... | 0.000008 | None | None | 18.535242 | 0.000008 |
| 1 | torch.bfloat16 | 32 | 7424 | 8192 | torch.float8_e4m3fnuz | ... | 0.000038 | None | None | 20.556840 | 0.000038 |
| 2 | torch.bfloat16 | 48 | 7424 | 8192 | torch.float8_e4m3fnuz | ... | 0.000025 | None | None | 25.910948 | 0.000025 |
| 3 | torch.bfloat16 | 64 | 7424 | 8192 | torch.float8_e4m3fnuz | ... | 0.000011 | None | None | 26.486979 | 0.000011 |
| 4 | torch.bfloat16 | 4096 | 7424 | 8192 | torch.float8_e4m3fnuz | ... | 0.000043 | None | None | 451.043682 | 0.000043 |
| 5 | torch.bfloat16 | 5120 | 7424 | 8192 | torch.float8_e4m3fnuz | ... | 0.000041 | None | None | 542.292120 | 0.000041 |
| 6 | torch.bfloat16 | 8192 | 7424 | 8192 | torch.float8_e4m3fnuz | ... | 0.000042 | None | None | 855.101822 | 0.000042 |
```

(fused bias)

```

| [aiter] summary: | dtype | m | n | k | quantDtype | ... | ck bpreshuffle err | asm us | asm err | hipmm bpreshuffle us | hipmm bpreshuffle err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | torch.bfloat16 | 16 | 7424 | 8192 | torch.float8_e4m3fnuz | ... | 0.000025 | None | None | 18.688043 | 0.001962 |
| 1 | torch.bfloat16 | 32 | 7424 | 8192 | torch.float8_e4m3fnuz | ... | 0.000017 | None | None | 20.724505 | 0.001612 |
| 2 | torch.bfloat16 | 48 | 7424 | 8192 | torch.float8_e4m3fnuz | ... | 0.000022 | None | None | 26.435531 | 0.001619 |
| 3 | torch.bfloat16 | 64 | 7424 | 8192 | torch.float8_e4m3fnuz | ... | 0.000011 | None | None | 26.714511 | 0.001766 |
| 4 | torch.bfloat16 | 4096 | 7424 | 8192 | torch.float8_e4m3fnuz | ... | 0.000043 | None | None | 453.944478 | 0.001728 |
| 5 | torch.bfloat16 | 5120 | 7424 | 8192 | torch.float8_e4m3fnuz | ... | 0.000040 | None | None | 547.005796 | 0.001683 |
| 6 | torch.bfloat16 | 8192 | 7424 | 8192 | torch.float8_e4m3fnuz | ... | 0.000042 | None | None | 864.310934 | 0.001727 |
```

# Script 2: `test_a16w16.py`

To run with the configuration where the bias addition is fused with gemm op:  `python3 test_gemm_a16w16.py -t normal -d bf16 -b -o bf16 -fb` 
Otherwise, run `python3 test_gemm_a16w16.py -t normal -d bf16 -b -o bf16`
