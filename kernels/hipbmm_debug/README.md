# Note:

# Script 1: `test_gemm_a8w8.py`

To run with the configuration where the bias addition is fused with gemm op: `python3 test_gemm_a8w8.py -q fp8 -d bf16 --fused_bias`. Otherwise, run `python3 test_gemm_a8w8.py -q fp8 -d bf16`

# Script 2: `test_a16w16.py`

To run with the configuration where the bias addition is fused with gemm op:  `python3 test_gemm_a16w16.py -t normal -d bf16 -b -o bf16 -fb` 
Otherwise, run `python3 test_gemm_a16w16.py -t normal -d bf16 -b -o bf16`
