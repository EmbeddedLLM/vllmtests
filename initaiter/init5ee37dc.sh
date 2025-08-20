#!/bin/bash
python3 aiter5ee37dc/test_batched_gemm_a8w8.py
python3 aiter5ee37dc/test_gemm_a8w8_blockscale.py
python3 aiter5ee37dc/test_gemm_a8w8.py
python3 aiter5ee37dc/test_gemm.py
python3 aiter5ee37dc/test_kvcache.py
python3 aiter5ee37dc/test_layernorm2d.py
python3 aiter5ee37dc/test_mha_varlen.py
python3 aiter5ee37dc/test_mha.py
python3 aiter5ee37dc/test_mla.py
python3 aiter5ee37dc/test_moe_2stage.py
python3 aiter5ee37dc/test_moe_blockscale.py
python3 aiter5ee37dc/test_moe_ep.py --token 1 --hidden_dim 4096 --inter_dim 1024 --expert 256 --topk 5 -ep 8
python3 aiter5ee37dc/test_moe_sorting.py -m 64
python3 aiter5ee37dc/test_moe_tkw1.py -m 1
python3 aiter5ee37dc/test_moe.py -m 128
python3 aiter5ee37dc/test_moeTopkSoftmax.py --token 1 -m 1 -e 64
python3 aiter5ee37dc/test_pa_v1.py
python3 aiter5ee37dc/test_pa.py -c 7 -n 4,1
python3 aiter5ee37dc/test_quant.py -m 1 -n 2
python3 aiter5ee37dc/test_rmsnorm2d.py -m 1 -n 4096
python3 aiter5ee37dc/test_rope.py
python3 aiter5ee37dc/test_sampling.py