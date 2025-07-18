#!/bin/bash
PORT=8000
SEED=0
CONCURRENCY=64
NREQUESTS=$(($CONCURRENCY * 10))
ISL=3584
OSL=1024
python3 ../../../sglang/python/sglang/bench_serving.py --backend vllm \
--dataset-name random \
--num-prompts ${NREQUESTS} \
--random-input ${ISL} \
--random-output ${OSL} \
--random-range-ratio 1.0 \
--seed ${SEED} \
--max-concurrency ${CONCURRENCY} --warmup-requests ${CONCURRENCY} --port ${PORT}\
| tee sglang_benchmark_vllm_random_isl${ISL}_osl${OSL}_con${CONCURRENCY}-w8a8_utils.log