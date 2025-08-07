#!/bin/bash

pushd ./mistral-evals

python3 -m eval.run eval_vllm \
        --model_name Qwen/Qwen2.5-VL-7B-Instruct \
        --url http://0.0.0.0:7899 \
        --output_dir ./chartqa \
        --eval_name "chartqa" \
        --max_new_tokens 1024 > lmeval_server_qwen2_5-vl-7b-instruct-mrope-upstream-v1.log 2>&1

popd