#!/bin/bash
python3 ../../fix-specdec-rocm/benchmarks/benchmark_serving.py  \
--backend openai-chat   \
--model Qwen/Qwen2.5-VL-7B-Instruct   \
--endpoint /v1/chat/completions   \
--dataset-name hf   \
--dataset-path lmarena-ai/VisionArena-Chat   \
--hf-split train   \
--num-prompts 1000 \
--max-concurrency 64 \
--port 7899 \
> speedtest_torch_upstream.log 2>&1