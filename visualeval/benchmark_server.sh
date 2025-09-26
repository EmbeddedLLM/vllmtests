#!/bin/bash
vllm bench serve --backend openai-chat \
--endpoint_type openai-chat \
--endpoint /v1/chat/completions \
--model Qwen/Qwen2.5-VL-72B-Instruct \
--dataset-name hf \
--dataset-path lmarena-ai/VisionArena-Chat \
--hf-split train \
--num-prompts 1000 \
--port 6789 \
> speedtest_compilesizes_expandedcudagraphsize_largebatchsize.log 2>&1
# --max-concurrency 64 \