python3 ../../llama_fp8_03122025_sync_upstream2/benchmarks/benchmark_serving.py \
    --backend vllm \
    --model EmbeddedLLM/Qwen2.5-1.5B-Instruct-FP8-Dynamic \
    --dataset-name random \
    --max-concurrency 64 \
    --num-prompts 64 \
    --random-input-len 10 \
    --random-output-len 100 \
    --port 8733 \
    --profile \
    > send_load.log
