#!/bin/bash

# Script Description
# How to use: ./Qwen3-Omni-30B-A3B-Instruct_text_eval.sh [1|2|3|4]
# 1: Eager + AITER
# 2: Non-Eager (Graph) + AITER
# 3: Eager + No AITER
# 4: Non-Eager (Graph) + No AITER


MODE=$1

# Configuration area

MODEL_PATH="/app/model/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695"

# Allow MODEL_PATH override via second argument
if [ -n "$2" ]; then
    MODEL_PATH="$2"
    echo ">>> Using custom MODEL_PATH: $MODEL_PATH"
fi

echo ">>> Using MODEL_PATH: $MODEL_PATH"

export HIP_VISIBLE_DEVICES=0,1
TP_SIZE=2
PORT=6789
SERVED_NAME="qwen-omni"

pkill -9 -f "vllm serve"
sleep 3

# Parameter check
if [ -z "$MODE" ]; then
    echo "Error: Please specify mode 1-4"
    exit 1
fi

# Mode selection logic
case $MODE in
    1)
        echo ">>> [Mode 1]: Eager + AITER (On)"
        export VLLM_ROCM_USE_AITER=1
        EAGER_ARG="--enforce-eager"
        LOG_SUFFIX="eager_aiter"
        ;;
    2)
        echo ">>> [Mode 2]: Non-Eager + AITER (On)"
        export VLLM_ROCM_USE_AITER=1
        EAGER_ARG=""
        LOG_SUFFIX="noneager_aiter"
        ;;
    3)
        echo ">>> [Mode 3]: Eager + No AITER (Off)"
        export VLLM_ROCM_USE_AITER=0
        EAGER_ARG="--enforce-eager"
        LOG_SUFFIX="eager_noaiter"
        ;;
    4)
        echo ">>> [Mode 4]: Non-Eager + No AITER (Off)"
        export VLLM_ROCM_USE_AITER=0
        EAGER_ARG=""
        LOG_SUFFIX="noneager_noaiter"
        ;;
    *)
        echo "Error: The pattern must be 1, 2, 3, or 4."
        exit 1
        ;;
esac

SERVER_LOG="server_${LOG_SUFFIX}.log"
EVAL_LOG="eval_${LOG_SUFFIX}.log"


# Step 1: Start the Server
echo "Starting vLLM Server...: $SERVER_LOG"
nohup vllm serve $MODEL_PATH \
    --served-model-name $SERVED_NAME \
    --tensor-parallel-size $TP_SIZE \
    --trust-remote-code \
    --port $PORT \
    --gpu-memory-utilization 0.8 \
    --swap-space 16 \
    --max-model-len 8192 \
    --disable-log-requests \
    $EAGER_ARG > $SERVER_LOG 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID. Waiting for service to be ready..."


# Step 2: Wait for the service to start
for i in {1..60}; do
    sleep 10
    if curl -s http://127.0.0.1:$PORT/health > /dev/null; then
        echo ">>> Server Ready! Starting Evaluation..."
        break
    fi
    echo "Waiting... ($((i*10))s)"
    
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "CRITICAL ERROR: Server died! Check $SERVER_LOG"
        tail -n 10 $SERVER_LOG
        exit 1
    fi

    if [ $i -eq 60 ]; then
        echo "Error: Timeout waiting for server."
        kill $SERVER_PID
        exit 1
    fi
done


# Step 3: Run lm_eval
echo ">>> [1/2]Running lm_eval (GSM8K)..."
lm_eval \
    --model local-completions \
    --tasks gsm8k \
    --model_args model=$SERVED_NAME,base_url=http://127.0.0.1:$PORT/v1/completions,tokenizer=$MODEL_PATH \
    --batch_size 64 \
    --output_path "results_${LOG_SUFFIX}" \
    > $EVAL_LOG 2>&1
echo "Text Eval Finished. Check $EVAL_LOG"



echo ">>> All Tests Done! Stopping server..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null


echo "GSM8K Result:"
grep "exact_match" $EVAL_LOG || tail -n 10 $EVAL_LOG