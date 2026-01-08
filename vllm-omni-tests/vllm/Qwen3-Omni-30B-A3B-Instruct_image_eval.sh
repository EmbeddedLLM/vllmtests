#!/bin/bash

# Script Description: Image Evaluation using Mistral Evals (MathVista)
# How to use: ./Qwen3-Omni-30B-A3B-Instruct_image_eval.sh [1|2|3|4]
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
PORT=8001
SERVED_NAME="qwen-omni"

# Kill previous instance
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

SERVER_LOG="server_image_${LOG_SUFFIX}.log"
EVAL_LOG="eval_mistral_${LOG_SUFFIX}.log"

# Step 1: Start the Server
echo "Starting vLLM Server...: $SERVER_LOG"
nohup vllm serve $MODEL_PATH \
    --served-model-name $SERVED_NAME \
    --tensor-parallel-size $TP_SIZE \
    --trust-remote-code \
    --port $PORT \
    --gpu-memory-utilization 0.8 \
    --swap-space 16 \
    --max-model-len 32768 \
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

# Step 3: Prepare Mistral Evals Environment
TEST_ROOT="/app"
MISTRAL_DIR="$TEST_ROOT/mistral-evals"

# Ensure directory exists
if [ ! -d "$TEST_ROOT" ]; then
    mkdir -p $TEST_ROOT
fi

# Clone mistral-evals if not present
if [ ! -d "$MISTRAL_DIR" ]; then
    echo ">>> Cloning mistral-evals to $MISTRAL_DIR..."
    git clone https://github.com/mistralai/mistral-evals.git $MISTRAL_DIR
else
    echo ">>> mistral-evals already exists at $MISTRAL_DIR"
fi

# Install dependencies
echo ">>> Installing dependencies (fire==0.6.0)..."
pip install fire==0.6.0

# Step 4: Run Mistral Eval (MathVista)
echo ">>> [1/1] Running Mistral Eval (mathvista)..."

# Switch to mistral-evals directory to run
cd $MISTRAL_DIR

# Define output path relative to where we are executing
OUTPUT_DIR="../results_mistral_${LOG_SUFFIX}"
LOG_FILE="../eval_mistral_${LOG_SUFFIX}.log"

# Run the python command
python3 -m eval.run eval_vllm \
    --model_name $SERVED_NAME \
    --url http://127.0.0.1:$PORT \
    --output_dir $OUTPUT_DIR \
    --eval_name "mathvista" \
    > $LOG_FILE 2>&1

echo "Image Eval Finished. Check $TEST_ROOT/eval_mistral_${LOG_SUFFIX}.log"

echo ">>> All Tests Done! Stopping server..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null


# Show last few lines of the log to see if there is a score
tail -n 20 $LOG_FILE

rm -rf $OUTPUT_DIR
rm -rf $LOG_FILE
cd ../
rm -rf $MISTRAL_DIR