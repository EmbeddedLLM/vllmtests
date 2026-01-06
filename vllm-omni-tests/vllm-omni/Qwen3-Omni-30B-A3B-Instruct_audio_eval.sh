#!/bin/bash

# Script Description: Audio Evaluation using lmms-eval (VoiceBench)
# How to use: ./Qwen3-Omni-30B-A3B-Instruct_audio_eval.sh [1|2|3|4]
# 1: Eager + AITER
# 2: Non-Eager (Graph) + AITER
# 3: Eager + No AITER
# 4: Non-Eager (Graph) + No AITER

MODE=$1

WORK_DIR=$(pwd)

# Configuration area
# Using the local path to avoid re-downloading from HF
MODEL_PATH="/app/model/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/26291f793822fb6be9555850f06dfe95f2d7e695"

export HIP_VISIBLE_DEVICES=0,1
TP_SIZE=2

# Parameter check
if [ -z "$MODE" ]; then
    echo "Error: Please specify mode 1-4"
    exit 1
fi

# Mode selection logic
# Note: Audio eval runs via python library, so we pass arguments to model_args
case $MODE in
    1)
        echo ">>> [Mode 1]: Eager + AITER (On)"
        export VLLM_ROCM_USE_AITER=1
        EAGER_BOOL="True"
        LOG_SUFFIX="eager_aiter"
        ;;
    2)
        echo ">>> [Mode 2]: Non-Eager + AITER (On)"
        export VLLM_ROCM_USE_AITER=1
        EAGER_BOOL="False"
        LOG_SUFFIX="noneager_aiter"
        ;;
    3)
        echo ">>> [Mode 3]: Eager + No AITER (Off)"
        export VLLM_ROCM_USE_AITER=0
        EAGER_BOOL="True"
        LOG_SUFFIX="eager_noaiter"
        ;;
    4)
        echo ">>> [Mode 4]: Non-Eager + No AITER (Off)"
        export VLLM_ROCM_USE_AITER=0
        EAGER_BOOL="False"
        LOG_SUFFIX="noneager_noaiter"
        ;;
    *)
        echo "Error: The pattern must be 1, 2, 3, or 4."
        exit 1
        ;;
esac

# Step 1: Prepare Environment (Outside vllmtests, in /app/)
LMMS_DIR="/app/lmms-eval"

# Check if directory exists
if [ ! -d "$LMMS_DIR" ]; then
    echo ">>> Cloning lmms-eval (openaiapi branch) to $LMMS_DIR..."
    git clone -b openaiapi https://github.com/EmbeddedLLM/lmms-eval.git $LMMS_DIR
else
    echo ">>> lmms-eval already exists at $LMMS_DIR"
fi

# Step 2: Install Dependencies (As requested)
echo ">>> Setting up Audio dependencies..."
cd $LMMS_DIR

# 1. Install lmms-eval
pip install -e .

# 2. Uninstall torchcodec to prevent conflict
pip uninstall -y torchcodec

# 3. Install audio libraries
pip install soundfile librosa

# 4. Downgrade datasets (crucial for soundfile support)
pip install "datasets<3.0.0"


# Step 3: Run Evaluation
# CRITICAL: Go back to the directory where we started the script to save logs there
cd $WORK_DIR

echo ">>> [1/1] Running lmms_eval (voicebench_openbookqa)..."
echo ">>> Log file will be in: $(pwd)/eval_audio_${LOG_SUFFIX}.log"

# Note: Using 'model=$MODEL_PATH' to use local weights
# 'enforce_eager' is controlled by $EAGER_BOOL
nohup python3 -m lmms_eval \
    --model vllm \
    --tasks voicebench_openbookqa \
    --model_args model=$MODEL_PATH,tensor_parallel_size=$TP_SIZE,gpu_memory_utilization=0.9,enforce_eager=$EAGER_BOOL,trust_remote_code=True \
    --output_path "results_audio_${LOG_SUFFIX}" \
    > "eval_audio_${LOG_SUFFIX}.log" 2>&1 &

EVAL_PID=$!
echo "Evaluation PID: $EVAL_PID"
echo "To follow logs: tail -f eval_audio_${LOG_SUFFIX}.log"

# Optional: Wait for completion if you don't want to return immediately
wait $EVAL_PID

echo "Audio Eval Finished."
tail -n 20 "eval_audio_${LOG_SUFFIX}.log"