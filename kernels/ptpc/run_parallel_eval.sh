#!/bin/bash

# This is a unified benchmark control script.
# You can switch between different test scenarios by simply modifying the MODE variable below.
#
# Available modes:
# - "tuned": Uses 4 GPUs in parallel and forces a kernel recompile (AITER_REBUILD=1)
# - "untuned_true": Uses 4 GPUs in parallel and runs tasks with preshuffle=True (no freezes)
# - "untuned_false": Uses a single GPU and runs tasks with preshuffle=False one by one (avoids freezes)

MODE="tuned"

# --- Script General Settings ---
PYTHON_SCRIPT="test_gemm_a8w8_ptpc.py"
GPUS_TO_USE=(0 1 2 3)
NUM_GPUS=${#GPUS_TO_USE[@]}
LOCK_FILE_1="/opt/aiter/aiter/jit/build/lock_module_gemm_a8w8"
LOCK_FILE_2="/opt/aiter/aiter/jit/build/module_gemm_a8w8/build/lock"


# Mode 1: TUNED (multi-GPU parallelism, forced recompilation)
if [ "$MODE" == "tuned" ]; then
  echo "[INFO] Running in TUNED mode with ${NUM_GPUS} GPUs."
  # vLLM Graph Capture Size, M value has 69
  TOTAL_TASKS=3450 # 69 * 25 * 2
  CHUNK_SIZE=$(( (TOTAL_TASKS + NUM_GPUS - 1) / NUM_GPUS ))
  
  for i in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_ID=${GPUS_TO_USE[$i]}
    START_INDEX=$(( i * CHUNK_SIZE ))
    END_INDEX=$(( (i + 1) * CHUNK_SIZE ))
    if [ $END_INDEX -gt $TOTAL_TASKS ]; then END_INDEX=$TOTAL_TASKS; fi

    echo "  -> Launching on GPU ${GPU_ID}, task range: [${START_INDEX}, ${END_INDEX})"
    (
      export HIP_VISIBLE_DEVICES=$GPU_ID
      # Key setting AITER_REBUILD=1
      AITER_REBUILD=1 python3 "$PYTHON_SCRIPT" \
        --start-index $START_INDEX \
        --end-index $END_INDEX > eval_gpu_${GPU_ID}.log 2>&1
    ) &
  done
  echo "All TUNED benchmark jobs launched in background. Check eval_gpu_*.log for progress."


# Mode 2: UNTUNED_TRUE (multiple CPUs in parallel, preshuffle=True)
elif [ "$MODE" == "untuned_true" ]; then
  echo "[INFO] Running in UNTUNED_TRUE mode with ${NUM_GPUS} GPUs."
  TOTAL_TASKS=250 # 10 * 25 * 1
  CHUNK_SIZE=$(( (TOTAL_TASKS + NUM_GPUS - 1) / NUM_GPUS ))

  for i in $(seq 0 $((NUM_GPUS - 1))); do
    GPU_ID=${GPUS_TO_USE[$i]}
    START_INDEX=$(( i * CHUNK_SIZE ))
    END_INDEX=$(( (i + 1) * CHUNK_SIZE ))
    if [ $END_INDEX -gt $TOTAL_TASKS ]; then END_INDEX=$TOTAL_TASKS; fi

    echo "  -> Launching on GPU ${GPU_ID}, task range: [${START_INDEX}, ${END_INDEX})"
    (
      export HIP_VISIBLE_DEVICES=$GPU_ID
      python3 "$PYTHON_SCRIPT" \
        --start-index $START_INDEX \
        --end-index $END_INDEX \
        --task-set true > eval_gpu_${GPU_ID}_untuned_true.log 2>&1
    ) &
  done
  echo "All UNTUNED_TRUE jobs launched in background. Check *.log for progress."


# Mode 3: UNTUNED_FALSE (runs each card one by one, avoiding the bug of the aiter operator being stuck)
elif [ "$MODE" == "untuned_false" ]; then
  echo "[INFO] Running in robust UNTUNED_FALSE mode on a single GPU."
  OUTPUT_CSV="ptpc_results_untuned_false.csv"
  
  M_VALUES=(10000 16384 20480 32768 65536 128000 131072 260000 262144 1048576)
  NK_VALUES=( "896,5120" "4096,640" "5120,640" "5120,2048" "1024,5632" "1056,1408" "1280,4096" "1408,352" "1408,1408" "4096,1024" "1536,2048" "1792,6144" "2048,1536" "2048,6144" "6144,768" "6144,1024" "3072,4096" "3584,6144" "4096,3072" "6144,1536" "6144,2048" "192,7168" "512,7168" "7168,256" "7168,2304" )
  PRESHUFFLE_STATE="False"
  
  rm -f "$OUTPUT_CSV"
  TOTAL_TASKS=$(( ${#M_VALUES[@]} * ${#NK_VALUES[@]} ))
  CURRENT_TASK=0

  for m in "${M_VALUES[@]}"; do
    for nk in "${NK_VALUES[@]}"; do
      CURRENT_TASK=$((CURRENT_TASK + 1))
      echo "Running task ${CURRENT_TASK} of ${TOTAL_TASKS}: M=${m}, NK=${nk}"
      
      rm -f "$LOCK_FILE_1" "$LOCK_FILE_2"

      timeout 300s python3 "$PYTHON_SCRIPT" \
        -m "$m" -nk "$nk" --preshuffle "$PRESHUFFLE_STATE" --output-file "$OUTPUT_CSV"

      if [ $? -eq 124 ]; then
        echo "!!!!!! WARNING: Task timed out after 5 minutes. Skipping. !!!!!!"
      fi
    done
  done
  echo "UNTUNED_FALSE benchmark finished. Results are in $OUTPUT_CSV"

else
  echo "[ERROR] Invalid MODE set. Please choose 'tuned', 'untuned_true', or 'untuned_false'."
fi
