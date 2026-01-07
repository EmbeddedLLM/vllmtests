# vLLM Omni Models Evaluation Scripts

This project contains automated scripts for evaluating the Qwen2.5-Omni-7B and Qwen3-Omni-30B-A3B-Instruct models on the AMD ROCm platform.
Other test scripts will be added later.
The scripts cover text, image, and audio modalities, and support performance and accuracy testing in different combinations of Eager/Graph modes and AITER Kernel switches.


All scripts accept a numeric parameter [1-4] to specify the run mode. The meanings of 1-4 are as follows:

- ### 1: Eager + AITER
- ### 2: Non-Eager (Graph) + AITER
- ### 3: Eager + No AITER
- ### 4: Non-Eager (Graph) + No AITER

Command format example (using Qwen2.5-Omni-7B as an example):
- cd vllm-omni-tests/vllm
- ./Qwen2.5-Omni-7B_text_eval.sh 1
- ./Qwen2.5-Omni-7B_text_eval.sh 2
- ./Qwen2.5-Omni-7B_text_eval.sh 3
- ./Qwen2.5-Omni-7B_text_eval.sh 4
- ./Qwen2.5-Omni-7B_image_eval.sh 1
- ./Qwen2.5-Omni-7B_image_eval.sh 2
- ./Qwen2.5-Omni-7B_image_eval.sh 3
- ./Qwen2.5-Omni-7B_image_eval.sh 4
- ./Qwen2.5-Omni-7B_audio_eval.sh 1
- ./Qwen2.5-Omni-7B_audio_eval.sh 2
- ./Qwen2.5-Omni-7B_audio_eval.sh 3
- ./Qwen2.5-Omni-7B_audio_eval.sh 4



# Special Note:

The model path in the script is the actual model loading path on the MI300X. This path may sometimes change. When using the vllm-omni script in vllmtests, please adjust accordingly.

First, find the actual model path. Example commands are as follows:

### 1. Find the actual path of the 7B model
- find /app/model/models--Qwen--Qwen2.5-Omni-7B -name config.json | xargs dirname

### 2. Find the actual path of the 30B model
- find /app/model/models--Qwen--Qwen3-Omni-30B-A3B-Instruct -name config.json | xargs dirname

Then, place the output in the MODEL_PATH field of the script. Currently, the model address in the script is correct, and you can run the script directly.