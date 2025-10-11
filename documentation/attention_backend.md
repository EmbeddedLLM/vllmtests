# Attention Backend on ROCm

Attention backend can be set through various ways:

- `VLLM_ATTENTION_BACKEND`
- through combinations of `VLLM_ROCM_` flags.

## Attention

On AMD ROCm there are `TRITON_ATTN`, `ROCM_ATTN`, `FLASH_ATTN` or `ROCM_AITER_UNIFIED_ATTN`.

- `TRITON_ATTN`:
    - Uses vLLM's triton unified attention backend. Both the prefill and decode are triton kernels.

    Example command:
        ```bash
        # Example 1
        vllm serve meta-llama/Llama-3.1-8B-Instruct

        # Example 2
        VLLM_ATTENTION_BACKEND="TRITON_ATTN" vllm serve meta-llama/Llama-3.1-8B-Instruct

        # Example 3 (When enable AITER but still want to use TRITON_ATTN)
        # This does not work yet, we need to make this case work on upstream
        VLLM_ROCM_USE_AITER=1 VLLM_ATTENTION_BACKEND="TRITON_ATTN" vllm serve meta-llama/Llama-3.1-8B-Instruct

        # OR
        VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MHA=0 vllm serve meta-llama/Llama-3.1-8B-Instruct
        ```

- `ROCM_ATTN`
    - Uses vLLM's chunked prefill paged decode kernel. The prefill is triton kernel and the decode is custom HIP paged attention kernel.

    - Examples
        ```bash
        # Example 1
        VLLM_ATTENTION_BACKEND="ROCM_ATTN" vllm serve meta-llama/Llama-3.1-8B-Instruct
        
        # Example 2 (When enable AITER but still want to use TRITON_ATTN)
        # This does not work yet, we need to make this case work on upstream. It still use ROCM AITER FA
        VLLM_ROCM_USE_AITER=1 VLLM_ATTENTION_BACKEND="ROCM_ATTN" vllm serve meta-llama/Llama-3.1-8B-Instruct

        # OR

        VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MHA=0 VLLM_ATTENTION_BACKEND="ROCM_ATTN" vllm serve meta-llama/Llama-3.1-8B-Instruct
        
        # OR
        VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MHA=0 VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 vllm serve meta-llama/Llama-3.1-8B-Instruct

        ```

- `ROCM_AITER_FA`
    - Use the AITER Flash Attention backend.

    - Examples
        ```bash
        # Example 1 (Only use AITER FA backend without enabling other AITER kernels)
        ## Fix this bug. Missing adding `ROCM_AITER_FA` to the V1 oracle list in 
        ## https://github.com/vllm-project/vllm/blob/9d6cff3edeb2421699671881592fd7558946695e/vllm/engine/arg_utils.py#L1627
        VLLM_ATTENTION_BACKEND="ROCM_AITER_FA" vllm serve meta-llama/Llama-3.1-8B-Instruct
        
        # Example 2
        VLLM_ROCM_USE_AITER=1 vllm serve meta-llama/Llama-3.1-8B-Instruct
        ```

- `ROCM_AITER_UNIFIED_ATTN`
    - Use AITER unified attention backend.

    - Examples
        ```bash
        # Example 1 (Only use AITER FA backend without enabling other AITER kernels)
        VLLM_ATTENTION_BACKEND="ROCM_AITER_UNIFIED_ATTN" vllm serve meta-llama/Llama-3.1-8B-Instruct
        
        # Example 2
        # This does not work yet, we need to make this case work on upstream. It still use ROCM AITER FA
        VLLM_ROCM_USE_AITER=1 VLLM_ATTENTION_BACKEND="ROCM_AITER_UNIFIED_ATTN" vllm serve meta-llama/Llama-3.1-8B-Instruct

        # OR 
        VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MHA=0 VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 vllm serve meta-llama/Llama-3.1-8B-Instruct
        ```




## MLA Backend:

On AMD ROCm, there are `TRITON_MLA` and `ROCM_AITER_MLA`


- `TRITON_MLA`:
    - Uses vLLM's triton MLA backend. The prefill uses triton flash attention/ CK flash attention varlen, and decode uses triton mla decode kernel.

    - Example commands:
        ```bash
        VLLM_ATTENTION_BACKEND="TRITON_MLA" vllm serve deepseek-ai/DeepSeek-R1 -tp 8
        
        VLLM_ROCM_USE_AITER=1 VLLM_ATTENTION_BACKEND="TRITON_MLA" vllm serve deepseek-ai/DeepSeek-R1 -tp 8
        ```

- `ROCM_AITER_MLA`:
    - Uses AITER MLA backend. It only supports `--block-size 1`. The default value of `block-size` in vLLM is `16`.

    - Example commands:
        ```bash
        VLLM_ATTENTION_BACKEND="ROCM_AITER_MLA" vllm serve deepseek-ai/DeepSeek-R1 -tp 8 --block-size 1
        
        VLLM_ROCM_USE_AITER=1 vllm serve deepseek-ai/DeepSeek-R1 -tp 8  --block-size 1
        ```
