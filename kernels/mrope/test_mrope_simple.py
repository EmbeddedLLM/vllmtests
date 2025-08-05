
import torch
from typing import Optional
from transformers import AutoConfig

from kernels import (
    mrope_forward_native, 
    mrope_forward_liger_kernel,
    mrope_forward_liger_kernel_adapted_to_vllm_input
)

import time

import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleMRoPEClass(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
        mrope_section: Optional[list[int]] = None,
    ) -> None:
        super().__init__()
        
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings * 4 # for video
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        cache = self._compute_cos_sin_cache()
        cache = cache.to(dtype)
        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)

        self.mrope_section = mrope_section
        if self.mrope_section:
            assert sum(self.mrope_section) == rotary_dim // 2
        
    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        """Compute the inverse frequency."""
        # NOTE(woosuk): To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        inv_freq = 1.0 / (base**(torch.arange(
            0, self.rotary_dim, 2, dtype=torch.float, device=device) / self.rotary_dim))
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float, device=device)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache



def generate_test_data( 
    num_tokens: int, 
    num_q_heads: int, 
    num_kv_heads: int, 
    head_size:int, 
    max_position_embeddings:int,
    dtype: torch.dtype, 
    device: torch.device
):
    """Generate test data for given configuration."""
    # Create 2D positions (3, num_tokens) for multimodal case
    positions = torch.randint(
        0, max_position_embeddings // 4, 
        (3, num_tokens), 
        device=device
    )
    
    # Create query and key tensors
    query = torch.randn(
        num_tokens, num_q_heads * head_size,
        dtype=dtype, device=device
    )
    key = torch.randn(
        num_tokens, num_kv_heads * head_size,
        dtype=dtype, device=device
    )
    
    return positions, query, key


model_tp_dict = {
    "Qwen/Qwen2-VL-2B-Instruct": [1],
    "Qwen/Qwen2-VL-7B-Instruct": [1],
    "Qwen/Qwen2-VL-72B-Instruct": [2, 4, 8],
    "Qwen/Qwen2.5-VL-3B-Instruct": [1, 2, 4, 8],
    "Qwen/Qwen2.5-VL-7B-Instruct": [1, 2, 4, 8],
    "Qwen/Qwen2.5-VL-72B-Instruct": [2, 4, 8]
}

# https://github.com/pytorch/pytorch/blob/main/torch/testing/_comparison.py#L1317
dtype_atol_rtol_list = [
    [torch.bfloat16, 1e-5, 1.6e-2],
]

num_tokens_list=[2**i for i in range(0, 18)]

WARMUP_ITER = 5
BENCHMARK_ITER = 100

for model_name, tp_list in model_tp_dict.items():
    
    config = AutoConfig.from_pretrained(model_name)

    for tp_size in tp_list:
        
        # get the model config
        total_num_kv_heads = config.num_key_value_heads
        total_num_heads = config.num_attention_heads
        num_heads = total_num_heads // tp_size
        num_kv_heads = max(1, total_num_kv_heads // tp_size)
        head_dim = config.hidden_size // total_num_heads
        q_size = num_heads * head_dim
        kv_size = num_kv_heads * head_dim
        is_neox_style=True
        mrope_section = config.rope_scaling["mrope_section"]

        if "Qwen2" in model_name:
            scaling = head_dim**-0.5
        else:
            scaling = 1.0

        rope_theta = config.rope_theta
        max_position = config.max_position_embeddings


        for (dtype, atol, rtol) in dtype_atol_rtol_list:
            
            # the parameters to compute the q k v size based on tp_size

            mrope_helper_class = SimpleMRoPEClass(
                head_size=head_dim,
                rotary_dim=head_dim,
                max_position_embeddings=max_position,
                base=rope_theta,
                is_neox_style=is_neox_style,
                dtype=dtype, # None in vLLM
                mrope_section=mrope_section,
            )
            for num_tokens in num_tokens_list:
                print(80 * "=")
                print(f"Evaluating model: {model_name} with tp_size: {tp_size} and num_tokens: {num_tokens}, dtype: {dtype}")
                # create q k v input tensors
                # create rotary pos emb input tensors
                positions, query, key = generate_test_data(
                    num_tokens,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    max_position,
                    dtype,
                    device
                )
                cos_sin = mrope_helper_class.cos_sin_cache[positions]
                cos, sin = cos_sin.chunk(2, dim=-1)

                query_native, key_native = mrope_forward_native(
                    positions,
                    query.clone(),
                    key.clone(),
                    cos,
                    sin,
                    mrope_section,
                    is_neox_style,
                    head_dim,
                    head_dim,
                )

                # native Liger Kernel

                query_liger_kernel, key_liger_kernel = mrope_forward_liger_kernel(
                    positions,
                    query.clone(),
                    key.clone(),
                    cos,
                    sin,
                    mrope_section,
                    is_neox_style,
                    head_dim,
                    head_dim,
                )

                torch.testing.assert_close(query_native, query_liger_kernel, atol=atol, rtol=rtol)
                torch.testing.assert_close(key_native, key_liger_kernel, atol=atol, rtol=rtol)
                

                # Liger Kernel Adapted to vLLM Input
                
                query_adapted_to_vllm, key_adapted_to_vllm = mrope_forward_liger_kernel_adapted_to_vllm_input(
                    positions,
                    query.clone(),
                    key.clone(),
                    cos,
                    sin,
                    mrope_section,
                    is_neox_style,
                    head_dim,
                    head_dim,
                )

                torch.testing.assert_close(query_native, query_adapted_to_vllm, atol=atol, rtol=rtol)
                torch.testing.assert_close(key_native, key_adapted_to_vllm, atol=atol, rtol=rtol)

                
                # start benchmarking
                # Warm up
                for _ in range(WARMUP_ITER):
                    mrope_forward_native(
                        positions,
                        query.clone(),
                        key.clone(),
                        cos,
                        sin,
                        mrope_section,
                        is_neox_style,
                        head_dim,
                        head_dim
                    )
                    mrope_forward_liger_kernel(
                        positions,
                        query.clone(),
                        key.clone(),
                        cos,
                        sin,
                        mrope_section,
                        is_neox_style,
                        head_dim,
                        head_dim
                    )
                    mrope_forward_liger_kernel_adapted_to_vllm_input(
                        positions,
                        query.clone(),
                        key.clone(),
                        cos,
                        sin,
                        mrope_section,
                        is_neox_style,
                        head_dim,
                        head_dim
                    )
                
                torch.cuda.synchronize()
                
                # Time reference implementation
                start_time = time.time()
                for _ in range(BENCHMARK_ITER):
                    mrope_forward_native(
                        positions,
                        query.clone(),
                        key.clone(),
                        cos,
                        sin,
                        mrope_section,
                        is_neox_style,
                        head_dim,
                        head_dim
                    )
                torch.cuda.synchronize()
                ref_time = (time.time() - start_time) / BENCHMARK_ITER
                
                # Time liger kernel mrope implementation
                start_time = time.time()
                for _ in range(BENCHMARK_ITER):
                    mrope_forward_liger_kernel(
                        positions,
                        query.clone(),
                        key.clone(),
                        cos,
                        sin,
                        mrope_section,
                        is_neox_style,
                        head_dim,
                        head_dim
                    )
                torch.cuda.synchronize()
                triton_liger_kernel_time = (time.time() - start_time) / BENCHMARK_ITER
                

                # Time mrope adapted to vllm input implementation
                start_time = time.time()
                for _ in range(BENCHMARK_ITER):
                    mrope_forward_liger_kernel_adapted_to_vllm_input(
                        positions,
                        query.clone(),
                        key.clone(),
                        cos,
                        sin,
                        mrope_section,
                        is_neox_style,
                        head_dim,
                        head_dim
                    )
                torch.cuda.synchronize()
                triton_liger_kernel_adapted_to_vllm_input_time = (time.time() - start_time) / BENCHMARK_ITER

                print(f"\nPerformance comparison for config ({num_tokens}, {num_heads}, {num_kv_heads}):")
                print(f"Reference implementation: {ref_time:.8f}s")
                print(f"Liger Kernel implementation: {triton_liger_kernel_time:.8f}s")
                print(f"Liger Kernel Adapted to vLLM Input implementation: {triton_liger_kernel_adapted_to_vllm_input_time:.8f}s")
                print(f"Liger Kernel Speedup over vLLM native: {ref_time/triton_liger_kernel_time:.8f}x")
                print(f"Liger Kernel Adapted to vLLM Input Speedup over vLLM native: {ref_time/triton_liger_kernel_adapted_to_vllm_input_time:.8f}x")
                print()

                
                
