# SPDX-License-Identifier: Apache-2.0
import argparse
import csv
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import torch
from vllm.platforms import current_platform
from vllm.utils import FlexibleArgumentParser

# --- Import RoPE Implementations ---
# Import vLLM's get_rope function
from vllm.model_executor.layers.rotary_embedding import \
    get_rope as vllm_get_rope

# Attempt to import AITeR's get_rope.
try:
    from aiter.rotary_embedding import get_rope as aiter_get_rope
except ImportError:
    print(
        "Warning: Could not import 'aiter.rotary_embedding'. "
        "The benchmark for aiter_get_rope will be skipped."
    )
    aiter_get_rope = None


def run_single_benchmark(
    name: str,
    rope_impl: Optional[torch.nn.Module],
    positions: torch.Tensor,
    query_base: torch.Tensor,
    key_base: torch.Tensor,
    warmup: int,
    repeats: int,
) -> Optional[float]:
    """Helper function to run warmup and timed benchmark for a given RoPE."""
    if rope_impl is None:
        return None

    # Clone tensors as the forward pass is in-place.
    q = query_base.clone()
    k = key_base.clone()
    device = q.device

    # Warmup runs
    for _ in range(warmup):
        _ = rope_impl.forward(positions, q, k)
    torch.cuda.synchronize(device)

    # Timed runs
    start_time = time.perf_counter()
    for _ in range(repeats):
        _ = rope_impl.forward(positions, q, k)
    torch.cuda.synchronize(device)
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time_ms = (total_time / repeats) * 1000
    return avg_time_ms


def benchmark_rope_with_real_shapes(
    # RoPE creation parameters
    head_size: int,
    rotary_dim: int,
    is_neox_style: bool,
    max_position: int,
    original_max_pos: int,
    base: int,
    scaling_factor: float,
    # Tensor parameters
    num_q_heads: int,
    num_kv_heads: int,
    dtype_str: str,
    # Benchmark control
    seed: int,
    device_str: str,
    warmup: int,
    repeats: int,
    output_csv: str,
) -> None:
    """
    Compares RoPE implementations using a list of real-world tensor shapes
    and logs the results to a CSV file.
    """
    # --- Setup Environment ---
    current_platform.seed_everything(seed)
    device = torch.device(device_str)
    torch.set_default_device(device)
    try:
        dtype = getattr(torch, dtype_str)
    except AttributeError:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    # --- Define RoPE Scaling Configuration ---
    rope_scaling: Dict[str, Any] = {
        "rope_type": "deepseek_yarn",
        "factor": scaling_factor,
        "original_max_position_embeddings": original_max_pos,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
    }

    # --- Instantiate RoPE Kernels ---
    print("\n--- Instantiating RoPE Kernels ---")
    vllm_rope = vllm_get_rope(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position=max_position,
        base=base,
        is_neox_style=is_neox_style,
        rope_scaling=rope_scaling,
        dtype=dtype,
    )
    vllm_rope_type = type(vllm_rope).__name__
    print(f"vLLM get_rope() returned object of type: {vllm_rope_type}")

    aiter_rope = None
    aiter_rope_type = "N/A"
    if aiter_get_rope:
        aiter_rope = aiter_get_rope(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position=max_position,
            base=base,
            is_neox_style=is_neox_style,
            rope_scaling=rope_scaling,
            dtype=dtype,
        )
        aiter_rope_type = type(aiter_rope).__name__
        print(f"AITeR get_rope() returned object of type: {aiter_rope_type}")
    else:
        print("AITeR RoPE implementation: Not available")

    # --- Define Shapes to Benchmark (collected from logs) ---
    # shapes_to_test: List[Tuple[int, str]] = [
    #     # Prefill shapes
    #     (8192, "Prefill (Large)"),
    #     (8186, "Prefill (Large)"),
    #     (8015, "Prefill (Large)"),
    #     # Decode shapes
    #     (177, "Decode (Medium)"),
    #     (119, "Decode (Medium)"),
    #     (58, "Decode (Small)"),
    #     (15, "Decode (Small)"),
    #     (6, "Decode (Small)"),
    # ]
    shapes_to_test: List[Tuple[int, str]] = []

    for i in range(1, 128):
        shapes_to_test.append((6144 * i, f"Prefill (Large) {i}"))
        shapes_to_test.append((3025 * i, f"Prefill (Large) {i}"))

    for i in range(1, 128):
        shapes_to_test.append((500 * i, f"Decode (Medium) {i}"))
        shapes_to_test.append((256 * i, f"Decode (Medium) {i}"))
        shapes_to_test.append((177 * i, f"Decode (Medium) {i}"))
        shapes_to_test.append((119 * i, f"Decode (Medium) {i}"))
        shapes_to_test.append((58 * i, f"Decode (Small) {i}"))
        shapes_to_test.append((15 * i, f"Decode (Small) {i}"))

    # --- Setup CSV Logging ---
    csv_headers = [
        "description", "num_tokens", "num_q_heads", "num_kv_heads", "head_size",
        "dtype", "device", "vllm_rope_type", "aiter_rope_type",
        "vllm_time_ms", "aiter_time_ms"
    ]
    
    print(f"\n--- Logging results to {output_csv} ---")
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

        # --- Print Table Header to Console ---
        print(f"\n--- Starting Benchmark Loop (Warmup={warmup}, Repeats={repeats}) ---")
        print("-" * 80)
        print(f"{'Description':<20} | {'Tokens':<8} | {'vLLM Time (ms)':<16} | {'AITeR Time (ms)':<16}")
        print("-" * 80)

        for num_tokens, description in shapes_to_test:
            # --- Create Input Tensors for the current shape ---
            positions = torch.randint(0, original_max_pos, (num_tokens,), device=device, dtype=torch.int64)
            query_base = torch.randn(num_tokens, num_q_heads, head_size, dtype=dtype, device=device)
            key_base = torch.randn(num_tokens, num_kv_heads, head_size, dtype=dtype, device=device)

            # --- Run Benchmarks for this shape ---
            vllm_time = run_single_benchmark(
                "vLLM", vllm_rope, positions, query_base, key_base, warmup, repeats
            )
            aiter_time = run_single_benchmark(
                "AITeR", aiter_rope, positions, query_base, key_base, warmup, repeats
            )

            # --- Log to Console ---
            vllm_time_str = f"{vllm_time:.8f}" if vllm_time is not None else "N/A"
            aiter_time_str = f"{aiter_time:.8f}" if aiter_time is not None else "N/A"
            print(f"{description:<20} | {str(num_tokens):<8} | {vllm_time_str:<16} | {aiter_time_str:<16}")

            # --- Log to CSV File ---
            row = [
                description, num_tokens, num_q_heads, num_kv_heads, head_size,
                dtype_str, device_str, vllm_rope_type, aiter_rope_type,
                vllm_time_str, aiter_time_str
            ]
            writer.writerow(row)

    print("-" * 80)
    print("Benchmark complete.")


if __name__ == '__main__':
    # Generate a default filename with a timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    default_csv_filename = f"rope_benchmark_results_{timestamp}.csv"

    parser = FlexibleArgumentParser(
        description="Benchmark vLLM and AITeR RoPE with real-world shapes and log to CSV."
    )
    # I/O
    parser.add_argument("--output-csv", type=str, default=default_csv_filename, help="Path to the output CSV file.")
    
    # Benchmark control
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations.")
    parser.add_argument("--repeats", type=int, default=100, help="Number of timed repeat iterations.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda:0", help="PyTorch device.")
    
    # Tensor parameters from logs
    parser.add_argument("--num-q-heads", type=int, default=16, help="Number of query heads.")
    parser.add_argument("--num-kv-heads", type=int, default=1, help="Number of key/value heads.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Tensor data type.")

    # RoPE-specific parameters (defaults from Deepseek config)
    parser.add_argument("--head-size", type=int, default=64, help="Dimension of each attention head (for RoPE).")
    parser.add_argument("--rotary-dim", type=int, default=64, help="Dimension of rotary embeddings.")
    parser.add_argument("--is-neox-style", action="store_true", default=False, help="Use Neox-style RoPE. Default is GPT-J style.")
    parser.add_argument("--max-position", type=int, default=163840, help="Maximum position for RoPE cache size (scaled).")
    parser.add_argument("--original-max-pos", type=int, default=4096, help="Original maximum position before scaling.")
    parser.add_argument("--base", type=int, default=10000, help="RoPE base theta value.")
    parser.add_argument("--scaling-factor", type=float, default=40.0, help="RoPE scaling factor.")

    args = parser.parse_args()

    benchmark_rope_with_real_shapes(
        # RoPE creation
        head_size=args.head_size,
        rotary_dim=args.rotary_dim,
        is_neox_style=args.is_neox_style,
        max_position=args.max_position,
        original_max_pos=args.original_max_pos,
        base=args.base,
        scaling_factor=args.scaling_factor,
        # Tensor creation
        num_q_heads=args.num_q_heads,
        num_kv_heads=args.num_kv_heads,
        dtype_str=args.dtype,
        # Benchmark control
        seed=args.seed,
        device_str=args.device,
        warmup=args.warmup,
        repeats=args.repeats,
        output_csv=args.output_csv,
    )