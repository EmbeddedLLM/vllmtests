import torch
from vllm.triton_utils import triton, tl

# Using Liger Kernel Built-in function
def liger_triton_qwen2vl_mrope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:

    from liger_kernel.ops.qwen2vl_mrope import _triton_qwen2vl_mrope


    n_row, n_q_head, head_dim = q.shape
    n_kv_head = k.shape[1]
    pad_hd = triton.next_power_of_2(head_dim)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)
    BLOCK_SIZE = max(pad_n_q_head, pad_n_kv_head)

    # ensure tensors passed into the kernel are contiguous. 
    # It will be no-op if they are already contiguous
    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()


    _triton_qwen2vl_mrope[(n_row,)](
        q,
        k,
        cos,
        sin,
        1, # seq_len is assigned to num_tokens
        n_row, # batch_size set to 1 to make it work with vllm
        n_q_head,
        n_kv_head,
        head_dim,
        pad_n_q_head,
        pad_n_kv_head,
        pad_hd,
        mrope_section[0],
        mrope_section[1],
        BLOCK_SIZE=BLOCK_SIZE,
        BACKWARD_PASS=False,
    )
    return q, k


def mrope_forward_liger_kernel(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
    is_neox_style: bool,
    head_size: int,
    rotary_dim: int
):

    """Liger Kernel Original Implementation.

    Args:
        positions:
            [num_tokens,] (text only) or
            [3, num_tokens] (T/H/W positions with multimodal inputs)
        query: [num_tokens, num_heads * head_size]
        key: [num_tokens, num_kv_heads * head_size]
        cos: [3, num_tokens, head_dim // 2]
        sin: [3, num_tokens, head_dim // 2]
    """

    # Original Implementation 
    assert positions.ndim == 1 or positions.ndim == 2
    assert key is not None

    num_tokens = positions.shape[-1]
    query_shape = query.shape
    key_shape = key.shape

    if positions.ndim == 2:
        assert mrope_section

        # Duplicate to make it head_dim (the kernel expects full head_dim)
        cos = torch.cat([cos, cos], dim=-1)  # [num_tokens, head_dim]
        sin = torch.cat([sin, sin], dim=-1)  # [num_tokens, head_dim]
        
        # Add batch dimension
        cos = cos.unsqueeze(1)  # [1, num_tokens, head_dim]
        sin = sin.unsqueeze(1)  # [1, num_tokens, head_dim]

        q, k = liger_triton_qwen2vl_mrope(
            query.reshape(num_tokens, -1, head_size), # query.shape: torch.Size([104, 28, 128]) [num_tokens, head_size, head_dim]
            key.reshape(num_tokens, -1, head_size),
            cos, # Expected shape: (3, 1, num_tokens, head_dim)
            sin, # Expected shape: (3, 1, num_tokens, head_dim)
            mrope_section,
        )

        return q.reshape(query_shape), k.reshape(key_shape)

    raise NotImplementedError("positions.ndim must be 2 and mrope_section must be defined")