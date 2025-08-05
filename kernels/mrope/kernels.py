import torch
from vllm.triton_utils import triton, tl
from typing import Optional

# Native Function
def apply_rotary_emb_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


def apply_rotary_emb_dispatch(x: torch.Tensor, cos: torch.Tensor,
                              sin: torch.Tensor,
                              is_neox_style: bool) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    return apply_rotary_emb_torch(x, cos, sin, is_neox_style)

# vLLM Native implementation of mrope forward pass

def mrope_forward_native(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
    is_neox_style: bool,
    head_size: int,
    rotary_dim: int
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """PyTorch-native implementation equivalent to forward().

    Args:
        positions:
            [num_tokens,] (text only) or
            [3, num_tokens] (T/H/W positions with multimodal inputs)
        query: [num_tokens, num_heads * head_size]
        key: [num_tokens, num_kv_heads * head_size]
        cos: [3, num_tokens, head_dim // 2]
        sin: [3, num_tokens, head_dim // 2]
    """

    assert positions.ndim == 1 or positions.ndim == 2
    assert key is not None

    num_tokens = positions.shape[-1]
    ## Do these two outside
    # cos_sin = self.cos_sin_cache[positions]
    # cos, sin = cos_sin.chunk(2, dim=-1)
    query_shape = query.shape
    key_shape = key.shape

    if positions.ndim == 2:
        # the mrope path
        assert mrope_section

        cos = torch.cat([
            m[i]
            for i, m in enumerate(cos.split(mrope_section, dim=-1))
        ],
                        dim=-1)
        sin = torch.cat([
            m[i]
            for i, m in enumerate(sin.split(mrope_section, dim=-1))
        ],
                        dim=-1)

    query = query.view(num_tokens, -1, head_size)
    query_rot = query[..., :rotary_dim]
    query_pass = query[..., rotary_dim:]
    query_rot = apply_rotary_emb_dispatch(query_rot, cos, sin,
                                            is_neox_style)
    query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

    key = key.view(num_tokens, -1, head_size)
    key_rot = key[..., :rotary_dim]
    key_pass = key[..., rotary_dim:]
    key_rot = apply_rotary_emb_dispatch(key_rot, cos, sin,
                                        is_neox_style)
    key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
    return query, key


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



# Adapted from Liger Kernel to take in vLLM input format
@triton.jit
def _triton_qwen2vl_mrope_forward(
    q_ptr,
    k_ptr,
    cos,
    sin,
    num_tokens,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    mrope_section_t: tl.constexpr,
    mrope_section_h: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Adapted from 
    # https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/qwen2vl_mrope.py
    # This version supports flatten input tensors from vllm
    # and supports cos and sin cache with shape (3, num_tokens, head_dim // 2) 
    # instead of (3, bsz, seq_len, head_dim)
    pid = tl.program_id(0)
    # locate start address
    q_ptr = q_ptr + pid * (n_qh * hd)
    k_ptr = k_ptr + pid * (n_kh * hd)
    
    # ####################################################################
    # get the cos(mθ_{i...d/2}) and sin(mθ_{i...d/2}) for token position
    # m of this program instance
    # ####################################################################
    # Note: cos and sin now have shape (3, num_tokens, head_dim // 2)
    
    t_end = mrope_section_t
    h_end = t_end + mrope_section_h
    
    # Updated stride calculation for half head_dim
    half_hd = hd // 2
    t_cos = cos + pid * half_hd
    h_cos = t_cos + num_tokens * half_hd
    w_cos = h_cos + num_tokens * half_hd
    t_sin = sin + pid * half_hd
    h_sin = t_sin + num_tokens * half_hd
    w_sin = h_sin + num_tokens * half_hd
    
    # Updated offsets for half head_dim
    cos_offsets = tl.arange(0, pad_hd // 2)
    t_mask = cos_offsets < t_end
    h_mask = (t_end <= cos_offsets) & (cos_offsets < h_end)
    w_mask = (h_end <= cos_offsets) & (cos_offsets < half_hd)
    
    t_cos_row = tl.load(t_cos + cos_offsets, mask=t_mask, other=0)
    h_cos_row = tl.load(h_cos + cos_offsets, mask=h_mask, other=0)
    w_cos_row = tl.load(w_cos + cos_offsets, mask=w_mask, other=0)
    t_sin_row = tl.load(t_sin + cos_offsets, mask=t_mask, other=0)
    h_sin_row = tl.load(h_sin + cos_offsets, mask=h_mask, other=0)
    w_sin_row = tl.load(w_sin + cos_offsets, mask=w_mask, other=0)
    
    cos_row = t_cos_row + h_cos_row + w_cos_row
    sin_row = t_sin_row + h_sin_row + w_sin_row
    
    # ####################################################################
    # Load the left and right half of q and k for the current
    # program instance (i.e. for the current token) separately
    # ####################################################################
    # left half of the head
    first_half_q_offsets = tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    first_half_k_offsets = tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (tl.arange(0, pad_hd // 2)[None, :] < hd // 2)
    first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (tl.arange(0, pad_hd // 2)[None, :] < hd // 2)
    
    q_tile_1 = tl.load(q_ptr + first_half_q_offsets, mask=first_q_mask, other=0).to(sin_row.dtype)
    k_tile_1 = tl.load(k_ptr + first_half_k_offsets, mask=first_k_mask, other=0).to(sin_row.dtype)
    
    # right half of the head
    second_half_q_offsets = first_half_q_offsets + (hd // 2)
    second_half_k_offsets = first_half_k_offsets + (hd // 2)
    second_q_mask = first_q_mask
    second_k_mask = first_k_mask
    
    q_tile_2 = tl.load(q_ptr + second_half_q_offsets, mask=second_q_mask, other=0).to(sin_row.dtype)
    k_tile_2 = tl.load(k_ptr + second_half_k_offsets, mask=second_k_mask, other=0).to(sin_row.dtype)
    
    # y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
    # Since cos and sin are now half-size, we use the same cos_row and sin_row for both halves
    new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
    tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
    new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
    tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)
    
    new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
    tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
    new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
    tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)

def triton_qwen2vl_mrope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:

    """Qwen2VL mrope kernel.

    Args:
        query: [num_tokens, num_heads * head_size]
        key: [num_tokens, num_kv_heads * head_size]
        cos: [3, num_tokens, head_size //2 ] 
            (T/H/W positions with multimodal inputs)
        sin: [3, num_tokens, head_size //2 ] 
            (T/H/W positions with multimodal inputs)
        mrope_section: [t, h, w]
        head_size: int
    """
    n_row, n_q_head_head_dim = q.shape
    n_q_head = n_q_head_head_dim // head_size
    n_kv_head = k.shape[1] // head_size
    pad_hd = triton.next_power_of_2(head_size)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)
    BLOCK_SIZE = max(pad_n_q_head, pad_n_kv_head)

    # ensure tensors passed into the kernel are contiguous. 
    # It will be no-op if they are already contiguous
    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    _triton_qwen2vl_mrope_forward[(n_row,)](
        q,
        k,
        cos,
        sin,
        n_row,
        n_q_head,
        n_kv_head,
        head_size,
        pad_n_q_head,
        pad_n_kv_head,
        pad_hd,
        mrope_section[0],
        mrope_section[1],
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return q, k

def mrope_forward_liger_kernel_adapted_to_vllm_input(
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

        q, k = triton_qwen2vl_mrope(
            query, # query.shape: torch.Size([104, 28, 128]) [num_tokens, head_size, head_dim]
            key,
            cos, # Expected shape: (3, 1, num_tokens, head_dim)
            sin, # Expected shape: (3, 1, num_tokens, head_dim)
            mrope_section,
            head_size,
        )

        return q.reshape(query_shape), k.reshape(key_shape)

    raise NotImplementedError("positions.ndim must be 2 and mrope_section must be defined")
