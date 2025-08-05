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

