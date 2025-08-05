import torch
from typing import Tuple, Dict, Any
from transformers import AutoConfig
from mrope import SimpleMRoPEClass

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Test configurations
MODEL_TP_DICT = {
    "Qwen/Qwen2-VL-2B-Instruct": [1],
    "Qwen/Qwen2-VL-7B-Instruct": [1],
    "Qwen/Qwen2-VL-72B-Instruct": [2, 4, 8],
    "Qwen/Qwen2.5-VL-3B-Instruct": [1, 2, 4, 8],
    "Qwen/Qwen2.5-VL-7B-Instruct": [1, 2, 4, 8],
    "Qwen/Qwen2.5-VL-72B-Instruct": [2, 4, 8]
}

DTYPE_ATOL_RTOL_LIST = [
    (torch.bfloat16, 1e-5, 1.6e-2),
]

DTYPE_LIST = [torch.bfloat16]

def generate_test_data(
    num_tokens: int, 
    num_q_heads: int, 
    num_kv_heads: int, 
    head_size: int, 
    max_position_embeddings: int,
    dtype: torch.dtype, 
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

def get_model_config(model_name: str, tp_size: int) -> Dict[str, Any]:
    """Get model configuration parameters."""
    config = AutoConfig.from_pretrained(model_name)
    
    total_num_kv_heads = config.num_key_value_heads
    total_num_heads = config.num_attention_heads
    num_heads = total_num_heads // tp_size
    num_kv_heads = max(1, total_num_kv_heads // tp_size)
    head_dim = config.hidden_size // total_num_heads
    
    mrope_section = config.rope_scaling["mrope_section"]
    scaling = head_dim**-0.5 if "Qwen2" in model_name else 1.0
    rope_theta = config.rope_theta
    max_position = config.max_position_embeddings
    
    return {
        'num_heads': num_heads,
        'num_kv_heads': num_kv_heads,
        'head_dim': head_dim,
        'mrope_section': mrope_section,
        'scaling': scaling,
        'rope_theta': rope_theta,
        'max_position': max_position,
        'config': config
    }

def create_mrope_helper(model_config: Dict[str, Any], dtype: torch.dtype) -> SimpleMRoPEClass:
    """Create MRoPE helper class from model configuration."""
    return SimpleMRoPEClass(
        head_size=model_config['head_dim'],
        rotary_dim=model_config['head_dim'],
        max_position_embeddings=model_config['max_position'],
        base=model_config['rope_theta'],
        is_neox_style=True,
        dtype=dtype,
        mrope_section=model_config['mrope_section'],
    )

def prepare_test_inputs(
    model_name: str,
    tp_size: int,
    dtype: torch.dtype,
    num_tokens: int,
    device: torch.device = DEVICE
) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare all inputs needed for MRoPE testing/benchmarking."""
    # Get model configuration
    model_config = get_model_config(model_name, tp_size)
    
    # Create MRoPE helper class
    mrope_helper_class = create_mrope_helper(model_config, dtype)
    
    # Generate test data
    positions, query, key = generate_test_data(
        num_tokens,
        model_config['num_heads'],
        model_config['num_kv_heads'],
        model_config['head_dim'],
        model_config['max_position'],
        dtype,
        device
    )
    
    cos_sin = mrope_helper_class.cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    
    return model_config, positions, query, key, cos, sin

def get_common_args(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    model_config: Dict[str, Any]
) -> Tuple:
    """Get common arguments for all MRoPE implementations."""
    return (
        positions,
        query.clone(),
        key.clone(),
        cos,
        sin,
        model_config['mrope_section'],
        True,  # is_neox_style
        model_config['head_dim'],
        model_config['head_dim'],
    )

def validate_inputs(model_name: str, tp_size: int, num_tokens: int) -> bool:
    """Validate input parameters."""
    if model_name not in MODEL_TP_DICT:
        raise ValueError(f"Model {model_name} not supported. Available models: {list(MODEL_TP_DICT.keys())}")
    
    if tp_size not in MODEL_TP_DICT[model_name]:
        raise ValueError(f"TP size {tp_size} not supported for model {model_name}. "
                        f"Available TP sizes: {MODEL_TP_DICT[model_name]}")
    
    if num_tokens <= 0:
        raise ValueError(f"Number of tokens must be positive, got {num_tokens}")
    
    return True

def print_config_info(model_name: str, tp_size: int, dtype: torch.dtype, num_tokens: int, prefix: str = ""):
    """Print configuration information."""
    print(f"{prefix}Model: {model_name}")
    print(f"{prefix}TP Size: {tp_size}")
    print(f"{prefix}Dtype: {dtype}")
    print(f"{prefix}Num Tokens: {num_tokens}")