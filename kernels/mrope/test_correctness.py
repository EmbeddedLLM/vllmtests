import pytest
import torch
from kernels import (
    mrope_forward_native, 
    mrope_forward_liger_kernel,
    mrope_forward_liger_kernel_adapted_to_vllm_input
)
from utils import (
    DEVICE, MODEL_TP_DICT, DTYPE_ATOL_RTOL_LIST,
    prepare_test_inputs, get_common_args, validate_inputs, print_config_info
)

# Reduced token list for correctness tests (faster execution)
NUM_TOKENS_LIST_CORRECTNESS = [1, 16, 256, 1024, 4096, 16384]

@pytest.fixture
def device():
    """Fixture to provide device."""
    return DEVICE

def generate_correctness_test_params():
    """Generate test parameter combinations for correctness tests."""
    params = []
    for model_name, tp_list in MODEL_TP_DICT.items():
        for tp_size in tp_list:
            for dtype, atol, rtol in DTYPE_ATOL_RTOL_LIST:
                for num_tokens in NUM_TOKENS_LIST_CORRECTNESS:
                    params.append((model_name, tp_size, dtype, atol, rtol, num_tokens))
    return params

def run_mrope_implementations(
    model_config, positions, query, key, cos, sin
) -> tuple:
    """Run all MRoPE implementations and return results."""
    args = get_common_args(positions, query, key, cos, sin, model_config)
    
    # Test native implementation
    query_native, key_native = mrope_forward_native(*args)

    args = get_common_args(positions, query, key, cos, sin, model_config)
    
    # Test Liger kernel implementation
    query_liger_kernel, key_liger_kernel = mrope_forward_liger_kernel(*args)
    
    args = get_common_args(positions, query, key, cos, sin, model_config)

    # Test vLLM adapted implementation
    query_adapted_to_vllm, key_adapted_to_vllm = mrope_forward_liger_kernel_adapted_to_vllm_input(*args)
    
    return (
        (query_native, key_native),
        (query_liger_kernel, key_liger_kernel),
        (query_adapted_to_vllm, key_adapted_to_vllm)
    )

@pytest.mark.parametrize("model_name,tp_size,dtype,atol,rtol,num_tokens", generate_correctness_test_params())
def test_mrope_correctness(model_name, tp_size, dtype, atol, rtol, num_tokens, device):
    """Test correctness of MRoPE implementations."""
    print(f"\nTesting: {model_name}, TP: {tp_size}, Tokens: {num_tokens}, Dtype: {dtype}")
    
    # Validate inputs
    validate_inputs(model_name, tp_size, num_tokens)
    
    # Prepare test inputs
    model_config, positions, query, key, cos, sin = prepare_test_inputs(
        model_name, tp_size, dtype, num_tokens, device
    )
    
    # Run all implementations
    (query_native, key_native), (query_liger, key_liger), (query_vllm, key_vllm) = run_mrope_implementations(
        model_config, positions, query, key, cos, sin
    )
    
    # Assertions for Liger kernel vs native
    torch.testing.assert_close(
        query_native, query_liger, 
        atol=atol, rtol=rtol,
        msg=f"Query mismatch for Liger kernel vs native (model: {model_name}, tp: {tp_size}, tokens: {num_tokens})"
    )
    torch.testing.assert_close(
        key_native, key_liger, 
        atol=atol, rtol=rtol,
        msg=f"Key mismatch for Liger kernel vs native (model: {model_name}, tp: {tp_size}, tokens: {num_tokens})"
    )
    
    # Assertions for vLLM adapted vs native
    torch.testing.assert_close(
        query_native, query_vllm, 
        atol=atol, rtol=rtol,
        msg=f"Query mismatch for vLLM adapted vs native (model: {model_name}, tp: {tp_size}, tokens: {num_tokens})"
    )
    torch.testing.assert_close(
        key_native, key_vllm, 
        atol=atol, rtol=rtol,
        msg=f"Key mismatch for vLLM adapted vs native (model: {model_name}, tp: {tp_size}, tokens: {num_tokens})"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])