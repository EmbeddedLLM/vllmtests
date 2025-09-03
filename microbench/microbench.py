import torch
import torch
import time

# Implement your functions here
###############################

def per_tensor_quant_impl(x:torch.Tensor, scale:torch.Tensor, dtype:torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter.ops.quant import per_tensor_quant_hip
    return per_tensor_quant_hip(x, scale, dtype)


def per_tensor_quant_fake(x:torch.Tensor, scale:torch.Tensor, dtype:torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x, dtype=dtype), torch.empty(1, dtype=torch.float32, device=x.device)

from vllm.utils import direct_register_custom_op
direct_register_custom_op(
    op_name="per_tensor_quant",
    op_func=per_tensor_quant_impl,
    mutates_args=[],
    fake_impl=per_tensor_quant_fake,
)

def per_tensor_quant(x:torch.Tensor, scale:torch.Tensor, dtype:torch.dtype):
    return torch.ops.vllm.per_tensor_quant(x, scale, dtype)

def scaled_fp8_quant(x:torch.Tensor, scale:torch.Tensor, dtype:torch.dtype):
    from vllm import _custom_ops as ops
    return ops.scaled_fp8_quant(
        x,
        scale,
        num_token_padding=None,
        scale_ub=None,
        use_per_token_if_dynamic=False)

per_tensor_quant_compiled = torch.compile(per_tensor_quant,
                            fullgraph=True,
                            backend="inductor",
                            mode="reduce-overhead",
                            dynamic=False)


scaled_fp8_quant_compiled = torch.compile(scaled_fp8_quant,
                            fullgraph=True,
                            backend="inductor",
                            mode="reduce-overhead",
                            dynamic=False)

###############################

# Implement your input generator factory here
###############################
def input_generator_factory(shape, dtype, device):
    def input_generator():
        from vllm.platforms import current_platform
        x = torch.randn(shape, dtype=dtype, device=device)
        scale = torch.ones((1,), dtype=torch.float32, device=device)
        quant_dtype = current_platform.fp8_dtype()
        return x, scale, quant_dtype
    return input_generator
###############################

# No change needed here
###############################
def benchmark_op(fn, input_generator, warmup=10, repeat=2000, label_discriptor= "", label=""):
    # Warmup
    for _ in range(warmup):
        out = fn(*input_generator())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    times = []
    inputs = input_generator()
    for _ in range(repeat):
        start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        t1 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if t0: t0.record()
        out = fn(*inputs)
        if t1: t1.record()
        # GPU timer if CUDA available
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            elapsed = t0.elapsed_time(t1) / 1000.0  # ms --> s
        else:
            elapsed = time.perf_counter() - start
        times.append(elapsed)
    mean_time = sum(times) / repeat
    print(f"{fn.__name__}: {label_discriptor}={label}, mean time={mean_time*1000:.3f} ms")

###############################

# Modify as needed here
def bench_loop():
# Uncomment to save profiler traces
#
# with torch.profiler.profile(
#     activities=[
#         torch.profiler.ProfilerActivity.CPU,
#         torch.profiler.ProfilerActivity.CUDA
#     ],
#     record_shapes=True,
#     with_stack=True,
#     on_trace_ready=torch.profiler.tensorboard_trace_handler('./traces') # saves a trace as .json
# ) as prof:
    dtype = torch.bfloat16
    device = torch.device('cuda')
    shapes = [(1024, 1024),]
    for shape in shapes:
        input_generator = input_generator_factory(shape, dtype, device)
        benchmark_op(per_tensor_quant, input_generator, label_discriptor="shape", label=shape,)

###############################

bench_loop()
