#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from transformers import AutoTokenizer, AutoProcessor, Qwen3VLMoeForConditionalGeneration

from quark.torch import ModelQuantizer, ModelExporter
from quark.torch.quantization import FP8E4M3PerChannelSpec
from quark.torch.quantization.config.config import Config, QuantizationConfig
from quark.torch.export import ExporterConfig, JsonExporterConfig

# Load the original floating-point model
ckpt_path = "Qwen/Qwen3-VL-235B-A22B-Instruct"
# model = AutoModelForCausalLM.from_pretrained(ckpt_path, device_map="auto", trust_remote_code=True)
model = Qwen3VLMoeForConditionalGeneration.from_pretrained(ckpt_path, device_map="auto", trust_remote_code=True)
model.eval()
# tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
tokenizer = AutoProcessor.from_pretrained(ckpt_path)

# Set the quantization configuration
FP8_PER_CHANNEL_SPEC = FP8E4M3PerChannelSpec(is_dynamic=False, ch_axis=0).to_quantization_spec()

FP8_PER_TOKEN_DYNAMIC_SPEC = FP8E4M3PerChannelSpec(is_dynamic=True, ch_axis=1).to_quantization_spec()

W_FP8_PER_CHANNEL_STATIC_A_FP8_PER_TOKEN_DYNAMIC_CONFIG = QuantizationConfig(input_tensors=FP8_PER_TOKEN_DYNAMIC_SPEC,
                                                                             weight=FP8_PER_CHANNEL_SPEC)
quant_config = Config(global_quant_config=W_FP8_PER_CHANNEL_STATIC_A_FP8_PER_TOKEN_DYNAMIC_CONFIG, exclude=["lm_head","*mlp.gate"])

# Apply quantization
quantizer = ModelQuantizer(quant_config, multi_device=True)
model = quantizer.quantize_model(model)

# Export quantized model
output_dir = "/app/model/QuantLLM/" + ckpt_path.rstrip("/").split("/")[-1] + "-FP8-PTPC"
model = quantizer.freeze(model)
export_config = ExporterConfig(json_export_config=JsonExporterConfig(weight_format="real_quantized"))
exporter = ModelExporter(config=export_config, export_dir=output_dir)
exporter.export_safetensors_model(
    model,
    quant_config=quant_config,
    custom_mode="quark",
    tokenizer=tokenizer
)
 