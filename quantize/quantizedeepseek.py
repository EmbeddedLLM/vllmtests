import torch
from datasets import load_dataset
from packaging.version import Version
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers.compression.helpers import calculate_offload_device_map

# NOTE: transformers 4.49.0 has an attribute error with DeepSeek.
# Please consider either downgrading your transformers version to a
# previous version or upgrading to a version where this bug is fixed

# this recipe is from deepseek-moe: llm-compressor/examples/quantizing_moe/deepseek_moe_w8a8_fp8.py

# select a Mixture of Experts model for quantization

MODEL_ID = "unsloth/DeepSeek-R1-BF16"
# device_map example is from: llm-compressor/examples/quantizing_moe/deepseek_moe_w8a8_int8.py

# adjust based off number of desired GPUs
# if not enough memory is available, some layers will automatically be offlaoded to cpu
device_map = calculate_offload_device_map(
    MODEL_ID,
    reserve_for_hessians=False,
    num_gpus=8,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map=device_map, torch_dtype=torch.bfloat16, trust_remote_code=True
)
num_paras=model.num_parameters()
print(f"num_paras: {num_paras}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Select calibration dataset.
# its recommended to use more calibration samples for MoE models so each expert is hit
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 1024
MAX_SEQUENCE_LENGTH = 2048


# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


ds = ds.map(preprocess)


# Tokenize inputs.
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)

# define a llmcompressor recipe for FP8 W8A8 quantization
# since the MoE gate layers are sensitive to quantization, we add them to the ignore
# list so they remain at full precision
recipe = [
    QuantizationModifier(
        targets="Linear",
        scheme="FP8_DYNAMIC",
        ignore=["lm_head", "re:.*mlp.gate$"],
    ),
]

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
)

# Confirm generations of the quantized model look sane.
# Generation is broken for deepseek models when using the latest transformers package
if Version(__version__) < Version("4.48"):
    print("========== SAMPLE GENERATION ==============")
    SAMPLE_INPUT = ["I love quantization because"]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    inputs = tokenizer(SAMPLE_INPUT, return_tensors="pt", padding=True).to(model.device)
    output = model.generate(**inputs, max_length=50)
    text_output = tokenizer.batch_decode(output)
    print(text_output)
else:
    print(
        "WARNING: cannot perform sample generation of "
        "deepseek models with transformers >= 4.48"
    )

# Save to disk in compressed-tensors format.
# SAVE_DIR = MODEL_ID.split("/")[-1] + "/raid/yuzho/0604_qwen1.5b/0607_ds/deepseek-r1-FP8-Dynamic-from-BF16"
SAVE_DIR = "/app/model/QuantLLM/deepseek-r1-FP8-Dynamic-from-BF16-calib1024"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)