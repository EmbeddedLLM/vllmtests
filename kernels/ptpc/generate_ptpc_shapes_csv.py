import csv
import argparse

# Define the generation configuration for different operators. 
# Adjust these settings flexibly according to the untuned file in the aiter utility.
# Here, key is the operator name you entered in the command line.
OPERATOR_CONFIGS = {
    'bpreshuffle': {
        'has_dtype': True,
        'dtype_value': ['torch.float8_e4m3fnuz']
    },
    'ck': {
        'has_dtype': False,
        'dtype_value': [None]
    },
    # If there's a new operator, simply add a line here, for example:
    # 'new_op': {'has_dtype': True, 'dtype_value': 'torch.int8'},
}

l_m = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 264, 272, 280, 288, 296, 304, 312, 320, 328, 336, 344, 352, 360, 368, 376, 384, 392, 400, 408, 416, 424, 432, 440, 448, 456, 464, 472, 480, 488, 496, 504, 512, 1024, 1536]
l_m += [2048, 4096, 8192, 16384, 32768, 65536, 131072] + [2**i for i in range(18, 19)] # for every power of two until it covers the model context length

# l_nk = [
#     # Llama-4-Maverick-17B-128E-Instruct-FP8
#     (896, 5120), #  TP8 - QKV Proj
#     (1792, 5120), # Llama-4-Maverick-17B-128E-Instruct-FP8 TP4 - QKV Proj
#     (5120, 640), # Llama-4-Maverick-17B-128E-Instruct-FP8 TP8 - Attn Output Proj
#     (5120, 1280), # Llama-4-Maverick-17B-128E-Instruct-FP8 TP4 - Attn Output Proj
#     (5120, 2048), # Llama-4-Maverick-17B-128E-Instruct-FP8 TP8 - MLP Down Proj
#     (5120, 4096), # Llama-4-Maverick-17B-128E-Instruct-FP8 TP4 - MLP Down Proj
#     (8192, 5120), # Llama-4-Maverick-17B-128E-Instruct-FP8 TP4 - MLP Up/Gate Proj

#     (1792, 6144), # Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic TP8 - QKV Proj
#     (2048, 6144), # Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic TP8 - MLP Up/Gate Proj
#     (3584, 6144), # Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic TP4 - QKV Proj
#     (4096, 5120), # Llama-4-Maverick-17B-128E-Instruct-FP8 TP8 - MLP Up/Gate Proj
#     (4096, 6144), # Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic TP4 - MLP Up/Gate Proj
#     (6144, 768), # Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic TP8 - Attn Output Proj
#     (6144, 1024), # Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic TP8 - MLP Down Proj
#     (6144, 1536), # Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic TP4 - Attn Output Proj
#     (6144, 2048), # Qwen3-Coder-480B-A35B-Instruct-FP8-Dynamic TP4 - MLP Down Proj
#     (8192, 5120), # Llama-4-Maverick-17B-128E-Instruct-FP8 TP4 - MLP Up/Gate Proj
# ]

l_nk = [
    # llama4_tp8
    (896, 5120), 
    (4096, 640), 
    (5120, 640), 
    (5120, 2048),

    #llama4_tp4
    (1024, 5632),
    (1056, 1408),
    (1280, 4096),
    (1408, 352),
    (1408, 1408),
    (4096, 1024),

    #qwen3_tp8 
    (1536, 2048),
    (1792, 6144),
    (2048, 1536), 
    (2048, 6144),
    (6144, 768),
    (6144, 1024),

    #qwen3_tp4 
    (3072, 4096),
    (3584, 6144),
    (4096, 3072),
    (6144, 1536),
    (6144, 2048),

    #deepseek_tp8
    (192, 7168), # maybe this is the shared expert shape
    
    # DeepSeek-R1 TP8
    (1536, 7168),
    (3072, 1536),
    (576, 7168),
    (7168, 256),
    (7168, 2048),
    (4608, 7168),
    (7168, 2304),
    (512, 7168),
    (4096, 512),
    (2112,7168), # This shape is from fused_qkv_a_proj in the MLAModule
]
output_csv_file = 'ptpc_untuned_shapes.csv'

def generate_csv():
    # 1. Configure command-line argument parsing
    parser = argparse.ArgumentParser(description='Generate untuned shapes CSV for different operators.')
    parser.add_argument('operator', type=str, choices=list(OPERATOR_CONFIGS.keys()),
                        help=f'Specify the operator type. Options: {list(OPERATOR_CONFIGS.keys())}')
    
    # Parse parameters
    args = parser.parse_args()
    
    # Get the configuration of the corresponding operator
    config = OPERATOR_CONFIGS[args.operator]
    enable_dtype = config['has_dtype']
    target_dtypes = config['dtype_value']


    print(f"Generating configuration file for [{args.operator}] operator...")
    print(f"Mode: {'4 columns (M, N, K, q_dtype_w)' if enable_dtype else '3 columns (M, N, K)'}")
    if enable_dtype:
        print(f"Data types: {target_dtypes}")

    with open(output_csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # 2. Dynamically generate table headers
        header = ['M', 'N', 'K']
        if enable_dtype:
            header.append('q_dtype_w')
        writer.writerow(header)

        # 3. Write data row
        count = 0
        for m in l_m:
            for n, k in l_nk:
                for dtype in target_dtypes:
                    # Basic data
                    row = [m, n, k]
                    if enable_dtype:
                        row.append(dtype)
                    writer.writerow(row)
                    count += 1
        
    print(f"CSV file '{output_csv_file}' has been generated with {len(l_m) * len(l_nk)} rows.")

if __name__ == "__main__":
    generate_csv()