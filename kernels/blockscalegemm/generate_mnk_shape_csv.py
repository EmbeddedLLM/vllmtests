import csv

l_m = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256, 264, 272, 280, 288, 296, 304, 312, 320, 328, 336, 344, 352, 360, 368, 376, 384, 392, 400, 408, 416, 424, 432, 440, 448, 456, 464, 472, 480, 488, 496, 504, 512, 1024, 8192]

l_nk = [
    # DeepSeek-R1
    (1536, 7168),
    (3072, 1536),
    (576, 7168),
    (7168, 256),
    (7168, 2048),
    (4608, 7168),
    (7168, 2304),
    (512, 7168),
    (4096, 512),
    (2112,7168), # from vllm shared expert

    # # Qwen3-235B-FP8
    (1280,4096), # TP8
    (4096,1024), # TP8
    
    # Qwen3-235B-A22B-Instruct-2507-FP8 TP 4
    (2304,4096), # TP4
    (4096,2048), # TP4
    
    # Qwen3-Coder-480B-FP8
    (1792,6144), # TP8
    (6144,1536), # TP8
    (3584, 6144), # TP4
    (6144, 3072), # TP4
]

# Generate CSV file
with open('matrix_dimensions.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write header
    writer.writerow(['M', 'N', 'K'])
    
    # Write all combinations of M with each (N, K) pair
    for m in l_m:
        for n, k in l_nk:
            writer.writerow([m, n, k])

print(f"CSV file 'matrix_dimensions.csv' has been generated with {len(l_m) * len(l_nk)} rows.")