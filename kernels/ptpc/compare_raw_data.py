import pandas as pd
import csv

def generate_comparison_csv():
    # Read the two CSV files
    df_bpreshuffle = pd.read_csv('mi300x_tuned_configs/a8w8_bpreshuffle_tuned_gemm.csv')
    df_regular = pd.read_csv('mi300x_tuned_configs/a8w8_tuned_gemm.csv')
    
    # Get the key columns
    key_cols = ['cu_num', 'M', 'N', 'K']
    
    # Create key tuples for both dataframes
    bpreshuffle_keys = set(df_bpreshuffle[key_cols].apply(tuple, axis=1))
    regular_keys = set(df_regular[key_cols].apply(tuple, axis=1))
    
    # Get union of all keys
    all_keys = bpreshuffle_keys.union(regular_keys)
    
    # Create dictionaries for fast lookup
    bpreshuffle_dict = {}
    for _, row in df_bpreshuffle.iterrows():
        key = tuple(row[key_cols])
        bpreshuffle_dict[key] = row['us']
    
    regular_dict = {}
    for _, row in df_regular.iterrows():
        key = tuple(row[key_cols])
        regular_dict[key] = row['us']
    
    # Generate the result data
    result_data = []
    
    for key in all_keys:
        cu_num, M, N, K = key
        
        # Get us values, default to 10000000 if missing
        us = regular_dict.get(key, 10000000)
        us_bpreshuffle = bpreshuffle_dict.get(key, 10000000)
        
        # Check if bpreshuffle is faster
        is_bpreshuffle_faster = us_bpreshuffle < us
        
        # Check if any entry is missing
        missing_entry = (key not in regular_dict) or (key not in bpreshuffle_dict)
        
        result_data.append({
            'cu_num': cu_num,
            'M': M,
            'N': N,
            'K': K,
            'us': us,
            'us_bpreshuffle': us_bpreshuffle,
            'is_bpreshuffle_faster': is_bpreshuffle_faster,
            'missing_entry': missing_entry
        })
    
    # Sort by cu_num, M, N, K for consistent output
    result_data.sort(key=lambda x: (x['cu_num'], x['M'], x['N'], x['K']))
    
    # Write to CSV file
    output_filename = 'comparison_result.csv'
    fieldnames = ['cu_num', 'M', 'N', 'K', 'us', 'us_bpreshuffle', 'is_bpreshuffle_faster', 'missing_entry']
    
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_data)
    
    print(f"Comparison CSV file generated: {output_filename}")
    print(f"Total entries: {len(result_data)}")
    print(f"Entries where bpreshuffle is faster: {sum(1 for row in result_data if row['is_bpreshuffle_faster'])}")
    print(f"Missing entries: {sum(1 for row in result_data if row['missing_entry'])}")

if __name__ == "__main__":
    generate_comparison_csv()