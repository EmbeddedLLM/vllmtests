import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
from collections import defaultdict


def parse_folder_name(folder_name):
    """Parse folder name to extract model, chunked size, and prefix caching info"""
    # Expected format: model-name-chunked-size-n-no-prefix-caching-yes-aiter
    # or: model-name-chunked-size-n-yes-prefix-caching-yes-aiter
    
    parts = folder_name.split('-')
    
    # The pattern should end with 'yes-aiter'
    if len(parts) < 6 or parts[-2] != 'yes' or parts[-1] != 'aiter':
        print(f"Warning: Unexpected folder name format: {folder_name}")
        return None, None, None
    
    # Find 'chunked' keyword and extract information
    chunked_size = None
    prefix_caching = None
    model_parts = []
    
    i = 0
    while i < len(parts):
        if parts[i] == 'Chunked' and i + 1 < len(parts) and parts[i + 1] == 'Size':
            # Extract chunked size (should be the next part after 'size')
            if i + 2 < len(parts):
                chunked_size = parts[i + 2]
            
            # Look for prefix caching info after chunked size
            # Pattern: chunked-size-n-[no/yes]-prefix-caching-yes-aiter
            if i + 3 < len(parts) and parts[i + 3] in ['no', 'yes']:
                if (i + 5 < len(parts) and 
                    parts[i + 3] in ['no', 'yes'] and 
                    parts[i + 4] == 'prefix' and 
                    parts[i + 5] == 'caching'):
                    prefix_caching = parts[i + 3] == 'yes'
            break
        else:
            model_parts.append(parts[i])
        i += 1
    
    model_name = '-'.join(model_parts) if model_parts else None
    
    return model_name, chunked_size, prefix_caching


def parse_json_filename(filename):
    """Parse JSON filename to extract input/output sequence lengths"""
    if filename == 'sharegpt.json':
        return 'sharegpt', None, None
    
    # Pattern: ISL-n-OSL-m.json
    match = re.match(r'ISL-(\d+)-OSL-(\d+)\.json', filename)
    if match:
        input_len = int(match.group(1))
        output_len = int(match.group(2))
        return 'sequence', input_len, output_len
    
    return None, None, None

def read_benchmark_data(results_dir):
    """Read all benchmark data from the results directory"""
    data = defaultdict(lambda: defaultdict(list))
    
    for folder_name in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        # Skip folders that don't contain "chunked-size" in their name
        if "Chunked-Size" not in folder_name:
            print(f"Skipping folder (no chunked-size): {folder_name}")
            continue
            
        model_name_parsed, chunked_size, prefix_caching = parse_folder_name(folder_name)
        
        # Skip if parsing failed
        if chunked_size is None:
            print(f"Skipping folder (parsing failed): {folder_name}")
            continue
        
        # Read all JSON files in the folder
        for json_file in os.listdir(folder_path):
            if not json_file.endswith('.json'):
                continue
                
            json_path = os.path.join(folder_path, json_file)
            
            try:
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                
                file_type, input_len, output_len = parse_json_filename(json_file)
                
                if file_type:
                    data[folder_name]['metadata'] = {
                        'model_name': model_name_parsed,
                        'chunked_size': chunked_size,
                        'prefix_caching': prefix_caching
                    }
                    
                    data[folder_name]['results'].append({
                        'file_type': file_type,
                        'input_len': input_len,
                        'output_len': output_len,
                        'filename': json_file,
                        'data': json_data
                    })
                    
            except Exception as e:
                print(f"Error reading {json_path}: {e}")
    
    return data


def create_folder_plots(folder_name, folder_data, output_dir):
    """Create plots for a specific folder"""
    results = folder_data['results']
    metadata = folder_data['metadata']
    
    # Separate sequence and sharegpt data
    sequence_data = [r for r in results if r['file_type'] == 'sequence']
    sharegpt_data = [r for r in results if r['file_type'] == 'sharegpt']
    
    if not sequence_data and not sharegpt_data:
        return
    
    # Create output directory for this folder
    folder_output_dir = os.path.join(output_dir, folder_name)
    os.makedirs(folder_output_dir, exist_ok=True)
    
    # Plot 1: TTFT vs Input/Output Length
    if sequence_data:
        create_sequence_length_plots(sequence_data, sharegpt_data, folder_output_dir, 'ttft', 'mean_ttft_ms', 'TTFT (ms)')
        create_sequence_length_plots(sequence_data, sharegpt_data, folder_output_dir, 'tpot', 'mean_tpot_ms', 'TPOT (ms)')
        create_sequence_length_plots(sequence_data, sharegpt_data, folder_output_dir, 'itl', 'mean_itl_ms', 'ITL (ms)')
        create_sequence_length_plots(sequence_data, sharegpt_data, folder_output_dir, 'output_throughput', 'output_throughput', 'Output Throughput (tokens/s)')

def create_sequence_length_plots(sequence_data, sharegpt_data, output_dir, metric_name, metric_key, ylabel):
    """Create bar plots for metrics vs sequence lengths"""
    if not sequence_data:
        return
    
    # Sort by input length for better visualization
    sequence_data.sort(key=lambda x: (x['input_len'], x['output_len']))
    
    # Prepare data
    labels = []
    values = []
    
    for item in sequence_data:
        label = f"ISL-{item['input_len']}\nOSL-{item['output_len']}"
        labels.append(label)
        values.append(item['data'].get(metric_key, 0))
    
    # Add ShareGPT data if available
    if sharegpt_data:
        labels.append('ShareGPT')
        values.append(sharegpt_data[0]['data'].get(metric_key, 0))
    
    # Create plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(labels)), values, color=['skyblue' if 'ISL' in label else 'orange' for label in labels])
    
    plt.xlabel('Dataset Configuration')
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} by Input/Output Sequence Length')
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                f'{value:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{metric_name}_by_sequence_length.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_cross_folder_plots(all_data, output_dir):
    """Create comparison plots across different folders for each dataset configuration"""
    # Organize data by model name, then by dataset configuration, then by chunked size and prefix caching
    model_comparison_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for folder_name, folder_data in all_data.items():
        metadata = folder_data['metadata']
        results = folder_data['results']
        model_name = metadata['model_name']
        
        # Process each result (dataset configuration) in the folder
        for r in results:
            # Create dataset configuration identifier
            if r['file_type'] == 'sharegpt':
                dataset_config = 'sharegpt'
            elif r['file_type'] == 'sequence':
                dataset_config = f"ISL-{r['input_len']}_OSL-{r['output_len']}"
            else:
                continue
            
            # Create key for this configuration (chunked size + prefix caching)
            key = f"size-{metadata['chunked_size']}-{'prefix' if metadata['prefix_caching'] else 'no-prefix'}"
            
            # Store the data organized by model name
            model_comparison_data[model_name][dataset_config][key] = {
                'metadata': metadata,
                'data': r['data']
            }
    
    if not model_comparison_data:
        print("No data for cross-folder comparison")
        return
    
    # Create comparison plots for each model, dataset configuration, and metric
    metrics = [
        ('mean_ttft_ms', 'TTFT (ms)'),
        ('mean_tpot_ms', 'TPOT (ms)'),
        ('mean_itl_ms', 'ITL (ms)'),
        ('output_throughput', 'Output Throughput (tokens/s)')
    ]
    
    for model_name, comparison_data in model_comparison_data.items():
        # Create model-specific output directory
        model_cross_folder_dir = os.path.join(output_dir, f'{model_name}_cross_chunked_sizes')
        os.makedirs(model_cross_folder_dir, exist_ok=True)
        
        for dataset_config, data_by_key in comparison_data.items():
            if len(data_by_key) < 2:  # Need at least 2 data points for comparison
                continue
                
            for metric_key, ylabel in metrics:
                create_cross_folder_line_plot(data_by_key, metric_key, ylabel, model_cross_folder_dir, dataset_config)

def create_cross_folder_line_plot(data_by_key, metric_key, ylabel, output_dir, dataset_config):
    """Create line plots comparing metrics across folders for a specific dataset configuration"""
    prefix_groups = defaultdict(list)
    
    for key, data in data_by_key.items():
        prefix_status = 'With Prefix Caching' if data['metadata']['prefix_caching'] else 'Without Prefix Caching'
        
        # Convert chunked_size to int for proper sorting
        try:
            chunked_size = int(data['metadata']['chunked_size'])
        except (ValueError, TypeError):
            chunked_size_str = str(data['metadata']['chunked_size'])
            numeric_part = ''.join(filter(str.isdigit, chunked_size_str))
            chunked_size = int(numeric_part) if numeric_part else 0
        
        metric_value = data['data'].get(metric_key, 0)
        prefix_groups[prefix_status].append((chunked_size, metric_value))
    
    # Skip if no data
    if not prefix_groups:
        return
    
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'red']
    markers = ['o', 's']
    
    for i, (prefix_status, data_points) in enumerate(prefix_groups.items()):
        # Sort by chunked size (numeric sorting)
        data_points.sort(key=lambda x: x[0])
        
        # Extract sizes and values separately
        sizes = [point[0] for point in data_points]
        values = [point[1] for point in data_points]
        
        plt.plot(sizes, values, marker=markers[i], color=colors[i], 
                label=prefix_status, linewidth=2, markersize=8)
        
        # Add value labels
        for size, value in zip(sizes, values):
            plt.annotate(f'{value:.1f}', (size, value), 
                        textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.xlabel('Chunked Size')
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} Comparison: Chunked Size vs Prefix Caching - {dataset_config}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set x-axis to show chunked sizes in ascending order
    all_sizes = []
    for data_points in prefix_groups.values():
        all_sizes.extend([point[0] for point in data_points])
    
    if all_sizes:
        unique_sizes = sorted(set(all_sizes))
        plt.xticks(unique_sizes)
    
    plt.tight_layout()
    
    # Create filename with dataset configuration
    clean_metric_name = metric_key.replace('_', '-').replace('mean-', '')
    filename = f'comparison_{clean_metric_name}_{dataset_config}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function to orchestrate the plotting process"""
    results_dir = '/path/to/benchmarking/result/folder'
    output_base_dir = '/path/to/benchmarking/output/plot'
    
    if not os.path.exists(results_dir):
        print(f"Results directory '{results_dir}' not found!")
        return
    
    # Create base output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Read all benchmark data
    print("Reading benchmark data...")
    all_data = read_benchmark_data(results_dir)
    
    if not all_data:
        print("No data found!")
        return
    
    print(f"Found data for {len(all_data)} folders")
    
    # Create plots for each folder
    for folder_name, folder_data in all_data.items():
        print(f"Creating plots for {folder_name}...")
        create_folder_plots(folder_name, folder_data, output_base_dir)
    
    # Create cross-folder comparison plots
    print("Creating cross-folder comparison plots...")
    create_cross_folder_plots(all_data, output_base_dir)
    
    print(f"All plots saved to '{output_base_dir}' directory")

if __name__ == "__main__":
    main()
