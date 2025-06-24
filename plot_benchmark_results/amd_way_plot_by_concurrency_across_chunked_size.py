import os
import json
import re
import matplotlib.pyplot as plt
from collections import defaultdict

def process_benchmark_results(results_dir='results'):
    base_dir = results_dir
    all_data = []
    # Regex for ISL-OSL-CON files
    filename_pattern = re.compile(r'ISL-(\d+)-OSL-(\d+)-CON-(\d+)\.json')
    # Regex for sharegpt files
    sharegpt_pattern = re.compile(r'sharegpt-(\d+)\.json')
    
    for folder in os.listdir(base_dir):
        # Skip folders that don't contain 'Chunked-Size'
        if 'Chunked-Size' not in folder:
            continue
            
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            # Extract model name (everything before the first dash followed by "Chunked")
            model_name_match = re.match(r'([^-]+(?:-[^-]+)*?)-Chunked', folder)
            model_name = model_name_match.group(1) if model_name_match else 'unknown-model'
            
            # Updated regex to match numbers instead of letters
            chunked_size_match = re.search(r'Chunked-Size-(\d+)', folder)
            prefix_caching = 'yes-prefix' if 'yes-prefix' in folder else 'no-prefix'
            
            if chunked_size_match:
                chunked_size = int(chunked_size_match.group(1))  # Convert to int for proper sorting
            else:
                chunked_size = None
            
            for file in os.listdir(folder_path):
                if file.endswith('.json'):
                    file_path = os.path.join(folder_path, file)
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Check if file matches ISL-OSL-CON pattern
                    match = filename_pattern.match(file)
                    if match:
                        ISL = int(match.group(1))
                        OSL = int(match.group(2))
                        CON = int(match.group(3))
                        dataset_type = 'ISL-OSL'
                    else:
                        # Check if file matches sharegpt pattern
                        sharegpt_match = sharegpt_pattern.match(file)
                        if sharegpt_match:
                            ISL = None
                            OSL = None
                            CON = int(sharegpt_match.group(1))
                            dataset_type = 'sharegpt'
                        else:
                            continue  # Skip files that don't match either pattern
                    
                    all_data.append({
                        'folder': folder,
                        'model_name': model_name,
                        'chunked_size': chunked_size,
                        'prefix_caching': prefix_caching,
                        'ISL': ISL,
                        'OSL': OSL,
                        'CON': CON,
                        'dataset_type': dataset_type,
                        'data': data
                    })
    
    return all_data

def create_plots(all_data, output_dir='plots'):
    # Group data by model_name, dataset_type, ISL, OSL, prefix_caching
    grouped_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    for entry in all_data:
        ISL = entry['ISL']
        OSL = entry['OSL']
        prefix_caching = entry['prefix_caching']
        chunked_size = entry['chunked_size']
        CON = entry['CON']
        data = entry['data']
        model_name = entry['model_name']
        dataset_type = entry['dataset_type']
        
        # Create different keys for ISL-OSL vs sharegpt datasets
        if dataset_type == 'ISL-OSL':
            key = (model_name, dataset_type, ISL, OSL, prefix_caching)
        else:  # sharegpt
            key = (model_name, dataset_type, None, None, prefix_caching)
        
        # Store data by chunked_size and concurrency
        if key not in grouped_data:
            grouped_data[key] = defaultdict(lambda: defaultdict(dict))
        
        grouped_data[key][chunked_size][CON] = {
            'median_ttft_ms': data['median_ttft_ms'],
            'median_tpot_ms': data['median_tpot_ms'],
            'median_itl_ms': data['median_itl_ms'],
            'output_throughput': data['output_throughput']
        }
    
    # Define metrics to plot
    metrics = {
        'median_ttft_ms': 'Median TTFT (ms)',
        'median_tpot_ms': 'Median TPOT (ms)', 
        'median_itl_ms': 'Median ITL (ms)',
        'output_throughput': 'Output Throughput (tokens/s)'
    }
    
    # Create plots for each combination
    for (model_name, dataset_type, ISL, OSL, prefix_caching), chunked_data in grouped_data.items():
        # Create model-specific directory
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Create separate plot for each metric
        for metric_key, metric_label in metrics.items():
            plt.figure(figsize=(10, 6))
            
            # Store handles and labels for sorting
            plot_handles = []
            plot_labels = []
            
            # Plot each chunked size as a separate line (sorted numerically)
            for chunked_size in sorted(chunked_data.keys()):
                # Get all concurrency values and corresponding metric values for this chunked size
                concurrency_values = sorted(chunked_data[chunked_size].keys())
                metric_values = [chunked_data[chunked_size][con][metric_key] for con in concurrency_values]
                
                # Plot line for this chunked size
                line, = plt.plot(concurrency_values, metric_values, 
                        marker='o', linewidth=2, markersize=6,
                        label=f'Chunked Size {chunked_size}')
                
                # Store handle and label for sorting
                plot_handles.append(line)
                plot_labels.append(f'Chunked Size {chunked_size}')
            
            # Sort legend by chunked size (extract number from label and sort)
            sorted_pairs = sorted(zip(plot_handles, plot_labels), 
                                key=lambda x: int(x[1].split()[-1]))  # Extract number from "Chunked Size X"
            sorted_handles, sorted_labels = zip(*sorted_pairs) if sorted_pairs else ([], [])
            
            plt.xlabel('Concurrency', fontsize=12)
            plt.ylabel(metric_label, fontsize=12)
            
            # Create appropriate title based on dataset type
            if dataset_type == 'ISL-OSL':
                title = f'{metric_label} vs Concurrency\nISL-{ISL} OSL-{OSL} ({prefix_caching})'
            else:  # sharegpt
                title = f'{metric_label} vs Concurrency\nShareGPT Dataset ({prefix_caching})'
            
            plt.title(title, fontsize=14)
            
            # Use sorted handles and labels for legend
            plt.legend(sorted_handles, sorted_labels)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Generate filename based on dataset type
            prefix_str = 'with_prefix' if prefix_caching == 'yes-prefix' else 'without_prefix'
            
            if dataset_type == 'ISL-OSL':
                if metric_key == 'output_throughput':
                    filename = f"throughput_ISL-{ISL}_OSL-{OSL}_{prefix_str}.png"
                else:
                    metric_name = metric_key.replace('median_', '').replace('_ms', '')
                    filename = f"median_{metric_name}_ISL-{ISL}_OSL-{OSL}_{prefix_str}.png"
            else:  # sharegpt
                if metric_key == 'output_throughput':
                    filename = f"throughput_sharegpt_{prefix_str}.png"
                else:
                    metric_name = metric_key.replace('median_', '').replace('_ms', '')
                    filename = f"median_{metric_name}_sharegpt_{prefix_str}.png"
            
            # Save to model-specific directory
            filepath = os.path.join(model_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved plot: {filepath}")

def main(results_directory='results', output_directory='plots'):
    try:
        # Check if results directory exists
        if not os.path.exists(results_directory):
            print(f"Error: Results directory '{results_directory}' not found.")
            return
            
        # Process the benchmark results from specified directory
        all_data = process_benchmark_results(results_directory)
        print(f"Processed {len(all_data)} benchmark results from '{results_directory}'")
        
        # Create plots in specified directory organized by model
        create_plots(all_data, output_directory)
        print(f"All plots have been generated successfully in '{output_directory}' directory!")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # You can specify both directories here
    results_dir = '/path/to/benchmarking/results'
    output_base_dir = '/path/to/output/plots'
    main(results_dir, output_base_dir)
