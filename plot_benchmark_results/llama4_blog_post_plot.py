import os
import json
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def parse_results_directory(base_dir='results'):
    """Parse all JSON files in the results directory structure."""
    results_data = defaultdict(dict)
    
    # Regex to extract ISL and OSL from filename
    pattern = re.compile(r"ISL-(\d+)-OSL-(\d+)\.json")
    
    # Loop through each folder in the base directory
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.json'):
                    match = pattern.match(file)
                    if match:
                        ISL, OSL = int(match.group(1)), int(match.group(2))
                        file_path = os.path.join(folder_path, file)
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        results_data[folder][(ISL, OSL)] = data
    
    return results_data

## if need to plot everything in one figure

# def extract_legend_info(folder_name):
#     """Extract aiter and prefix caching status from folder name."""
#     aiter = 'yes' if 'yes-aiter' in folder_name else 'no'
#     prefix_caching = 'yes' if 'yes-prefix-caching' in folder_name else 'no'
#     return f'aiter: {aiter}, prefix caching: {prefix_caching}'

# def create_plots(results_data, save_dir='plots'):
#     """Create bar plots for each metric and save them."""
#     # Create save directory if it doesn't exist
#     os.makedirs(save_dir, exist_ok=True)
    
#     # Get all folders and ISL/OSL pairs
#     folders = list(results_data.keys())
#     all_isl_osl_pairs = set()
#     for folder_data in results_data.values():
#         all_isl_osl_pairs.update(folder_data.keys())
#     isl_osl_pairs = sorted(list(all_isl_osl_pairs))
    
#     # Define metrics to plot
#     metrics = {
#         'mean_ttft_ms': 'Mean Time to First Token (ms)',
#         'mean_tpot_ms': 'Mean Time per Output Token (ms)', 
#         'mean_itl_ms': 'Mean Inter-Token Latency (ms)',
#         'output_throughput': 'Output Throughput (tokens/s)'
#     }
    
#     # Prepare data for plotting
#     plot_data = {metric: {folder: [] for folder in folders} for metric in metrics.keys()}
    
#     # Fill the data
#     for folder in folders:
#         for isl_osl in isl_osl_pairs:
#             if isl_osl in results_data[folder]:
#                 data = results_data[folder][isl_osl]
#                 for metric in metrics.keys():
#                     plot_data[metric][folder].append(data[metric])
#             else:
#                 # Fill with None or 0 if data is missing
#                 for metric in metrics.keys():
#                     plot_data[metric][folder].append(0)
    
#     # Generate legends
#     legends = [extract_legend_info(folder) for folder in folders]
    
#     # Prepare x-axis labels
#     x_labels = [f'{isl}/{osl}' for isl, osl in isl_osl_pairs]
    
#     # Create plots for each metric
#     for metric, title in metrics.items():
#         fig, ax = plt.subplots(figsize=(12, 8))
#         bar_width = 0.2
#         indices = np.arange(len(x_labels))
        
#         # Create bars for each configuration
#         for i, folder in enumerate(folders):
#             ax.bar(indices + i * bar_width, plot_data[metric][folder], 
#                    bar_width, label=legends[i], alpha=0.8)
        
#         # Customize the plot
#         ax.set_xlabel('ISL/OSL', fontsize=12)
#         ax.set_ylabel(title, fontsize=12)
#         ax.set_title(f'{title} vs ISL/OSL', fontsize=14, fontweight='bold')
#         ax.set_xticks(indices + bar_width * (len(folders) - 1) / 2)
#         ax.set_xticklabels(x_labels)
#         ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#         ax.grid(True, alpha=0.3)
        
#         # Save the plot
#         plt.tight_layout()
#         filename = f'{metric}_comparison.png'
#         filepath = os.path.join(save_dir, filename)
#         plt.savefig(filepath, dpi=300, bbox_inches='tight')
#         plt.close()
        
#         print(f"Saved plot: {filepath}")

def extract_legend_info(folder_name):
    aiter = 'with' if 'yes-aiter' in folder_name else 'without'
    prefix_caching = 'with' if 'yes-prefix-caching' in folder_name else 'without'
    return aiter, prefix_caching

def create_plots_separated(results_data, save_dir='plots'):
    """Create separate bar plots for each prefix caching configuration."""
    os.makedirs(save_dir, exist_ok=True)
    
    folders = list(results_data.keys())
    all_isl_osl_pairs = set()
    for folder_data in results_data.values():
        all_isl_osl_pairs.update(folder_data.keys())
    isl_osl_pairs = sorted(list(all_isl_osl_pairs))
    
    metrics = {
        'mean_ttft_ms': 'Mean Time to First Token (ms)',
        'mean_tpot_ms': 'Mean Time per Output Token (ms)', 
        'mean_itl_ms': 'Mean Inter-Token Latency (ms)',
        'output_throughput': 'Output Throughput (tokens/s)'
    }
    
    # Group folders by prefix caching status
    prefix_groups = {'with': [], 'without': []}
    for folder in folders:
        _, prefix_caching = extract_legend_info(folder)
        prefix_groups[prefix_caching].append(folder)
    
    x_labels = [f'{isl}/{osl}' for isl, osl in isl_osl_pairs]
    
    # Create separate plots for each prefix caching configuration
    for prefix_status, group_folders in prefix_groups.items():
        for metric, title in metrics.items():
            fig, ax = plt.subplots(figsize=(12, 8))
            bar_width = 0.35
            indices = np.arange(len(x_labels))
            
            for i, folder in enumerate(group_folders):
                values = []
                for isl_osl in isl_osl_pairs:
                    if isl_osl in results_data[folder]:
                        values.append(results_data[folder][isl_osl][metric])
                    else:
                        values.append(0)
                
                aiter_status, _ = extract_legend_info(folder)
                ax.bar(indices + i * bar_width, values, bar_width, 
                       label=f'{aiter_status} aiter', alpha=0.8)
            
            ax.set_xlabel('ISL/OSL', fontsize=12)
            ax.set_ylabel(title, fontsize=12)
            ax.set_title(f'{title} vs ISL/OSL ({prefix_status} prefix caching)', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(indices + bar_width * (len(group_folders) - 1) / 2)
            ax.set_xticklabels(x_labels)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            filename = f'{metric}_comparison_{prefix_status}_prefix.png'
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved plot: {filepath}")

def main():
    """Main function to run the complete analysis."""
    results_dir = '/path/to/benchmarking/result/folder'
    output_base_dir = '/path/to/benchmarking/output/plot'
    # Parse the results
    print("Parsing results directory...")
    results_data = parse_results_directory(results_dir)
    
    if not results_data:
        print("No data found. Please check if the 'results' directory exists and contains the expected structure.")
        return
    
    print(f"Found data for {len(results_data)} configurations:")
    for folder in results_data.keys():
        print(f"  - {folder}: {len(results_data[folder])} ISL/OSL combinations")
    
    # Create plots
    print("\nGenerating plots...")
    create_plots_separated(results_data, save_dir=output_base_dir)
    
    print("\nAnalysis complete! Check the 'plots' directory for generated visualizations.")

if __name__ == "__main__":
    main()
