import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import re
import numpy as np
from matplotlib.gridspec import GridSpec

def analyze_stable_diffusion_benchmarks(csv_file):
    """
    Analyzes Stable Diffusion benchmark data and creates visualization charts.
    Handles multiple models and GPU types.
    
    Args:
        csv_file (str): Path to the CSV file containing benchmark data
    """
    # Read data
    try:
        # Use engine='python' to help with odd formatting and asterisks in column names
        df = pd.read_csv(csv_file, engine='python')
        
        # Clean column names by removing asterisks
        df.columns = [col.replace('*', '') for col in df.columns]
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Extract key information for analysis
        df['precision'] = df['precision'].astype(str)
        df['batch_size'] = df['batch_size'].astype(int)
        df['throughput'] = df['throughput'].astype(float)
        df['time_per_image'] = df['time_per_image'].astype(float)
        
        # Clean up GPU info - extract just the GPU model name
        df['gpu_model'] = df['gpu_info'].apply(lambda x: re.sub(r'\s*\([^)]*\)', '', x).strip())
        
        # Extract model version for simpler comparison
        df['model_version'] = df['model_name'].apply(lambda x: x.split('-')[-1])
        
        # Check how many different models and GPUs we have
        models = df['model_version'].unique()
        gpus = df['gpu_model'].unique()
        
        print(f"Loaded {len(df)} benchmark records")
        print(f"Found {len(models)} models: {', '.join(models)}")
        print(f"Found {len(gpus)} GPUs: {', '.join(gpus)}")
        
        # Create visualizations based on number of unique models and GPUs
        create_visualizations(df, models, gpus)
        
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

def create_visualizations(df, models, gpus):
    """
    Creates visualizations from the benchmark data, handling multiple models and GPUs.
    
    Args:
        df (DataFrame): Pandas DataFrame containing the benchmark data
        models (array): Unique model versions
        gpus (array): Unique GPU models
    """
    # Set style
    sns.set(style="whitegrid")
    
    # Create output directory for multiple charts if needed
    import os
    if not os.path.exists('benchmark_charts'):
        os.makedirs('benchmark_charts')
    
    # ----- 1. Create overview comparisons -----
    create_overview_comparison(df, models, gpus)
    
    # ----- 2. Create model-specific comparisons if we have multiple models -----
    if len(models) > 1:
        create_model_comparison(df, models, gpus)
    
    # ----- 3. Create GPU-specific comparisons if we have multiple GPUs -----
    if len(gpus) > 1:
        create_gpu_comparison(df, models, gpus)
    
    # ----- 4. Create detailed precision and batch size analysis for each model-GPU combo -----
    create_detailed_analysis(df, models, gpus)
    
    print(f"\nAnalysis complete! Visualizations saved in 'benchmark_charts/' directory")
    
    # Print summary statistics
    print_summary_statistics(df, models, gpus)

def create_overview_comparison(df, models, gpus):
    """Creates overview comparison charts"""
    plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=plt.gcf())
    
    # 1. Throughput comparison by model
    ax1 = plt.subplot(gs[0, 0])
    model_throughput = df.groupby(['model_version', 'precision'])['throughput'].mean().reset_index()
    sns.barplot(x='model_version', y='throughput', hue='precision', data=model_throughput, ax=ax1)
    ax1.set_title('Average Throughput by Model and Precision')
    ax1.set_xlabel('Model Version')
    ax1.set_ylabel('Throughput (images/sec)')
    
    # 2. Throughput comparison by GPU
    ax2 = plt.subplot(gs[0, 1])
    gpu_throughput = df.groupby(['gpu_model', 'precision'])['throughput'].mean().reset_index()
    sns.barplot(x='gpu_model', y='throughput', hue='precision', data=gpu_throughput, ax=ax2)
    ax2.set_title('Average Throughput by GPU and Precision')
    ax2.set_xlabel('GPU Model')
    ax2.set_ylabel('Throughput (images/sec)')
    
    # 3. Time per image comparison across models and batch sizes
    ax3 = plt.subplot(gs[1, 0])
    model_time = df.groupby(['model_version', 'batch_size'])['time_per_image'].mean().reset_index()
    sns.lineplot(x='batch_size', y='time_per_image', hue='model_version', 
                marker='o', markersize=8, data=model_time, ax=ax3)
    ax3.set_title('Avg Time per Image by Model and Batch Size')
    ax3.set_xlabel('Batch Size')
    ax3.set_ylabel('Time per Image (seconds)')
    
    # 4. Batch scaling efficiency by GPU
    ax4 = plt.subplot(gs[1, 1])
    # Calculate theoretical vs actual speedup
    pivot_df = df.pivot_table(
        index=['gpu_model', 'model_version', 'precision'], 
        columns='batch_size', 
        values='time_per_batch'
    ).reset_index()
    
    scaling_data = []
    for _, row in pivot_df.iterrows():
        if 1 in pivot_df.columns and row[1] is not None:  # Check if batch size 1 exists
            base_time = row[1]  # Time for batch_size=1
            for batch_size in [col for col in pivot_df.columns if isinstance(col, int)]:
                if row[batch_size] is not None:
                    scaling_data.append({
                        'gpu_model': row['gpu_model'],
                        'batch_size': batch_size,
                        'speedup': base_time / (row[batch_size] / batch_size),  # Actual vs ideal scaling
                        'model_version': row['model_version'],
                        'precision': row['precision']
                    })
    
    scaling_df = pd.DataFrame(scaling_data)
    sns.lineplot(x='batch_size', y='speedup', hue='gpu_model', 
                marker='o', markersize=8, data=scaling_df, ax=ax4)
    ax4.axhline(y=1, linestyle='--', color='gray', alpha=0.7)
    ax4.set_title('Batch Scaling Efficiency by GPU (Higher is Better)')
    ax4.set_xlabel('Batch Size')
    ax4.set_ylabel('Speedup (relative to perfect scaling)')
    
    plt.tight_layout()
    plt.savefig('benchmark_charts/overview_comparison.png', dpi=300)
    plt.close()

def create_model_comparison(df, models, gpus):
    """Creates model comparison charts"""
    plt.figure(figsize=(15, 10))
    
    # 1. Model comparison by throughput for different batch sizes
    plt.subplot(2, 1, 1)
    sns.barplot(x='model_version', y='throughput', hue='batch_size', data=df)
    plt.title('Model Comparison: Throughput by Batch Size')
    plt.xlabel('Model Version')
    plt.ylabel('Throughput (images/sec)')
    
    # 2. Model comparison with precision and GPU factors
    plt.subplot(2, 1, 2)
    model_precision_gpu = df.groupby(['model_version', 'precision', 'gpu_model'])['throughput'].mean().reset_index()
    sns.barplot(x='model_version', y='throughput', hue='precision', data=model_precision_gpu)
    plt.title('Model Comparison: Throughput by Precision and GPU')
    plt.xlabel('Model Version')
    plt.ylabel('Throughput (images/sec)')
    
    plt.tight_layout()
    plt.savefig('benchmark_charts/model_comparison.png', dpi=300)
    plt.close()

def create_gpu_comparison(df, models, gpus):
    """Creates GPU comparison charts"""
    plt.figure(figsize=(15, 10))
    
    # 1. GPU throughput comparison across all models
    plt.subplot(2, 1, 1)
    sns.barplot(x='gpu_model', y='throughput', hue='model_version', data=df)
    plt.title('GPU Comparison: Throughput by Model')
    plt.xlabel('GPU Model')
    plt.ylabel('Throughput (images/sec)')
    
    # 2. GPU scaling with batch size
    plt.subplot(2, 1, 2)
    batch_gpu = df.groupby(['gpu_model', 'batch_size', 'precision'])['throughput'].mean().reset_index()
    # Create a unique combination of batch_size and precision for x-axis
    batch_gpu['batch_precision'] = batch_gpu['batch_size'].astype(str) + '-' + batch_gpu['precision']
    sns.barplot(x='batch_precision', y='throughput', hue='gpu_model', data=batch_gpu)
    plt.title('GPU Scaling: Throughput by Batch Size and Precision')
    plt.xlabel('Batch Size - Precision')
    plt.ylabel('Throughput (images/sec)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('benchmark_charts/gpu_comparison.png', dpi=300)
    plt.close()

def create_detailed_analysis(df, models, gpus):
    """Creates detailed analysis for each model-GPU combination"""
    
    # Create a combination chart for each model
    for model in models:
        model_df = df[df['model_version'] == model]
        
        plt.figure(figsize=(15, 12))
        fig_title = f'Detailed Analysis: {model}'
        plt.suptitle(fig_title, fontsize=16)
        
        # 1. Precision comparison across batch sizes
        plt.subplot(2, 2, 1)
        sns.barplot(x='batch_size', y='throughput', hue='precision', data=model_df)
        plt.title(f'Throughput by Precision and Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Throughput (images/sec)')
        
        # 2. GPU comparison for this model
        if len(gpus) > 1:
            plt.subplot(2, 2, 2)
            sns.barplot(x='gpu_model', y='throughput', hue='precision', data=model_df)
            plt.title(f'Throughput by GPU and Precision')
            plt.xlabel('GPU Model')
            plt.ylabel('Throughput (images/sec)')
        
        # 3. Time per image across batch sizes
        plt.subplot(2, 2, 3)
        sns.lineplot(x='batch_size', y='time_per_image', hue='precision', 
                     style='gpu_model', markers=True, data=model_df)
        plt.title(f'Time per Image by Batch Size')
        plt.xlabel('Batch Size')
        plt.ylabel('Time per Image (seconds)')
        
        # 4. Total batch time scaling
        plt.subplot(2, 2, 4)
        sns.lineplot(x='batch_size', y='time_per_batch', hue='precision', 
                     style='gpu_model', markers=True, data=model_df)
        plt.title(f'Total Batch Time Scaling')
        plt.xlabel('Batch Size')
        plt.ylabel('Time per Batch (seconds)')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'benchmark_charts/detailed_{model}.png', dpi=300)
        plt.close()

def print_summary_statistics(df, models, gpus):
    """Prints summary statistics for benchmark data"""
    print("\nSummary Statistics:")
    print("-" * 70)
    
    # Create summary tables for throughput
    print("\nThroughput Comparison (images/second):")
    print("-" * 70)
    
    # Create a pivot table for model-GPU-precision combinations
    pivot = pd.pivot_table(
        df, 
        values='throughput',
        index=['model_version', 'gpu_model', 'precision'],
        columns='batch_size',
        aggfunc='mean'
    )
    
    print(pivot.round(4))
    
    # Create a summary of time per image
    print("\nTime per Image Comparison (seconds):")
    print("-" * 70)
    
    time_pivot = pd.pivot_table(
        df, 
        values='time_per_image',
        index=['model_version', 'gpu_model', 'precision'],
        columns='batch_size',
        aggfunc='mean'
    )
    
    print(time_pivot.round(4))
    
    # Find the overall best performer
    best_throughput = df.loc[df['throughput'].idxmax()]
    print("\nBest Performance Configuration:")
    print("-" * 70)
    print(f"Model: {best_throughput['model_version']}")
    print(f"GPU: {best_throughput['gpu_model']}")
    print(f"Precision: {best_throughput['precision']}")
    print(f"Batch Size: {best_throughput['batch_size']}")
    print(f"Throughput: {best_throughput['throughput']:.4f} images/second")
    print(f"Time per Image: {best_throughput['time_per_image']:.4f} seconds")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 script.py file.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    analyze_stable_diffusion_benchmarks(csv_file)