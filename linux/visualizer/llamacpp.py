import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import re
import numpy as np

def analyze_llm_benchmarks(csv_file):
    """
    Analyzes LLM benchmark data and creates visualization charts.
    
    Args:
        csv_file (str): Path to the CSV file containing benchmark data
    """
    # Read data
    try:
        # Use engine='python' to help with odd formatting and asterisks in column names
        df = pd.read_csv(csv_file, engine='python')
        
        # Clean column names by removing asterisks
        df.columns = [col.replace('*', '') for col in df.columns]
        
        # Parse the tokens per second column
        # The format seems to be "value ± uncertainty"
        df['tokens_per_second'] = df['t/s'].apply(lambda x: float(x.split('±')[0].strip()))
        df['tokens_per_second_error'] = df['t/s'].apply(lambda x: float(x.split('±')[1].strip()))
        
        # Extract model name (base model), quantization type
        df['base_model'] = df['model'].apply(lambda x: x.split()[0].lower())
        
        # Extract model size in GB from the size column
        df['size_gb'] = df['size'].apply(lambda x: float(re.search(r'(\d+\.\d+)\s*GiB', x).group(1)))
        
        # Extract quantization from model column
        df['quantization'] = df['model'].apply(lambda x: re.search(r'Q\d+_\w+', x).group(0) if re.search(r'Q\d+_\w+', x) else 'Unknown')
        
        print(f"Loaded {len(df)} benchmark records")
        
        # Create visualizations
        create_visualizations(df)
        
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def create_visualizations(df):
    """
    Creates visualizations from the LLM benchmark data.
    
    Args:
        df (DataFrame): Pandas DataFrame containing the benchmark data
    """
    # Set style
    sns.set(style="whitegrid")
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Create a GridSpec layout
    gs = plt.GridSpec(2, 2, figure=plt.gcf())
    
    # 1. Tokens per second by test type and quantization
    ax1 = plt.subplot(gs[0, 0])
    sns.barplot(x='test', y='tokens_per_second', hue='quantization', data=df, ax=ax1)
    ax1.set_title('LLM Performance by Test Type and Quantization')
    ax1.set_xlabel('Test Type')
    ax1.set_ylabel('Tokens per Second')
    
    # Add error bars
    for i, bar in enumerate(ax1.patches):
        idx = i % len(df)
        error = df.iloc[idx]['tokens_per_second_error']
        x = bar.get_x() + bar.get_width() / 2
        y = bar.get_height()
        ax1.errorbar(x, y, yerr=error, fmt='none', color='black', capsize=5)
    
    # 2. Comparison of model size vs performance
    ax2 = plt.subplot(gs[0, 1])
    sns.scatterplot(x='size_gb', y='tokens_per_second', hue='test', style='quantization', 
                   s=200, data=df, ax=ax2)
    ax2.set_title('Size vs Performance')
    ax2.set_xlabel('Model Size (GiB)')
    ax2.set_ylabel('Tokens per Second')
    
    # 3. Performance by test type with error bars
    ax3 = plt.subplot(gs[1, :])
    
    # Prepare data for grouped bar chart with model on x-axis
    model_data = []
    for _, row in df.iterrows():
        model_data.append({
            'model_name': f"{row['base_model']} {row['quantization']}",
            'test': row['test'],
            'tokens_per_second': row['tokens_per_second'],
            'error': row['tokens_per_second_error']
        })
    
    model_df = pd.DataFrame(model_data)
    
    # Create the plot with custom error bars
    sns.barplot(x='model_name', y='tokens_per_second', hue='test', data=model_df, ax=ax3)
    
    # Add error bars
    bars = ax3.patches
    errors = model_df['error'].values
    bar_centers = np.array([bar.get_x() + bar.get_width() / 2 for bar in bars])
    bar_heights = np.array([bar.get_height() for bar in bars])
    
    # We need to be careful with the indexing of errors since they correspond to rows in model_df
    for i, (x, y) in enumerate(zip(bar_centers, bar_heights)):
        ax3.errorbar(x, y, yerr=errors[i//2], fmt='none', color='black', capsize=5)
    
    ax3.set_title('Performance Comparison by Model and Test Type')
    ax3.set_xlabel('Model')
    ax3.set_ylabel('Tokens per Second (higher is better)')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
    
    # Add a log scale option for y-axis if the numbers vary widely
    if max(df['tokens_per_second']) / min(df['tokens_per_second']) > 10:
        ax1.set_yscale('log')
        ax3.set_yscale('log')
    
    # Create a detailed text summary
    summary_text = f"""
    Model: {df['Model'].iloc[0].split('-')[0].strip()}
    Parameter Count: {df['params'].iloc[0]}
    Backend: {df['backend'].iloc[0]}
    NGL Layers: {df['ngl'].iloc[0]}
    """
    
    plt.figtext(0.5, 0.01, summary_text, ha='center', fontsize=10, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=1'))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig('llm_benchmark_analysis.png', dpi=300)
    
    print("Analysis complete! Visualization saved as 'llm_benchmark_analysis.png'")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 50)
    for test in df['test'].unique():
        test_df = df[df['test'] == test]
        print(f"\nTest: {test}")
        
        # Find best performer for this test
        best_idx = test_df['tokens_per_second'].idxmax()
        best_model = test_df.loc[best_idx]
        
        print(f"  Best Performance:")
        print(f"    - Model: {best_model['model']}")
        print(f"    - Tokens per Second: {best_model['tokens_per_second']:.2f} ± {best_model['tokens_per_second_error']:.2f}")
        print(f"    - Size: {best_model['size']}")
        print(f"    - Params: {best_model['params']}")
        
        # Performance comparison between quantizations
        if len(test_df['quantization'].unique()) > 1:
            print("\n  Quantization Comparison:")
            for quant in test_df['quantization'].unique():
                quant_data = test_df[test_df['quantization'] == quant]
                print(f"    - {quant}: {quant_data['tokens_per_second'].values[0]:.2f} t/s, "
                      f"Size: {quant_data['size'].values[0]}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 script.py file.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    analyze_llm_benchmarks(csv_file)
