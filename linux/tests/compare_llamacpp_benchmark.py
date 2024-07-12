import pandas as pd
import argparse
from tabulate import tabulate

def load_and_process_csv(file_path):
    df = pd.read_csv(file_path)
    # Split the 'Model' column into model info and file name
    df[['Model_Info', 'File']] = df['Model'].str.split(' - ', expand=True)
    # Split the model info into type, variant, and size
    df[['Model_Type', 'Variant', 'Size']] = df['Model_Info'].str.split(' ', expand=True, n=2)
    # Convert t/s to numeric, removing the '±' part
    df['t/s'] = df['t/s'].str.split('±').str[0].astype(float)
    return df

def compare_performance(file1, file2):
    df1 = load_and_process_csv(file1)
    df2 = load_and_process_csv(file2)

    # Merge the two dataframes, preserving the original order
    merged_df = pd.merge(df1, df2, on=['File', 'test', 'Model_Type', 'Variant', 'Size'], how='left', suffixes=('_1', '_2'))

    # Calculate performance difference
    merged_df['performance_diff'] = merged_df['t/s_2'] - merged_df['t/s_1']
    merged_df['performance_diff_percent'] = (merged_df['performance_diff'] / merged_df['t/s_1']) * 100

    # Sort by the original order (File)
    merged_df = merged_df.sort_values('File')

    # Prepare the results
    results = []
    current_model = None
    for _, row in merged_df.iterrows():
        if current_model != row['File']:
            if current_model is not None:
                results.append([])  # Add an empty row
            current_model = row['File']
        results.append([row['File'], row['Model_Type'], row['Variant'], row['Size'], row['test'], f"{row['t/s_1']:.2f}", f"{row['t/s_2']:.2f}", f"{row['performance_diff']:.2f}", f"{row['performance_diff_percent']:.2f}"])

    # Rename the columns
    columns = ['File', 'Model_Type', 'Variant', 'Size', 'test', f't/s ({file1})', f't/s ({file2})', 'Diff (t/s)', 'Diff (%)']
    results = pd.DataFrame(results, columns=columns)

    return results

def main():
    parser = argparse.ArgumentParser(description="Compare performance between two CSV files")
    parser.add_argument("file1", help="Path to the first CSV file")
    parser.add_argument("file2", help="Path to the second CSV file")
    parser.add_argument("--output", help="Path to save the comparison results (optional)")
    args = parser.parse_args()

    results = compare_performance(args.file1, args.file2)

    # Print the results
    print(tabulate(results, headers='keys', tablefmt='pipe', floatfmt=".2f"))

    # Save to file if output path is provided
    if args.output:
        results.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
