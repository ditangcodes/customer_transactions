import cProfile
import pstats
import io
import pandas as pd
from concurrent.futures import ProcessPoolExecutor


# Define file paths for CSV files and directories
accounts_csv = '../landing/accounts.csv'
customers_csv = '../landing/customers.csv'
transactions_csv = '../landing/transactions.csv'
staging_dir = '../staging'
transformed_dir = '../transformed'

# Function to load and process data
def load_and_process_data():
    # Task 1: Load data from CSV files
    accounts_df = pd.read_csv(accounts_csv)
    customers_df = pd.read_csv(customers_csv)
    transactions_df = pd.read_csv(transactions_csv)
    
    # Clean data: Strip leading and trailing spaces from column names
    accounts_df.columns = accounts_df.columns.str.strip()
    customers_df.columns = customers_df.columns.str.strip()
    transactions_df.columns = transactions_df.columns.str.strip()
    
    # Save cleaned DataFrames to Parquet format in the staging directory
    accounts_df.to_parquet(f'{staging_dir}/accounts_cleaned.parquet', index=False)
    customers_df.to_parquet(f'{staging_dir}/customers_cleaned.parquet', index=False)
    transactions_df.to_parquet(f'{staging_dir}/transactions_cleaned.parquet', index=False)

# Function to profile the execution time of load_and_process_data
def profile_code():
    pr = cProfile.Profile()
    pr.enable()  
    
    # Call the function you want to profile
    load_and_process_data()
    
    pr.disable()  # Stop profiling
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()  # Print cumulative profiling statistics
    print(s.getvalue())  # Print profile results

if __name__ == "__main__":
    # Profile the load_and_process_data function when this script is run directly
    profile_code()

# Function to read and clean a CSV file
def read_and_clean_csv(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # Clean column names
    return df

# Function to parallelize loading and cleaning of multiple CSV files
def parallel_load_data(files):
    with ProcessPoolExecutor() as executor:
        data_frames = list(executor.map(read_and_clean_csv, files))
    return data_frames

if __name__ == "__main__":
    # Parallelize loading and cleaning of CSV files when this script is run directly
    files = [accounts_csv, customers_csv, transactions_csv]
    data_frames = parallel_load_data(files)

    # Save cleaned DataFrames to Parquet format in the staging directory
    for i, df in enumerate(data_frames):
        df.to_parquet(f'{staging_dir}/file_{i}_cleaned.parquet', index=False)

# Function to optimize memory usage of a DataFrame
def optimize_memory_usage(df):
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='unsigned')  
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')  
    return df

if __name__ == "__main__":
    # Optimize memory usage of cleaned DataFrames when this script is run directly
    files = [accounts_csv, customers_csv, transactions_csv]
    data_frames = parallel_load_data(files)

    # Apply memory optimization and save optimized DataFrames to Parquet format
    optimized_dfs = [optimize_memory_usage(df) for df in data_frames]

    for i, df in enumerate(optimized_dfs):
        df.to_parquet(f'{staging_dir}/file_{i}_cleaned.parquet', index=False)