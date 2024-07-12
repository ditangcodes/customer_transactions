import csv
import pandas as pd
import os

#Task one Extracting the data from csv files
#Read CSV Files
def read_csv_file(file_paths):
    data_frames = []
    for x in file_paths:
        try:
            df= pd.read_csv(x)
        #Remove white spaces from column
            df.columns = df.columns.str.strip()
            data_frames.append(df)
        except FileNotFoundError:
            print(f"File not found: {x}")
    return data_frames
print("csv file successfully read")

# Testing
csv_files = ['../landing/accounts.csv', "../landing/customers.csv","../landing/transactions.csv"]
loaded_data = read_csv_file(csv_files)

if loaded_data:
    for i, df in enumerate(loaded_data):
        print(f"DataFrame {i+1}:")
        print(df.head()) 
else:
    print("No data loaded.")
print("CSV file succesfully loaded")

def comparing_columns(orginal_df, cleaned_df):
    orginal_rows, original_cols = orginal_df.shape
    cleaned_rows, cleaned_cols = cleaned_df.shape
    return orginal_rows == cleaned_rows and original_cols == cleaned_cols

if loaded_data:
    for x, df in enumerate(loaded_data):
        print(f"DataFrame {i+1}:")
        print(df.head())
        # Add consistency check
        if comparing_columns(df, loaded_data[0]):
            print("Consistency check passed.")
        else:
            print("Consistency check failed.")
else:
    print("No data loaded.")


# Migrate CSV into Staging Folder
def convert_file_into_parquet(data_frames, file_location):
    for i, df in enumerate(data_frames):
        original_csv = os.path.basename(csv_files[i])
        parquet_file_path = f"{file_location}/{original_csv}_cleaned.parquet"
        df.to_parquet(parquet_file_path, index=False)
        print(f"Saved DataFrame {i+1} to {parquet_file_path}")

#Testing
csv_files = ['../landing/accounts.csv', '../landing/customers.csv','../landing/transactions.csv']
loaded_data = read_csv_file(csv_files)

output_directory = '../staging'  # Specify the directory where Parquet files will be saved
convert_file_into_parquet(loaded_data, output_directory)
print("Parquet files successfully imported")