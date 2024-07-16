import csv
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

def aggregate_transactions_by_acc(file1, file2, file3):
    #Load Parquet Files
    transactions_1 = pd.read_parquet(file1)
    transactions_2 = pd.read_parquet(file2)
    customers = pd.read_parquet(file3)

    common_key = 'account_id'

    # Merge DataFrames based on 'Account_id'
    merged_df = pd.merge(transactions_1, transactions_2, on=common_key)

    # Merge with customers DataFrame using 'Customer_id'
    merged_df = pd.merge(merged_df, customers, left_on='customer_id', right_on='customer_id')

    # Group by Account Type , Customer ID and calculate sum of transaction amounts
    aggregated_file = merged_df.groupby(['customer_id','account_type','transaction_type'])['amount'].sum().reset_index()
    
    # Save to Parquet file 
    aggregated_file.to_parquet("../transformed/aggregated_transactions_data.parquet", index=False)
    print("Aggregated data saved to 'transformed/aggregated_transactions_data.parquet'")

    return aggregated_file
   

#Testing
file1 = "../staging/accounts.csv_cleaned.parquet"
file2 = "../staging/transactions.csv_cleaned.parquet"
file3 = "../staging/customers.csv_cleaned.parquet"
aggregated_data = aggregate_transactions_by_acc(file1, file2, file3)

print("Aggregated data by account type:")
print(aggregated_data)

# Function to identify high-risk transactions
def identify_high_risk_transactions(transactions_df, amount=10000):
    # Identify high-value transactions that exceed the specified amount
    high_transactions = transactions_df[transactions_df['amount'] > amount]

    # Group by Account ID and calculate the sum of transaction amounts
    grouped_data = high_transactions.groupby('account_id')['amount'].sum().reset_index()

    # Save to Parquet file 
    output_path = "../transformed/high_risk_transactions.parquet"
    grouped_data.to_parquet(output_path, index=False)
    print(f"Aggregated data saved to '{output_path}'")

    return grouped_data

# Example usage
transactions_df = pd.read_parquet("../staging/transactions.csv_cleaned.parquet")
high_risk_df = identify_high_risk_transactions(transactions_df)

print("High-risk transactions:")
print(high_risk_df)

# Function to calculate cumulative balances
def cumulative_balances(accounts_file, transaction_file):
    # Read data
    accounts_df = pd.read_parquet(accounts_file)
    transactions_df = pd.read_parquet(transaction_file)

    # Merge accounts and transactions based on account_id
    merged_df = pd.merge(transactions_df, accounts_df, on='account_id', how='left')

    # Calculate cumulative balances
    merged_df['cumulative_balance'] = merged_df.groupby('account_id')['amount'].cumsum()

    # Select relevant columns
    cumulative_balances_df = merged_df[['account_id', 'cumulative_balance']].drop_duplicates()

    # Save to Parquet file 
    output_path = "../transformed/cumulative_balances.parquet"
    cumulative_balances_df.to_parquet(output_path, index=False)
    print(f"Cumulative balances saved to '{output_path}'")

    return cumulative_balances_df.set_index('account_id')['cumulative_balance'].to_dict()

# Example usage
if __name__ == "__main__":
    accounts_csv = "../staging/accounts.csv_cleaned.parquet"
    transactions_csv = "../staging/transactions.csv_cleaned.parquet"

    cumulative_balances_dict = cumulative_balances(accounts_csv, transactions_csv)

    # Print cumulative balances for each account
    for account_id, balance in cumulative_balances_dict.items():
        print(f"Account {account_id}: Cumulative Balance = £{balance:.2f}")

# Function to calculate monthly transactions        
def monthly_transactions(transactions_file):
    # Load transactions DataFrame
    transactions_df = pd.read_parquet(transactions_file)

    # Clean date column (remove extra spaces and convert to datetime)
    transactions_df['date'] = transactions_df['date'].str.strip()
    transactions_df['date'] = pd.to_datetime(transactions_df['date'], format="%Y-%m-%d")

    # Calculate month and year
    transactions_df['month'] = transactions_df['date'].dt.to_period('M')

    # Aggregate monthly data
    agg_funcs = {'amount': ['sum', 'mean', 'count']}
    transaction_months = transactions_df.groupby('month').agg(agg_funcs).reset_index()

    # Rename columns
    transaction_months.columns = ['Month', 'Total Amount', 'Average Amount', 'Transaction Count']

    # Round average amount to 2 decimal places
    transaction_months['Average Amount'] = transaction_months['Average Amount'].round(2)

    # Save results to a Parquet file
    output_path = "../transformed/monthly_transactions.parquet"
    transaction_months.to_parquet(output_path, index=False)
    print(f"Monthly transaction data saved to '{output_path}'")

    return transaction_months

# Testing
transactions_file = "../staging/transactions.csv_cleaned.parquet"
monthly_metrics_df = monthly_transactions(transactions_file)

print("Monthly Transaction Summary:")
print(monthly_metrics_df)

# Function to calculate average transaction per customer
def average_transaction_per_customer(transactions_file, customers_file):
    # Load transactions and customers DataFrames
    transactions_df = pd.read_parquet(transactions_file)
    customers_df = pd.read_parquet(customers_file)

    # Merge files together using a left Join
    merged_df = pd.merge(transactions_df, customers_df, how='left', left_on='account_id', right_on='customer_id')

    # Group by customer ID and calculate the average transaction amount per customer
    avg_transaction_per_customer = merged_df.groupby('customer_id')['amount'].mean().reset_index()

    # Rename Columns
    avg_transaction_per_customer.columns = ['Customer ID', 'Average Transaction Amount']
    
    # Rounding the Average Amount to 2 decimal places
    avg_transaction_per_customer['Average Transaction Amount'] = avg_transaction_per_customer['Average Transaction Amount'].round(2)

    # Adding results as a parquet file
    output_path = "../transformed/average_transaction_per_customer.parquet"
    avg_transaction_per_customer.to_parquet(output_path, index=False)
    print(f"Average transaction per customer data saved to '{output_path}'")

    return avg_transaction_per_customer

# Testing
transactions_csv = "../staging/transactions.csv_cleaned.parquet"
customers_csv = "../staging/customers.csv_cleaned.parquet"
avg_transaction_per_customer = average_transaction_per_customer(transactions_csv, customers_csv)
print("Average Transaction Values per Customer:")
print(avg_transaction_per_customer)

# Function to train a customer churn model
def customer_churn(transactions_file, accounts_file, customers_file, threshold_days_list=400):
    transactions_df = pd.read_parquet(transactions_file)
    accounts_df = pd.read_parquet(accounts_file)
    customers_df = pd.read_parquet(customers_file)

    # Print column names to verify
    print("Transactions columns:", transactions_df.columns)
    print("Accounts columns:", accounts_df.columns)
    print("Customers columns:", customers_df.columns)

    # Merge Transactions and Accounts on 'account_id'
    merged_df = pd.merge(transactions_df, accounts_df, how='left', on='account_id')
    print("After merging transactions and accounts:")
    print(merged_df.head())

    # Merge the result with Customers on 'customer_id'
    merged_df = pd.merge(merged_df, customers_df, how='left', left_on='customer_id', right_on='customer_id')
    print("After merging with customers:")
    print(merged_df.head())

    last_transaction_df = merged_df.groupby('account_id')['date'].max().reset_index()
    last_transaction_df.rename(columns={'date': 'last_transaction_date'}, inplace=True)

    merged_df = pd.merge(merged_df, last_transaction_df, on='account_id', how='left')
    merged_df['last_transaction_date'] = pd.to_datetime(merged_df['last_transaction_date'], format='%Y-%m-%d')
    merged_df['join_date'] = pd.to_datetime(merged_df['join_date'], format='%Y-%m-%d')

    # Calculate the difference (in days) between last transaction date and current date
    current_date = pd.to_datetime('today')
    merged_df['last_transaction_days_ago'] = (current_date - merged_df['last_transaction_date']).dt.days

    # Calculate the difference (in days) between join date and current date
    merged_df['join_days_ago'] = (current_date - merged_df['join_date']).dt.days

    # Handle NaN values
    merged_df['join_days_ago'].fillna(merged_df['join_days_ago'].mean(), inplace=True)

    # Initialize an empty DataFrame to store results
    result_df = pd.DataFrame()

    # Create a temporary DataFrame for each threshold
    temp_df = merged_df.copy()
    temp_df['churn'] = (temp_df['last_transaction_days_ago'] > threshold_days_list).astype(int)
    
    # Concatenate with the result DataFrame
    result_df = pd.concat([result_df, temp_df])

    # Check the distribution of the churn column
    churn_counts = result_df['churn'].value_counts()
    print("Churn distribution in the dataset:", churn_counts)

    # Ensure there are samples of both classes
    if len(churn_counts) < 2:
        raise ValueError("The data contains only one class for churn. Ensure your data has both churned and non-churned samples.")

    # Splitting the data into train and test sets
    x = result_df[['last_transaction_days_ago', 'join_days_ago']]
    y = result_df['churn']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train a logistic regression model
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # Make predictions
    y_pred = model.predict(x_test)

    # Evaluate model performance (optional)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    
    # Save the result DataFrame to the transformed folder
    transformed_dir = "../transformed"
    os.makedirs(transformed_dir, exist_ok=True)
    result_file_path = os.path.join(transformed_dir, "customer_churn_results.parquet")
    result_df.to_parquet(result_file_path, index=False, compression='snappy')
    print(f"Customer churn results saved to '{result_file_path}'")

    return model

# Create example data
example_accounts = pd.DataFrame({
    'account_id': [1, 2, 3, 4],
    'customer_id': [101, 102, 103, 104],
    'account_type': ['checking', 'savings', 'checking', 'savings'],
    'balance': [500, 1000, 1500, 2000],
    'created_date': ['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01']
})

example_customers = pd.DataFrame({
    'customer_id': [101, 102, 103, 104],
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'email': ['alice@example.com', 'bob@example.com', 'charlie@example.com', 'david@example.com'],
    'join_date': ['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01']
})

example_transactions = pd.DataFrame({
    'transaction_id': [1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008],
    'account_id': [1, 1, 2, 2, 3, 3, 4, 4],
    'date': ['2024-01-01', '2024-01-05', '2024-02-01', '2024-02-05', '2023-12-01', '2023-12-05', '2021-01-01', '2021-01-05'],
    'amount': [100, 200, 150, 250, 300, 400, 500, 600],
    'transaction_type': ['debit', 'credit', 'debit', 'credit', 'debit', 'credit', 'debit', 'credit']
})

# Save example data to parquet files in the transformed directory
example_accounts.to_parquet("../transformed/accounts_churned.parquet", index=False)
example_customers.to_parquet("../transformed/customers_churned.parquet", index=False)
example_transactions.to_parquet("../transformed/transactions_churned.parquet", index=False)

# File paths
transactions_file = "../transformed/transactions_churned.parquet"
accounts_file = "../transformed/accounts_churned.parquet"
customers_file = "../transformed/customers_churned.parquet"

# Train the customer churn model
trained_model = customer_churn(transactions_file, accounts_file, customers_file)

print("Churn detection model trained successfully!")
