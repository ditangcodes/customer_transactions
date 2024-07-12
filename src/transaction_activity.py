import csv
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

def aggregate_transactions_by_acc(file1, file2, file3):
    # Load Parquet Files
    transactions_1 = pd.read_parquet(file1, columns=['customer_id', 'account_id', 'amount'])
    transactions_2 = pd.read_parquet(file2, columns=['customer_id', 'account_id', 'amount'])
    customers = pd.read_parquet(file3, columns=['customer_id', 'account_type'])

    # Merge DataFrames
    merged_df = pd.concat([transactions_1, transactions_2])
    merged_df = pd.merge(merged_df, customers, on='customer_id')

    # Group by Account Type, Customer ID, and calculate sum of transaction amounts
    aggregated_file = merged_df.groupby(['customer_id', 'account_type', 'transaction_type'])['amount'].sum().reset_index()

    # Save to Parquet file (optimized for Azure Synapse Analytics)
    aggregated_file.to_parquet("../transformed/aggregated_transactions_data.parquet", index=False, compression='snappy')
    print("Aggregated data saved to 'transformed/aggregated_transactions_data.parquet'")

    return aggregated_file

#Testing
file1 = "../staging/accounts.csv_cleaned.parquet"
file2 = "../staging/transactions.csv_cleaned.parquet"
file3 = "../staging/customers.csv_cleaned.parquet"
aggregated_data = aggregate_transactions_by_acc(file1, file2, file3)

print("Aggregated data by account type:")
print(aggregated_data)

def identify_high_risk_transactions(transactions_df, amount=10000):
    # Identify high-value transactions that exceed the specified amount
    high_transactions = transactions_df[transactions_df['amount'] > amount]

    # Group by Account ID and calculate the sum of transaction amounts
    grouped_data = high_transactions.groupby('account_id')['amount'].sum().reset_index()

    # Save to Parquet file (optimized for Azure Synapse Analytics)
    output_path = "../transformed/high_risk_transactions.parquet"
    grouped_data.to_parquet(output_path, index=False)
    print(f"Aggregated data saved to '{output_path}'")

    return grouped_data

# Example usage
transactions_df = pd.read_parquet("../staging/transactions.csv_cleaned.parquet")
high_risk_df = identify_high_risk_transactions(transactions_df)

print("High-risk transactions:")
print(high_risk_df)

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
    avg_transaction_per_customer.to_parquet("../transformed/average_transaction_per_customer.parquet", index=False)

    return avg_transaction_per_customer
#Testing
transactions_csv = "../staging/transactions.csv_cleaned.parquet"
customers_csv = "../staging/customers.csv_cleaned.parquet"
avg_transaction_per_customer = average_transaction_per_customer(transactions_csv, customers_csv)
print("Average Transaction Values per Customer:")
print(avg_transaction_per_customer)


def customer_churn(transactions_file, customers_file, threshold_days_list=[30, 60, 90]):
    # Load Transactions and accounts DF
    transactions_df = pd.read_parquet(transactions_file)
    customers_df = pd.read_parquet(customers_file)

    # Merge files together using a left join
    merged_df = pd.merge(transactions_df, customers_df, how='left', left_on='account_id', right_on='customer_id')

    # Calculate the last transaction date for each account
    last_transaction_df = merged_df.groupby('account_id')['date'].max().reset_index()
    last_transaction_df.rename(columns={'date': 'last_transaction_date'}, inplace=True)

    # Merge the last transaction date back to the main DataFrame
    merged_df = pd.merge(merged_df, last_transaction_df, on='account_id', how='left')

    # Strip any extra spaces from 'last_transaction_date' column
    merged_df['last_transaction_date'] = merged_df['last_transaction_date'].str.strip()

    # Convert 'last_transaction_date' to Timestamp
    merged_df['last_transaction_date'] = pd.to_datetime(merged_df['last_transaction_date'], format='%Y-%m-%d')

    # Convert 'join_date' to Timestamp
    merged_df['join_date'] = pd.to_datetime(merged_df['join_date'], format='%Y-%m-%d')

    # Calculate the difference (in days) between last transaction date and current date
    current_date = pd.to_datetime('today')
    merged_df['last_transaction_days_ago'] = (current_date - merged_df['last_transaction_date']).dt.days

    # Calculate the difference (in days) between join date and current date
    merged_df['join_days_ago'] = (current_date - merged_df['join_date']).dt.days

    # Initialize an empty DataFrame to store results
    result_df = pd.DataFrame()

    for x in threshold_days_list:
        # Create a temporary DataFrame for each threshold
        temp_df = merged_df.copy()
        temp_df['churn'] = (temp_df['last_transaction_days_ago'] > x).astype(int)

        # Concatenate with the result DataFrame
        result_df = pd.concat([result_df, temp_df])

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

    return model

transactions_file = "../staging/transactions.csv_cleaned.parquet"
customers_file = "../staging/customers.csv_cleaned.parquet"
trained_model = customer_churn(transactions_file, customers_file)

print("Churn detection model trained successfully!")
